"""Coordinator session against a fake Flower Grid that calls the real ClientApp handlers."""

import pytest
import torch

from fake_grid import IDS, FakeGrid
from coordinator.checkpoint_manager.manager import CheckpointManager
from coordinator.flower_app import protocol
from coordinator.flower_app.protocol import MODEL_CONFIG
from coordinator.flower_app.session import SessionError, register_institutions, run_session, run_turn
from coordinator.relay.relay import Relay
from coordinator.relay.transport import RelayServer
from coordinator.scheduling.schedule import Turn, build_schedule
from split_learning.backbone import build_backbone


@pytest.fixture
def manager(state_dir):
    manager = CheckpointManager(state_dir / "coordinator" / "checkpoints")
    manager.initialize(build_backbone(MODEL_CONFIG).state_dict())
    return manager


@pytest.fixture
def relay_server():
    with RelayServer(Relay(), port=0) as server:
        yield server


def quiet(_text):
    pass


def turn_args(grid, manager, relay_server):
    nodes = register_institutions(grid, len(grid.contexts), timeout=5, log=quiet)
    return nodes, manager, relay_server.relay, relay_server.url


def test_registration_maps_institutions_to_nodes():
    grid = FakeGrid(IDS)
    nodes = register_institutions(grid, 4, timeout=5, log=quiet)
    assert nodes == {institution_id: grid.node_of(institution_id) for institution_id in IDS}


def test_registration_times_out_when_institutions_are_missing():
    with pytest.raises(SessionError, match="only 2 of 4"):
        register_institutions(FakeGrid(IDS[:2]), 4, timeout=0.05, poll_interval=0.01, log=quiet)


def test_registration_rejects_duplicate_institution_ids():
    with pytest.raises(SessionError, match="two nodes"):
        register_institutions(FakeGrid(["inst-1", "inst-1"]), 2, timeout=5, log=quiet)


def test_full_session_with_four_institutions_and_two_rounds(manager, relay_server, state_dir):
    grid = FakeGrid(IDS)
    nodes, manager, relay, url = turn_args(grid, manager, relay_server)
    initial, _ = manager.latest()

    results = run_session(
        grid, nodes, build_schedule(IDS, 2), manager, relay, url, num_batches=2, turn_timeout=30, log=quiet
    )

    assert [r.checkpoint.version for r in results] == list(range(1, 9))
    assert [r.checkpoint.created_by for r in results] == [t.compute_peer_id for t in build_schedule(IDS, 2)]
    assert all(r.num_batches == 2 for r in results)
    assert manager.latest_version == 8
    final, _ = manager.latest()
    assert any(not torch.equal(initial[key], final[key]) for key in initial)
    for institution_id in IDS:  # Head and Tail stay with each institution
        assert (state_dir / "institutions" / f"{institution_id}.pt").is_file()
    assert relay.active_turn is None


@pytest.mark.parametrize("failing_role", [protocol.TRAIN_AS_PEER, protocol.TRAIN_AS_OWNER])
def test_failed_turn_does_not_change_the_backbone(manager, relay_server, failing_role):
    turn = Turn(1, "inst-1", "inst-2")
    failing_member = turn.compute_peer_id if failing_role == protocol.TRAIN_AS_PEER else turn.data_owner_id
    grid = FakeGrid(IDS, failing={(failing_member, failing_role)})
    nodes, manager, relay, url = turn_args(grid, manager, relay_server)

    with pytest.raises(SessionError, match="failed"):
        run_turn(grid, nodes, turn, manager, relay, url, num_batches=2, turn_timeout=2, log=quiet)
    assert manager.latest_version == 0
    assert relay.active_turn is None


def test_missing_reply_does_not_change_the_backbone(manager, relay_server):
    grid = FakeGrid(IDS)
    nodes, manager, relay, url = turn_args(grid, manager, relay_server)
    grid.silent = {"inst-2"}

    with pytest.raises(SessionError, match="did not reply"):
        run_turn(grid, nodes, Turn(1, "inst-1", "inst-2"), manager, relay, url, 2, turn_timeout=2, log=quiet)
    assert manager.latest_version == 0


def test_turn_with_unregistered_institution_is_rejected(manager, relay_server):
    grid = FakeGrid(IDS[:2])
    nodes, manager, relay, url = turn_args(grid, manager, relay_server)
    with pytest.raises(SessionError, match="not registered"):
        run_turn(grid, nodes, Turn(1, "inst-1", "inst-3"), manager, relay, url, 2, 2, log=quiet)


def test_handler_rejects_a_turn_meant_for_another_institution(manager, relay_server):
    grid = FakeGrid(IDS)
    nodes, manager, relay, url = turn_args(grid, manager, relay_server)
    nodes["inst-2"], nodes["inst-3"] = nodes["inst-3"], nodes["inst-2"]  # coordinator misroutes

    with pytest.raises(SessionError, match="meant for"):
        run_turn(grid, nodes, Turn(1, "inst-1", "inst-2"), manager, relay, url, 2, turn_timeout=2, log=quiet)
    assert manager.latest_version == 0
