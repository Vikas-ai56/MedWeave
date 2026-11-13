"""Coordinator session against a fake Flower Grid that calls the real ClientApp handlers."""

import time

import pytest
import torch

from fake_grid import IDS, FakeGrid
from coordinator.checkpoint_manager.manager import CheckpointManager
from coordinator.flower_app import protocol
from coordinator.flower_app.protocol import MODEL_CONFIG
from coordinator.flower_app.session import (
    EvaluationSettings,
    SessionError,
    register_institutions,
    run_session,
    run_turn,
)
from coordinator.relay.relay import Relay
from coordinator.relay.transport import RelayServer
from coordinator.scheduling.schedule import Turn, build_schedule
from split_learning.backbone import build_backbone
from tracking import NullTracker


class RecordingTracker(NullTracker):
    def __init__(self):
        self.metrics = []

    def log_metrics(self, metrics, step):
        self.metrics.append((step, dict(metrics)))


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
    tracker = RecordingTracker()

    session = run_session(
        grid,
        nodes,
        build_schedule(IDS, 2),
        manager,
        relay,
        url,
        num_batches=2,
        turn_timeout=30,
        log=quiet,
        evaluation=EvaluationSettings("val", batch_size=5, timeout=30),
        tracker=tracker,
    )
    results = session.turns

    assert [r.checkpoint.version for r in results] == list(range(1, 9))
    assert [r.checkpoint.created_by for r in results] == [t.compute_peer_id for t in build_schedule(IDS, 2)]
    assert all(r.num_batches == 2 for r in results)
    assert all(r.noise_std == 0.1 and r.activation_change_pct > 0 for r in results)
    assert manager.latest_version == 8
    final, _ = manager.latest()
    assert any(not torch.equal(initial[key], final[key]) for key in initial)
    for institution_id in IDS:  # Head and Tail stay with each institution
        assert (state_dir / "institutions" / f"{institution_id}.pt").is_file()
    assert relay.active_turn is None
    assert [(e.round_number, e.backbone_version) for e in session.evaluations] == [(1, 4), (2, 8)]
    for evaluation in session.evaluations:
        assert set(evaluation.accuracy) == set(IDS) and evaluation.num_samples == 12
        assert all(0.0 <= accuracy <= 1.0 for accuracy in evaluation.accuracy.values())
    turn_logs = [(step, m) for step, m in tracker.metrics if "train_loss" in m]
    round_logs = [(step, m) for step, m in tracker.metrics if "val_accuracy_mean" in m]
    assert [step for step, _ in turn_logs] == list(range(1, 9))
    assert [step for step, _ in round_logs] == [1, 2]
    assert {f"val_accuracy/{institution_id}" for institution_id in IDS} <= set(round_logs[0][1])


class TurnRecordingGrid(FakeGrid):
    """Also keeps the TurnInfo of every training message, keyed by message type."""

    def __init__(self, institution_ids):
        super().__init__(institution_ids)
        self.turns = []

    def send_and_receive(self, messages, *, timeout=None):
        messages = list(messages)
        self.turns += [
            (m.metadata.message_type, protocol.TurnInfo.from_record(m.content[protocol.TURN]))
            for m in messages
            if protocol.TURN in m.content
        ]
        return super().send_and_receive(messages, timeout=timeout)


def test_each_turn_trains_at_its_scheduled_learning_rate(manager, relay_server, state_dir):
    grid = TurnRecordingGrid(IDS)
    nodes, manager, relay, url = turn_args(grid, manager, relay_server)
    rates = [8e-3, 7e-3, 6e-3, 5e-3, 4e-3, 3e-3, 2e-3, 1e-3]

    session = run_session(
        grid, nodes, build_schedule(IDS, 2), manager, relay, url, 1, 30, log=quiet, learning_rates=rates
    )

    assert [result.learning_rate for result in session.turns] == rates
    peer_rates = [turn.learning_rate for kind, turn in grid.turns if kind == protocol.TRAIN_AS_PEER]
    owner_rates = [turn.learning_rate for kind, turn in grid.turns if kind == protocol.TRAIN_AS_OWNER]
    assert peer_rates == owner_rates == rates
    # inst-1 trained in turns 0 and 4: its saved optimizers carry turn 4's rate, not the one restored from turn 0
    saved = torch.load(state_dir / "institutions" / "inst-1.pt", weights_only=True)
    assert saved["head_optimizer"]["param_groups"][0]["lr"] == rates[4]
    assert saved["tail_optimizer"]["param_groups"][0]["lr"] == rates[4]


def test_learning_rates_must_match_the_schedule(manager, relay_server):
    grid = FakeGrid(IDS)
    nodes, manager, relay, url = turn_args(grid, manager, relay_server)
    with pytest.raises(ValueError, match="one learning rate per turn"):
        run_session(grid, nodes, build_schedule(IDS, 1), manager, relay, url, 1, 30, log=quiet, learning_rates=[1e-3])


def test_evaluation_failure_names_the_institution(manager, relay_server):
    grid = FakeGrid(IDS, failing={("inst-3", protocol.EVALUATE_HELD_OUT)})
    nodes, manager, relay, url = turn_args(grid, manager, relay_server)

    with pytest.raises(SessionError, match="round 1 evaluation failed: inst-3 failed"):
        run_session(
            grid, nodes, build_schedule(IDS, 1), manager, relay, url, 1, 30, log=quiet,
            evaluation=EvaluationSettings("val", batch_size=5, timeout=10),
        )
    assert manager.latest_version == 4


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


def test_owner_without_its_data_fails_the_turn_without_waiting_for_the_timeout(manager, relay_server):
    grid = FakeGrid(IDS)
    grid.contexts[grid.node_of("inst-1")].run_config["partition-seed"] = 5
    nodes, manager, relay, url = turn_args(grid, manager, relay_server)

    started = time.monotonic()
    with pytest.raises(SessionError, match="inst-1 failed: .*no partition manifest"):
        run_turn(grid, nodes, Turn(1, "inst-1", "inst-2"), manager, relay, url, 2, turn_timeout=20, log=quiet)
    assert time.monotonic() - started < 10
    assert manager.latest_version == 0
