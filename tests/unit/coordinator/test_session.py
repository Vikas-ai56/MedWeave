"""Coordinator session against a fake Flower Grid that calls the real ClientApp handlers."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from flwr.app import Context, Error, Message, RecordDict
from flwr.supercore.task_identity import TaskIdentity

from coordinator.checkpoint_manager.manager import CheckpointManager
from coordinator.flower_app import client_app, protocol
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

IDS = ["inst-1", "inst-2", "inst-3", "inst-4"]
RUN_CONFIG = {"optimizer": "adam", "learning-rate": 0.001, "partition-seed": 0, "batch-size": 4, "noise-std": 0.1, "noise-seed": 0}
HANDLERS = {
    protocol.REGISTER: client_app.register,
    protocol.TRAIN_AS_PEER: client_app.compute_for_peer,
    protocol.TRAIN_AS_OWNER: client_app.train_own_data,
    protocol.EVALUATE_HELD_OUT: client_app.evaluate_locally,
}


class RecordingTracker(NullTracker):
    def __init__(self):
        self.metrics = []

    def log_metrics(self, metrics, step):
        self.metrics.append((step, dict(metrics)))


class FakeGrid:
    """Delivers each message to the ClientApp handler of its node, concurrently."""

    def __init__(self, institution_ids, failing=(), silent=()):
        self.contexts = {
            100 + index: Context(
                run_id=1,
                node_id=100 + index,
                node_config={protocol.INSTITUTION_ID: institution_id},
                state=RecordDict(),
                run_config=dict(RUN_CONFIG),
            )
            for index, institution_id in enumerate(institution_ids)
        }
        self.failing = set(failing)  # (institution_id, message_type) pairs that raise
        self.silent = set(silent)  # institution ids whose replies get lost

    def node_of(self, institution_id):
        return next(n for n, c in self.contexts.items() if c.node_config[protocol.INSTITUTION_ID] == institution_id)

    def get_node_ids(self):
        return list(self.contexts)

    def send_and_receive(self, messages, *, timeout=None):
        messages = list(messages)
        with ThreadPoolExecutor(max_workers=len(messages)) as pool:
            replies = list(pool.map(self._deliver, messages))
        return [reply for reply in replies if reply is not None]

    def _deliver(self, msg):
        context = self.contexts[msg.metadata.dst_node_id]
        institution_id = context.node_config[protocol.INSTITUTION_ID]
        try:
            if (institution_id, msg.metadata.message_type) in self.failing:
                raise RuntimeError("simulated crash")
            reply = HANDLERS[msg.metadata.message_type](msg, context)
        except Exception as error:  # Flower turns handler exceptions into error replies
            reply = Message(Error(code=1, reason=str(error)), reply_to=msg)
        return None if institution_id in self.silent else reply


@pytest.fixture(autouse=True)
def flower_task_identity():
    saved = (TaskIdentity._run_id, TaskIdentity._task_id, TaskIdentity._node_id)
    TaskIdentity.run_id, TaskIdentity.task_id, TaskIdentity.node_id = 1, 1, 0
    yield
    TaskIdentity._run_id, TaskIdentity._task_id, TaskIdentity._node_id = saved


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch, provisioned_data_dir):
    monkeypatch.setenv("MEDWEAVE_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("MEDWEAVE_DATA_DIR", str(provisioned_data_dir))
    return tmp_path


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
