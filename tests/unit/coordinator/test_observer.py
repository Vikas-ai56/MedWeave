"""A full session recorded through DatabaseObserver, read back as the dashboard would."""

import pytest

from coordinator.checkpoint_manager.manager import CheckpointManager
from coordinator.flower_app.observer import DatabaseObserver
from coordinator.flower_app.protocol import MODEL_CONFIG
from coordinator.flower_app.session import SessionError, register_institutions, run_session
from coordinator.relay.relay import Relay
from coordinator.relay.transport import RelayServer
from coordinator.scheduling.schedule import build_schedule
from dashboard.backend.snapshot import build_snapshot
from database.store import MetadataStore, connect_readonly
from fake_grid import IDS, FakeGrid
from split_learning.backbone import build_backbone


def quiet(_text):
    pass


@pytest.fixture
def store(state_dir):
    store = MetadataStore(state_dir / "medweave.db")
    yield store
    store.close()


def run(grid, store, state_dir, num_rounds=1, num_batches=2):
    observer = DatabaseObserver(store)
    nodes = register_institutions(grid, len(grid.contexts), timeout=5, log=quiet, observer=observer)
    schedule = build_schedule(sorted(nodes), num_rounds)
    observer.session_started(schedule, num_rounds)
    manager = CheckpointManager(state_dir / "checkpoints")
    observer.checkpoint_saved(manager.initialize(build_backbone(MODEL_CONFIG).state_dict()))
    relay = Relay(on_message=observer.message_relayed)
    with RelayServer(relay, port=0) as server:
        run_session(grid, nodes, schedule, manager, relay, server.url, num_batches, 10, log=quiet, observer=observer)
    return observer


def snapshot(store):
    db = connect_readonly(store.path)
    try:
        return build_snapshot(db)
    finally:
        db.close()


def test_completed_session_is_fully_recorded(store, state_dir):
    run(FakeGrid(IDS), store, state_dir, num_rounds=2, num_batches=3)
    view = snapshot(store)

    session = view["session"]
    assert session["status"] == "completed"
    assert (session["completed_turns"], session["total_turns"], session["num_rounds"]) == (8, 8, 2)
    assert session["current_owner"] is None and session["current_peer"] is None

    turns = view["turns"]
    assert [(t["owner"], t["peer"]) for t in turns[:4]] == [
        ("inst-1", "inst-2"), ("inst-2", "inst-3"), ("inst-3", "inst-4"), ("inst-4", "inst-1"),
    ]
    assert [t["backbone_version"] for t in turns] == list(range(8))
    assert all(t["status"] == "completed" and t["loss"] is not None for t in turns)

    batches = store.query("SELECT batches_done, batches_total FROM compute_assignments")
    assert all(row["batches_done"] == row["batches_total"] == 3 for row in batches)  # counted from relay traffic

    assert view["model"]["backbone_version"] == 8
    assert view["model"]["completed_contributions"] == 8
    for institution in view["institutions"]:
        assert institution["status"] == "online"
        assert institution["role"] == "idle"
        assert (institution["contributions"], institution["computations"]) == (2, 2)
        assert institution["last_contribution"]["round"] == 2

    assert all(step["state"] in ("done", "skipped") for step in view["current_turn"]["steps"])
    assert {e["type"] for e in view["events"]} >= {"assignment", "training", "checkpoint", "completion"}
    registrations = store.query("SELECT COUNT(*) FROM training_events WHERE event_type = 'registration'")
    assert registrations[0][0] == 4


def test_noise_step_shows_as_skipped_until_noise_exists(store, state_dir):
    run(FakeGrid(IDS), store, state_dir)
    steps = {s["key"]: s for s in snapshot(store)["current_turn"]["steps"]}
    assert steps["noise"]["state"] == "skipped"
    assert snapshot(store)["noise"] is None


def test_failed_turn_is_recorded_and_stops_the_session(store, state_dir):
    grid = FakeGrid(IDS, failing={("inst-2", "train.peer")})
    with pytest.raises(SessionError):
        run(grid, store, state_dir)
    view = snapshot(store)

    assert view["session"]["status"] == "failed"
    assert [t["status"] for t in view["turns"]] == ["failed"]
    assert "simulated crash" in view["turns"][0]["error"]
    assert "failed" in [s["state"] for s in view["current_turn"]["steps"]]
    assert view["model"]["backbone_version"] == 0
    assert view["events"][0]["type"] == "failure"
    assert store.query("SELECT COUNT(*) FROM training_events WHERE message LIKE 'Session stopped%'")[0][0] == 1


def test_offline_institution_is_marked_before_the_turn(store, state_dir):
    grid = FakeGrid(IDS)
    observer = DatabaseObserver(store)
    nodes = register_institutions(grid, 4, timeout=5, log=quiet, observer=observer)
    observer.session_started(build_schedule(IDS, 1), 1)
    observer.connection_checked({iid: iid != "inst-3" for iid in nodes})
    statuses = {i["id"]: i["status"] for i in snapshot(store)["institutions"]}
    assert statuses == {"inst-1": "online", "inst-2": "online", "inst-3": "offline", "inst-4": "online"}


def test_noise_configuration_reaches_the_dashboard(store, state_dir):
    observer = DatabaseObserver(store)
    observer.session_started(build_schedule(IDS, 1), 1)
    assert snapshot(store)["noise"] is None

    observer.noise_configured("gaussian", 0.25)
    view = snapshot(store)
    assert view["noise"] == {"mechanism": "gaussian", "noise_scale": 0.25}
    assert any("activation noise" in event["message"] for event in view["events"])
