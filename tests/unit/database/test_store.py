import sqlite3

import pytest

from database.store import MetadataStore, connect_readonly, display_name

TABLES = {
    "institutions",
    "training_sessions",
    "compute_assignments",
    "model_checkpoints",
    "training_events",
    "noise_configurations",
}


@pytest.fixture
def store(tmp_path):
    store = MetadataStore(tmp_path / "medweave.db")
    yield store
    store.close()


def test_creates_every_spec_table(store):
    names = {row["name"] for row in store.query("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert TABLES <= names


def test_no_column_can_hold_tensor_data(store):
    """SPEC §3.1: no images, labels or activations in the database."""
    for table in TABLES:
        for column in store.query(f"PRAGMA table_info({table})"):
            assert column["type"].upper() != "BLOB", f"{table}.{column['name']}"
            if not column["name"].endswith("_at"):  # timestamps like first_activation_at are fine
                assert not any(word in column["name"] for word in ("image", "label", "activation", "gradient"))


def test_uses_wal_so_the_dashboard_can_read_while_training(store):
    assert store.query("PRAGMA journal_mode")[0][0] == "wal"


def test_reopening_keeps_data(tmp_path):
    path = tmp_path / "medweave.db"
    first = MetadataStore(path)
    first.register_institution("inst-1")
    first.close()
    second = MetadataStore(path)
    assert second.query("SELECT institution_id FROM institutions")[0][0] == "inst-1"
    second.close()


def test_registering_twice_updates_instead_of_duplicating(store):
    store.register_institution("inst-1", "node:1")
    store.set_institution_status("inst-1", "offline")
    store.register_institution("inst-1", "node:2")
    rows = store.query("SELECT client_address, status FROM institutions")
    assert [tuple(r) for r in rows] == [("node:2", "online")]


def test_assignment_lifecycle(store):
    for iid in ("inst-1", "inst-2"):
        store.register_institution(iid)
    session = store.start_session(num_rounds=1, total_turns=4)
    assignment = store.start_assignment(session, 1, "inst-1", "inst-2", backbone_version=0, batches_total=3)
    store.mark_first_activation(assignment)
    store.mark_first_activation(assignment)  # second call keeps the first timestamp
    for _ in range(3):
        store.record_batch_done(assignment)
    store.complete_assignment(assignment, loss=0.5, accuracy=0.75)

    row = store.query("SELECT * FROM compute_assignments")[0]
    assert (row["status"], row["batches_done"], row["loss"], row["accuracy"]) == ("completed", 3, 0.5, 0.75)
    assert row["first_activation_at"] is not None and row["completed_at"] is not None


def test_session_finish_clears_the_current_turn(store):
    for iid in ("inst-1", "inst-2"):
        store.register_institution(iid)
    session = store.start_session(1, 4)
    store.set_current_turn(session, 1, "inst-1", "inst-2")
    store.finish_session(session, "completed")
    row = store.query("SELECT * FROM training_sessions")[0]
    assert (row["status"], row["current_institution_id"], row["current_peer_id"]) == ("completed", None, None)
    assert row["completed_at"] is not None


@pytest.mark.parametrize(
    ("sql", "params"),
    [
        ("UPDATE institutions SET status = ? WHERE institution_id = 'inst-1'", ("asleep",)),
        ("INSERT INTO training_events (session_id, event_type, message, created_at) VALUES ('s', ?, 'm', 't')", ("party",)),
    ],
)
def test_invalid_values_are_rejected(store, sql, params):
    store.register_institution("inst-1")
    store.start_session(1, 1)
    with pytest.raises(sqlite3.IntegrityError):
        store._write(sql, params)


def test_readonly_connection_cannot_write(store):
    store.register_institution("inst-1")
    reader = connect_readonly(store.path)
    assert reader.execute("SELECT COUNT(*) FROM institutions").fetchone()[0] == 1
    with pytest.raises(sqlite3.OperationalError):
        reader.execute("DELETE FROM institutions")
    reader.close()


def test_readonly_connection_to_missing_file_raises(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        connect_readonly(tmp_path / "missing.db")


@pytest.mark.parametrize(("iid", "name"), [("inst-3", "Institution 3"), ("hospital-a", "hospital-a")])
def test_display_name(iid, name):
    assert display_name(iid) == name
