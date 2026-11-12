import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import create_app
from database.store import MetadataStore


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "medweave.db"


@pytest.fixture
def store(db_path):
    store = MetadataStore(db_path)
    for iid in ("inst-1", "inst-2", "inst-3", "inst-4"):
        store.register_institution(iid)
    yield store
    store.close()


def client(db_path, frontend=None):
    return TestClient(create_app(db_path, frontend_dist=frontend))


def test_missing_database_gives_an_empty_dashboard(db_path):
    response = client(db_path).get("/api/dashboard")
    assert response.status_code == 200
    body = response.json()
    assert body["session"] is None and body["database_exists"] is False


def test_database_without_sessions_lists_institutions(store, db_path):
    body = client(db_path).get("/api/dashboard").json()
    assert body["session"] is None
    assert [i["name"] for i in body["institutions"]] == [f"Institution {n}" for n in range(1, 5)]


def test_live_turn_shows_roles_progress_and_steps(store, db_path):
    session = store.start_session(num_rounds=1, total_turns=4)
    store.set_current_turn(session, 1, "inst-1", "inst-2")
    assignment = store.start_assignment(session, 1, "inst-1", "inst-2", backbone_version=0, batches_total=5)
    store.mark_first_activation(assignment)
    store.record_batch_done(assignment)
    store.record_batch_done(assignment)

    body = client(db_path).get("/api/dashboard").json()
    roles = {i["id"]: i["role"] for i in body["institutions"]}
    assert roles == {"inst-1": "owner", "inst-2": "peer", "inst-3": "idle", "inst-4": "idle"}
    owner = next(i for i in body["institutions"] if i["id"] == "inst-1")
    assert owner["progress"] == {"done": 2, "total": 5}

    steps = {s["key"]: s["state"] for s in body["current_turn"]["steps"]}
    assert steps == {
        "selected": "done",
        "assigned": "done",
        "activations": "done",
        "noise": "skipped",
        "backbone": "active",
        "checkpoint": "pending",
        "completed": "pending",
    }


def test_noise_configuration_is_reported(store, db_path):
    session = store.start_session(1, 4)
    store.add_noise_configuration(session, "gaussian", 0.1)
    store.start_assignment(session, 1, "inst-1", "inst-2", 0, 5)
    body = client(db_path).get("/api/dashboard").json()
    assert body["noise"] == {"mechanism": "gaussian", "noise_scale": 0.1}
    assert next(s for s in body["current_turn"]["steps"] if s["key"] == "noise")["state"] == "pending"


def test_sessions_are_listed_newest_first_and_selectable(store, db_path):
    first = store.start_session(1, 4)
    second = store.start_session(2, 8)
    app = client(db_path)
    assert [s["session_id"] for s in app.get("/api/sessions").json()] == [second, first]
    assert app.get("/api/dashboard", params={"session_id": first}).json()["session"]["id"] == first
    assert app.get("/api/dashboard", params={"session_id": "nope"}).status_code == 404


def numeric_lists(value):
    if isinstance(value, dict):
        return [found for item in value.values() for found in numeric_lists(item)]
    if isinstance(value, list):
        if value and all(isinstance(item, (int, float)) for item in value):
            return [value]
        return [found for item in value for found in numeric_lists(item)]
    return []


def test_api_never_returns_tensor_like_data(store, db_path):
    """Tensors would arrive as lists of numbers; the API has none (SPEC §4.3 D)."""
    session = store.start_session(1, 4)
    assignment = store.start_assignment(session, 1, "inst-1", "inst-2", 0, 5)
    store.complete_assignment(assignment, 0.5, 0.5)
    assert numeric_lists(client(db_path).get("/api/dashboard").json()) == []


def test_serves_the_built_frontend(tmp_path, db_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<h1>MedWeave</h1>")
    assert "MedWeave" in client(db_path, frontend=dist).get("/").text


def test_explains_how_to_build_a_missing_frontend(db_path):
    assert "npm run build" in client(db_path, frontend=None).get("/").text
