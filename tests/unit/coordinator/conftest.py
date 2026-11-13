"""Fixtures for coordinator tests that use Flower messages."""

import pytest
from flwr.supercore.task_identity import TaskIdentity


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
