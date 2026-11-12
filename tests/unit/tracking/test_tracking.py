"""Tracking interface: no-op default, backend selection, and a real MLflow round trip in a temporary store."""

import re

import pytest

from tracking import NullTracker, open_tracker, source_tags, tracking_uri


def test_none_gives_a_tracker_that_does_nothing(tmp_path):
    tracker = open_tracker("none", "anything")
    assert isinstance(tracker, NullTracker)
    with tracker.run("run", {"a": 1}, nested=True):
        tracker.log_metrics({"x": 1.0}, step=1)
        tracker.log_artifact(tmp_path / "missing.txt")


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="tracking must be one of"):
        open_tracker("wandb", "anything")


def test_store_follows_the_tracking_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDWEAVE_TRACKING_DIR", str(tmp_path))
    assert tracking_uri() == f"sqlite:///{tmp_path / 'mlflow.db'}"


def test_source_tags_record_the_git_commit():
    tags = source_tags()
    assert re.fullmatch(r"[0-9a-f]{40}", tags["git_commit"]) and tags["git_dirty"] in ("true", "false")


def test_mlflow_records_nested_runs_metrics_artifacts_and_failures(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MEDWEAVE_TRACKING_DIR", str(tmp_path))
    artifact = tmp_path / "results.csv"
    artifact.write_text("a,b\n1,2\n")
    tracker = open_tracker("mlflow", "test-experiment")

    with tracker.run("parent", {"rounds": 2}, tags={"git_commit": "abc"}):
        with tracker.run("child", {"noise_std": 0.25}, nested=True):
            tracker.log_metrics({"val_accuracy": 0.5}, step=1)
            tracker.log_metrics({"val_accuracy": 0.7}, step=2)
        tracker.log_artifact(artifact)
    with pytest.raises(RuntimeError), tracker.run("broken", {}):
        raise RuntimeError("session failed")

    client = mlflow.tracking.MlflowClient()
    runs = {run.info.run_name: run for run in client.search_runs([client.get_experiment_by_name("test-experiment").experiment_id])}
    assert runs["parent"].data.params == {"rounds": "2"} and runs["parent"].data.tags["git_commit"] == "abc"
    assert runs["child"].data.tags["mlflow.parentRunId"] == runs["parent"].info.run_id
    history = client.get_metric_history(runs["child"].info.run_id, "val_accuracy")
    assert [(m.step, m.value) for m in history] == [(1, 0.5), (2, 0.7)]
    assert [a.path for a in client.list_artifacts(runs["parent"].info.run_id)] == ["results.csv"]
    assert runs["parent"].info.status == "FINISHED" and runs["broken"].info.status == "FAILED"
