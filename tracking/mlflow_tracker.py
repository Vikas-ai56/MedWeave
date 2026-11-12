"""MLflow implementation of `Tracker`; the only module in the repo that imports `mlflow`.

Runs go to a local SQLite store and file artifacts under one directory per machine (see
`tracking.locations`), so the coordinator, whose working directory is Flower's app folder, and the
experiment runner write to the same place. An experiment is created on first use with its artifact
location pinned there; nested runs follow MLflow's own active-run stack.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from tracking.tracker import Scalar


class MlflowTracker:
    def __init__(self, tracking_uri: str, experiment: str, artifact_root: Path) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        existing = client.get_experiment_by_name(experiment)
        experiment_id = (
            existing.experiment_id
            if existing is not None
            else client.create_experiment(experiment, artifact_location=(artifact_root / experiment).as_uri())
        )
        mlflow.set_experiment(experiment_id=experiment_id)

    @contextmanager
    def run(
        self, name: str, params: Mapping[str, Scalar], tags: Mapping[str, str] | None = None, nested: bool = False
    ) -> Iterator[None]:
        with mlflow.start_run(run_name=name, nested=nested, tags=dict(tags or {})):
            mlflow.log_params(dict(params))
            yield

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        mlflow.log_metrics(dict(metrics), step=step)

    def log_artifact(self, path: Path) -> None:
        mlflow.log_artifact(str(path))
