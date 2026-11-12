"""Flower ServerApp: the coordinator's entry point for a training session.

The whole session is one tracked run (`tracking` setting: `mlflow` or `none`) in
experiment `medweave-sessions`: all run settings and the model config as params,
the git commit as tags, per-turn and per-round metrics, and the final Backbone
checkpoint as an artifact. A failed session leaves a failed run.
"""

from dataclasses import asdict
from pathlib import Path

from flwr.app import Context, UserConfig
from flwr.serverapp import Grid, ServerApp

from coordinator.checkpoint_manager.manager import CheckpointManager
from coordinator.flower_app.client_app import state_dir
from coordinator.flower_app.protocol import MODEL_CONFIG
from coordinator.flower_app.session import EvaluationSettings, register_institutions, run_session
from coordinator.relay.relay import Relay
from coordinator.relay.transport import RelayServer
from coordinator.scheduling.schedule import build_schedule
from split_learning.backbone import build_backbone
from tracking import Tracker, open_tracker, source_tags

app = ServerApp()


def log(text: str) -> None:
    print(f"[coordinator] {text}", flush=True)


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = context.run_config
    tracker = open_tracker(str(config["tracking"]), "medweave-sessions")
    params = {**dict(config), **{f"model.{key}": str(value) for key, value in asdict(MODEL_CONFIG).items()}}
    with tracker.run(f"session-{context.run_id}", params, tags=source_tags()):
        run_coordinator(grid, config, tracker)


def run_coordinator(grid: Grid, config: UserConfig, tracker: Tracker) -> None:
    nodes = register_institutions(
        grid, int(config["num-institutions"]), float(config["registration-timeout"]), log=log
    )

    manager = CheckpointManager(Path(state_dir()) / "coordinator" / "checkpoints")
    if manager.latest_version is None:
        manager.initialize(build_backbone(MODEL_CONFIG).state_dict())
        log("initialised Backbone v0")
    else:
        log(f"continuing from stored Backbone v{manager.latest_version}")

    evaluation_split = str(config["evaluation-split"])
    evaluation = (
        None
        if evaluation_split == "none"
        else EvaluationSettings(evaluation_split, int(config["evaluation-batch-size"]), float(config["turn-timeout"]))
    )
    schedule = build_schedule(sorted(nodes), int(config["num-rounds"]))
    relay = Relay()
    with RelayServer(relay, str(config["relay-host"]), int(config["relay-port"])):
        session = run_session(
            grid,
            nodes,
            schedule,
            manager,
            relay,
            str(config["relay-url"]),
            int(config["batches-per-turn"]),
            float(config["turn-timeout"]),
            log=log,
            evaluation=evaluation,
            tracker=tracker,
        )
    if session.turns:
        tracker.log_artifact(session.turns[-1].checkpoint.storage_path)
