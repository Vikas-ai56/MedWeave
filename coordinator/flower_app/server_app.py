"""Flower ServerApp: the coordinator's entry point for a training session.

The whole session is one tracked run (`tracking` setting: `mlflow` or `none`) in
experiment `medweave-sessions`: all run settings and the model config as params,
the git commit as tags, per-turn and per-round metrics, and the final Backbone
checkpoint as an artifact. A failed session leaves a failed run.

Learning rate: `learning-rate` is the base rate and `lr-schedule` (`constant` or
`cosine` with `lr-warmup` of the turns as linear warm-up) turns it into one rate
per turn, which each turn's `TurnInfo` carries to both parties.

`final-evaluation-split` (default `none`) scores every institution once more on
that split after the last round, e.g. `test`. Who may use the test split is a
team decision (docs/features/data-and-privacy.md), so it is off by default.
"""

from dataclasses import asdict
from pathlib import Path

from flwr.app import Context, UserConfig
from flwr.serverapp import Grid, ServerApp

from coordinator.checkpoint_manager.manager import CheckpointManager
from coordinator.flower_app.client_app import state_dir
from coordinator.flower_app.observer import DatabaseObserver
from coordinator.flower_app.protocol import MODEL_CONFIG
from coordinator.flower_app.session import (
    EvaluationSettings,
    evaluate_institutions,
    evaluation_metrics,
    register_institutions,
    run_session,
)
from coordinator.relay.relay import Relay
from coordinator.relay.transport import RelayServer
from coordinator.scheduling.schedule import build_schedule
from database.store import MetadataStore
from split_learning.backbone import build_backbone
from split_learning.schedule import LRSchedule
from tracking import Tracker, open_tracker, source_tags

app = ServerApp()


def log(text: str) -> None:
    print(f"[coordinator] {text}", flush=True)


def database_path() -> Path:
    """The metadata database the dashboard reads."""
    return Path(state_dir()) / "coordinator" / "medweave.db"


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = context.run_config
    tracker = open_tracker(str(config["tracking"]), "medweave-sessions")
    params = {**dict(config), **{f"model.{key}": str(value) for key, value in asdict(MODEL_CONFIG).items()}}
    store = MetadataStore(database_path())
    observer = DatabaseObserver(store)
    try:
        with tracker.run(f"session-{context.run_id}", params, tags=source_tags()):
            run_coordinator(grid, config, tracker, observer)
    except Exception as error:
        observer.session_failed(str(error))
        raise
    finally:
        store.close()


def run_coordinator(grid: Grid, config: UserConfig, tracker: Tracker, observer: DatabaseObserver) -> None:
    nodes = register_institutions(
        grid, int(config["num-institutions"]), float(config["registration-timeout"]), log=log, observer=observer
    )

    num_rounds = int(config["num-rounds"])
    schedule = build_schedule(sorted(nodes), num_rounds)
    observer.session_started(schedule, num_rounds)
    noise_std = float(config["noise-std"])
    if noise_std > 0:  # every institution reads the same setting (client_app.train_own_data)
        observer.noise_configured("gaussian", noise_std)

    manager = CheckpointManager(Path(state_dir()) / "coordinator" / "checkpoints")
    if manager.latest_version is None:
        observer.checkpoint_saved(manager.initialize(build_backbone(MODEL_CONFIG).state_dict()))
        log("initialised Backbone v0")
    else:
        log(f"continuing from stored Backbone v{manager.latest_version}")

    evaluation_split = str(config["evaluation-split"])
    evaluation = (
        None
        if evaluation_split == "none"
        else EvaluationSettings(evaluation_split, int(config["evaluation-batch-size"]), float(config["turn-timeout"]))
    )
    base_rate = float(config["learning-rate"])
    lr_schedule = LRSchedule(str(config["lr-schedule"]), len(schedule), float(config["lr-warmup"]))
    learning_rates = [base_rate * lr_schedule.factor(index) for index in range(len(schedule))]

    relay = Relay(on_message=observer.message_relayed)
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
            observer=observer,
            learning_rates=learning_rates,
        )
    final_split = str(config["final-evaluation-split"])
    if final_split != "none":
        settings = EvaluationSettings(final_split, int(config["evaluation-batch-size"]), float(config["turn-timeout"]))
        final = evaluate_institutions(grid, nodes, manager, num_rounds, settings, log)
        tracker.log_metrics({f"final_{key}": value for key, value in evaluation_metrics(final).items()}, step=num_rounds)
    if session.turns:
        tracker.log_artifact(session.turns[-1].checkpoint.storage_path)
