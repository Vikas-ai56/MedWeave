"""Flower ServerApp: the coordinator's entry point for a training session."""

from pathlib import Path

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from coordinator.checkpoint_manager.manager import CheckpointManager
from coordinator.flower_app.client_app import state_dir
from coordinator.flower_app.protocol import MODEL_CONFIG
from coordinator.flower_app.session import register_institutions, run_session
from coordinator.relay.relay import Relay
from coordinator.relay.transport import RelayServer
from coordinator.scheduling.schedule import build_schedule
from split_learning.backbone import build_backbone

app = ServerApp()


def log(text: str) -> None:
    print(f"[coordinator] {text}", flush=True)


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = context.run_config
    nodes = register_institutions(
        grid, int(config["num-institutions"]), float(config["registration-timeout"]), log=log
    )

    manager = CheckpointManager(Path(state_dir()) / "coordinator" / "checkpoints")
    if manager.latest_version is None:
        manager.initialize(build_backbone(MODEL_CONFIG).state_dict())
        log("initialised Backbone v0")
    else:
        log(f"continuing from stored Backbone v{manager.latest_version}")

    schedule = build_schedule(sorted(nodes), int(config["num-rounds"]))
    relay = Relay()
    with RelayServer(relay, str(config["relay-host"]), int(config["relay-port"])):
        run_session(
            grid,
            nodes,
            schedule,
            manager,
            relay,
            str(config["relay-url"]),
            int(config["batches-per-turn"]),
            float(config["turn-timeout"]),
            log=log,
        )
