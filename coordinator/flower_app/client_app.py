"""Flower ClientApp run by every institution's SuperNode.

Each SuperNode is started with `--node-config 'institution-id="inst-N"'`. In a
turn it is told to act as either the data owner or the compute peer. After each
round it evaluates its own Head and Tail with the latest Backbone on a public
held-out split, locally, and reports only the metrics.

The owner's data is opened lazily, inside `run_owner_turn`, so a data error goes
through the turn's failure path and the compute peer is told to stop at once
instead of waiting for its timeout.

As data owner it trains on its own provisioned PathMNIST shard (`institution_data`,
data root `$MEDWEAVE_DATA_DIR`) and perturbs the Head activations with
`privacy.GaussianActivationNoise` before they are sent. Both roles build their
optimizers from the `optimizer` and `learning-rate` run settings, so the Head,
Tail and Backbone always train with the same optimizer. Run settings used:
`optimizer`, `learning-rate`, `partition-seed`, `batch-size`, `noise-std`, `noise-seed`.
"""

import os
from collections.abc import Iterator
from pathlib import Path

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict, UserConfig
from flwr.clientapp import ClientApp

from coordinator.flower_app import protocol
from coordinator.flower_app.protocol import MODEL_CONFIG, TurnInfo
from coordinator.flower_app.roles import Batch, run_owner_turn, run_peer_turn
from coordinator.relay.transport import RelayClient
from institution_data import default_data_dir, evaluation_loader, open_evaluation, open_institution_training, training_loader
from privacy import GaussianActivationNoise, derive_seed
from split_learning.checkpoint import load_institution_checkpoint, save_institution_checkpoint
from split_learning.compute_peer import ComputePeer
from split_learning.config import OptimizerFactory, optimizer_factory
from split_learning.institution import Institution
from split_learning.training import evaluate

app = ClientApp()


def state_dir() -> Path:
    """Where this machine keeps its local state (Head/Tail checkpoints)."""
    return Path(os.environ.get("MEDWEAVE_STATE_DIR", Path.home() / ".medweave"))


def session_optimizer(context: Context) -> OptimizerFactory:
    return optimizer_factory(str(context.run_config["optimizer"]), float(context.run_config["learning-rate"]))


def institution_id(context: Context) -> str:
    try:
        return str(context.node_config[protocol.INSTITUTION_ID])
    except KeyError:
        raise ValueError(f"SuperNode was started without --node-config '{protocol.INSTITUTION_ID}=...'") from None


def local_institution(me: str, context: Context) -> tuple[Institution, Path]:
    institution = Institution(me, MODEL_CONFIG, session_optimizer(context))
    checkpoint = state_dir() / "institutions" / f"{me}.pt"
    if checkpoint.exists():
        load_institution_checkpoint(checkpoint, institution)
    return institution, checkpoint


def turn_batches(me: str, settings: UserConfig, turn: TurnInfo) -> Iterator[Batch]:
    partition_seed = int(settings["partition-seed"])
    local_data = open_institution_training(default_data_dir(), partition_seed, me)
    yield from training_loader(local_data, int(settings["batch-size"]), partition_seed, turn.round_number, turn.num_batches)


@app.query("register")
def register(msg: Message, context: Context) -> Message:
    info = ConfigRecord({protocol.INSTITUTION_ID: institution_id(context)})
    return Message(RecordDict({protocol.INFO: info}), reply_to=msg)


@app.train("peer")
def compute_for_peer(msg: Message, context: Context) -> Message:
    """Run the shared Backbone for another institution's turn and return the update."""
    me = institution_id(context)
    turn = TurnInfo.from_record(msg.content[protocol.TURN])
    if turn.peer_id != me:
        raise ValueError(f"{me} received a compute-peer message meant for {turn.peer_id}")

    peer = ComputePeer(MODEL_CONFIG, session_optimizer(context))
    peer.load_backbone(msg.content[protocol.BACKBONE].to_torch_state_dict(), turn.backbone_version)
    update = run_peer_turn(peer, RelayClient(turn.relay_url), me, turn.timeout)

    content = RecordDict(
        {
            protocol.BACKBONE: ArrayRecord(torch_state_dict=update.state_dict),
            protocol.UPDATE: ConfigRecord({"base_version": update.base_version, "num_batches": update.num_batches}),
        }
    )
    return Message(content, reply_to=msg)


@app.train("owner")
def train_own_data(msg: Message, context: Context) -> Message:
    """Train on local data through the remote Backbone; Head and Tail stay here."""
    me = institution_id(context)
    turn = TurnInfo.from_record(msg.content[protocol.TURN])
    if turn.owner_id != me:
        raise ValueError(f"{me} received a data-owner message meant for {turn.owner_id}")

    institution, checkpoint = local_institution(me, context)
    settings = context.run_config
    noise = GaussianActivationNoise(
        float(settings["noise-std"]), derive_seed(int(settings["noise-seed"]), me, turn.round_number)
    )
    summary = run_owner_turn(
        institution,
        turn_batches(me, settings, turn),
        RelayClient(turn.relay_url),
        turn.peer_id,
        turn.backbone_version,
        turn.timeout,
        perturb=noise,
    )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_institution_checkpoint(checkpoint, institution)
    metrics = MetricRecord(
        {
            "loss": summary.loss,
            "accuracy": summary.accuracy,
            "num_batches": summary.num_batches,
            "num_samples": summary.num_samples,
            "noise_std": noise.noise_std,
            "activation_change_pct": noise.summary().relative_change_pct,
        }
    )
    return Message(RecordDict({protocol.METRICS: metrics}), reply_to=msg)


@app.evaluate("held_out")
def evaluate_locally(msg: Message, context: Context) -> Message:
    me = institution_id(context)
    request = msg.content[protocol.EVALUATION]
    institution, _ = local_institution(me, context)
    backbone = ComputePeer(MODEL_CONFIG)
    backbone.load_backbone(msg.content[protocol.BACKBONE].to_torch_state_dict(), str(request["backbone_version"]))
    held_out = open_evaluation(default_data_dir(), str(request["split"]))
    summary = evaluate(institution, backbone, evaluation_loader(held_out, int(request["batch_size"])))
    metrics = MetricRecord({"loss": summary.loss, "accuracy": summary.accuracy, "num_samples": summary.num_samples})
    return Message(RecordDict({protocol.METRICS: metrics}), reply_to=msg)
