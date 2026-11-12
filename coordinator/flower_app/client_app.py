"""Flower ClientApp run by every institution's SuperNode.

Each SuperNode is started with `--node-config 'institution-id="inst-N"'`. In a
turn it is told to act as either the data owner or the compute peer.
"""

import os
import zlib
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from coordinator.flower_app import protocol
from coordinator.flower_app.protocol import MODEL_CONFIG, TurnInfo
from coordinator.flower_app.roles import Batch, run_owner_turn, run_peer_turn
from coordinator.relay.transport import RelayClient
from split_learning.checkpoint import load_institution_checkpoint, save_institution_checkpoint
from split_learning.compute_peer import ComputePeer
from split_learning.config import SplitModelConfig
from split_learning.institution import Institution

app = ClientApp()


def state_dir() -> Path:
    """Where this machine keeps its local state (Head/Tail checkpoints)."""
    return Path(os.environ.get("MEDWEAVE_STATE_DIR", Path.home() / ".medweave"))


def institution_id(context: Context) -> str:
    try:
        return str(context.node_config[protocol.INSTITUTION_ID])
    except KeyError:
        raise ValueError(f"SuperNode was started without --node-config '{protocol.INSTITUTION_ID}=...'") from None


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

    peer = ComputePeer(MODEL_CONFIG)
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

    institution = Institution(me, MODEL_CONFIG)
    checkpoint = state_dir() / "institutions" / f"{me}.pt"
    if checkpoint.exists():
        load_institution_checkpoint(checkpoint, institution)

    summary = run_owner_turn(
        institution,
        placeholder_batches(me, MODEL_CONFIG, turn.num_batches),
        RelayClient(turn.relay_url),
        turn.peer_id,
        turn.backbone_version,
        turn.timeout,
        perturb=None,  # Member 3's activation noise plugs in here
    )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_institution_checkpoint(checkpoint, institution)
    metrics = MetricRecord(
        {
            "loss": summary.loss,
            "accuracy": summary.accuracy,
            "num_batches": summary.num_batches,
            "num_samples": summary.num_samples,
        }
    )
    return Message(RecordDict({protocol.METRICS: metrics}), reply_to=msg)


def placeholder_batches(
    institution_id: str, config: SplitModelConfig, num_batches: int, batch_size: int = 8
) -> list[Batch]:
    """Random stand-in data, different per institution, until Member 3's loaders exist."""
    generator = torch.Generator().manual_seed(zlib.crc32(institution_id.encode()))
    return [
        (
            torch.randn(batch_size, *config.input_shape, generator=generator),
            torch.randint(0, config.num_classes, (batch_size,), generator=generator),
        )
        for _ in range(num_batches)
    ]
