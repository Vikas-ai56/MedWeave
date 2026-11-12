"""What the coordinator and institutions say to each other through Flower.

Flower carries control only: registration, "start your turn" instructions,
the Backbone checkpoint in both directions, and the end-of-round evaluation
request (Backbone out, each institution's held-out metrics back). Per-batch
tensors go through the relay instead (see `coordinator.relay`).
"""

from dataclasses import asdict, dataclass

from flwr.app import ConfigRecord, MessageType

from split_learning.config import SplitModelConfig

REGISTER = f"{MessageType.QUERY}.register"
TRAIN_AS_PEER = f"{MessageType.TRAIN}.peer"
TRAIN_AS_OWNER = f"{MessageType.TRAIN}.owner"
EVALUATE_HELD_OUT = f"{MessageType.EVALUATE}.held_out"

# RecordDict keys
TURN = "turn"
BACKBONE = "backbone"
UPDATE = "update"
METRICS = "metrics"
EVALUATION = "evaluation"
INFO = "info"
INSTITUTION_ID = "institution-id"

MODEL_CONFIG = SplitModelConfig(in_channels=3, image_size=(28, 28), num_classes=9, group_norm=True)


@dataclass(frozen=True)
class TurnInfo:
    """Everything both parties of a turn need to know, sent in a ConfigRecord."""

    round_number: int
    owner_id: str
    peer_id: str
    backbone_version: str
    relay_url: str
    num_batches: int
    timeout: float

    def to_record(self) -> ConfigRecord:
        return ConfigRecord(asdict(self))

    @classmethod
    def from_record(cls, record: ConfigRecord) -> "TurnInfo":
        return cls(
            round_number=int(record["round_number"]),
            owner_id=str(record["owner_id"]),
            peer_id=str(record["peer_id"]),
            backbone_version=str(record["backbone_version"]),
            relay_url=str(record["relay_url"]),
            num_batches=int(record["num_batches"]),
            timeout=float(record["timeout"]),
        )
