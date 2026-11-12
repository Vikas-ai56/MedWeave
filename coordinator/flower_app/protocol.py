"""What the coordinator and institutions say to each other through Flower.

Flower carries control only: registration, "start your turn" instructions, and
the Backbone checkpoint in both directions. Per-batch tensors go through the
relay instead (see `coordinator.relay`).
"""

from dataclasses import asdict, dataclass

from flwr.app import ConfigRecord, MessageType

from split_learning.config import SplitModelConfig

REGISTER = f"{MessageType.QUERY}.register"
TRAIN_AS_PEER = f"{MessageType.TRAIN}.peer"
TRAIN_AS_OWNER = f"{MessageType.TRAIN}.owner"

# RecordDict keys
TURN = "turn"
BACKBONE = "backbone"
UPDATE = "update"
METRICS = "metrics"
INFO = "info"
INSTITUTION_ID = "institution-id"

# Placeholder model shared by every party until the dataset and task are chosen.
MODEL_CONFIG = SplitModelConfig()


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
