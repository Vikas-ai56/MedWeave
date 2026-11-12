"""Coordinator relay for per-batch split-learning traffic (SPEC §2.6).

During a turn the data owner and its compute peer never talk directly: every
activation and gradient goes through the coordinator's relay. The relay only
checks the envelope (who, which batch, which Backbone version) and forwards the
tensor bytes unopened. It never decodes, stores or logs a tensor, and it drops
undelivered messages when the turn closes.
"""

import io
import queue
import threading
from dataclasses import dataclass, replace
from enum import StrEnum

import torch
from torch import Tensor


class RelayError(Exception):
    """A message does not belong to the active turn or cannot be delivered."""


class RelayTimeout(RelayError):
    """No message arrived for the recipient before the timeout."""


class Kind(StrEnum):
    """Message kinds, in the order they flow for one batch."""

    ACTIVATIONS = "activations"  # owner -> peer: noisy Head output
    BACKBONE_OUTPUT = "backbone_output"  # peer -> owner
    GRADIENT = "gradient"  # owner -> peer: d loss / d backbone output
    INPUT_GRADIENT = "input_gradient"  # peer -> owner: d loss / d activations
    DONE = "done"  # owner -> peer: no more batches this turn


_TO_PEER = {Kind.ACTIVATIONS, Kind.GRADIENT, Kind.DONE}


@dataclass(frozen=True)
class Envelope:
    """One relayed message. `institution_id` is always the data owner's id."""

    kind: Kind
    institution_id: str
    batch_id: str
    backbone_version: str
    body: bytes = b""

    def reply(self, kind: Kind, body: bytes) -> "Envelope":
        return replace(self, kind=kind, body=body)


@dataclass(frozen=True)
class ActiveTurn:
    owner_id: str
    peer_id: str
    backbone_version: str


def encode_tensor(tensor: Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return buffer.getvalue()


def decode_tensor(body: bytes) -> Tensor:
    """Decode bytes from `encode_tensor`. Uses `weights_only=True`, so no code runs."""
    try:
        tensor = torch.load(io.BytesIO(body), weights_only=True)
    except Exception as error:
        raise RelayError(f"message body is not a tensor: {error}") from error
    if not isinstance(tensor, Tensor):
        raise RelayError(f"message body is a {type(tensor).__name__}, not a tensor")
    return tensor


class Relay:
    """Per-recipient mailboxes for the one turn that is currently running."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turn: ActiveTurn | None = None
        self._mailboxes: dict[str, queue.Queue[Envelope]] = {}

    @property
    def active_turn(self) -> ActiveTurn | None:
        return self._turn

    def open_turn(self, owner_id: str, peer_id: str, backbone_version: str) -> None:
        if owner_id == peer_id:
            raise ValueError(f"{owner_id!r} cannot relay to itself")
        with self._lock:
            if self._turn is not None:
                raise RelayError(f"turn for {self._turn.owner_id!r} is still open")
            self._turn = ActiveTurn(owner_id, peer_id, backbone_version)
            self._mailboxes = {owner_id: queue.Queue(), peer_id: queue.Queue()}

    def close_turn(self) -> None:
        """End the turn and drop anything that was never collected."""
        with self._lock:
            self._turn = None
            self._mailboxes = {}

    def send(self, recipient: str, envelope: Envelope) -> None:
        with self._lock:
            turn = self._require_turn()
            expected = turn.peer_id if envelope.kind in _TO_PEER else turn.owner_id
            if recipient != expected:
                raise RelayError(f"{envelope.kind} must go to {expected!r}, not {recipient!r}")
            if envelope.institution_id != turn.owner_id:
                raise RelayError(
                    f"message is for {envelope.institution_id!r}, active data owner is {turn.owner_id!r}"
                )
            if envelope.backbone_version != turn.backbone_version:
                raise RelayError(
                    f"message is for Backbone version {envelope.backbone_version!r}, "
                    f"active turn uses {turn.backbone_version!r}"
                )
            mailbox = self._mailboxes[recipient]
        mailbox.put(envelope)

    def receive(self, recipient: str, timeout: float) -> Envelope:
        with self._lock:
            turn = self._require_turn()
            if recipient not in (turn.owner_id, turn.peer_id):
                raise RelayError(f"{recipient!r} is not part of the active turn")
            mailbox = self._mailboxes[recipient]
        try:
            return mailbox.get(timeout=timeout)
        except queue.Empty:
            raise RelayTimeout(f"nothing arrived for {recipient!r} within {timeout}s") from None

    def _require_turn(self) -> ActiveTurn:
        if self._turn is None:
            raise RelayError("no turn is open")
        return self._turn
