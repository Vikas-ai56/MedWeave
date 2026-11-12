"""The two jobs an institution can have during a turn, independent of Flower.

`run_owner_turn` drives the batches from the data owner's side; `run_peer_turn`
answers them from the compute peer's side. They exchange messages through any
transport with `send(recipient, envelope)` and `receive(recipient, timeout)`:
the in-process `Relay` in tests, or `RelayClient` over HTTP in a real run.
Inputs and labels stay inside the `Institution`; only activations and
gradients are sent.
"""

from collections.abc import Callable, Iterable
from typing import Protocol

from torch import Tensor

from coordinator.relay.relay import Envelope, Kind, RelayError, decode_tensor, encode_tensor
from split_learning.compute_peer import BackboneUpdate, ComputePeer
from split_learning.institution import Institution, StepMetrics
from split_learning.training import Summary, summarize

Batch = tuple[Tensor, Tensor]
Perturb = Callable[[Tensor], Tensor]


class Transport(Protocol):
    def send(self, recipient: str, envelope: Envelope) -> None: ...

    def receive(self, recipient: str, timeout: float) -> Envelope: ...


def run_owner_turn(
    institution: Institution,
    batches: Iterable[Batch],
    transport: Transport,
    peer_id: str,
    backbone_version: str,
    timeout: float,
    perturb: Perturb | None = None,
) -> Summary:
    """Train every batch through the remote Backbone, then tell the peer to stop.

    `perturb` is where the privacy layer's noise is applied to the activations.
    """
    owner_id = institution.institution_id
    institution.train()
    steps: list[StepMetrics] = []
    try:
        for index, (inputs, labels) in enumerate(batches):
            batch_id = f"{owner_id}-v{backbone_version}-{index}"
            envelope = Envelope(Kind.ACTIVATIONS, owner_id, batch_id, backbone_version)

            activations = institution.head_forward(batch_id, inputs)
            if perturb is not None:
                activations = perturb(activations)
            transport.send(peer_id, envelope.reply(Kind.ACTIVATIONS, encode_tensor(activations)))

            output = _expect(transport.receive(owner_id, timeout), Kind.BACKBONE_OUTPUT, batch_id)
            tail = institution.tail_step(batch_id, decode_tensor(output.body), labels)
            transport.send(peer_id, envelope.reply(Kind.GRADIENT, encode_tensor(tail.gradient)))

            gradient = _expect(transport.receive(owner_id, timeout), Kind.INPUT_GRADIENT, batch_id)
            institution.head_backward(batch_id, decode_tensor(gradient.body))
            steps.append(tail.metrics)
    except Exception:
        institution.discard_pending()
        raise
    finally:
        _tell_peer_done(transport, peer_id, owner_id, backbone_version)
    return summarize(steps)


def run_peer_turn(peer: ComputePeer, transport: Transport, peer_id: str, timeout: float) -> BackboneUpdate:
    """Serve Backbone forward and backward passes until the data owner is done.

    `peer` must already hold the turn's Backbone (`ComputePeer.load_backbone`).
    """
    peer.train()
    try:
        while (envelope := transport.receive(peer_id, timeout)).kind is not Kind.DONE:
            tensor = decode_tensor(envelope.body)
            owner_id = envelope.institution_id
            if envelope.kind is Kind.ACTIVATIONS:
                output = peer.forward(envelope.batch_id, tensor, envelope.backbone_version)
                transport.send(owner_id, envelope.reply(Kind.BACKBONE_OUTPUT, encode_tensor(output)))
            elif envelope.kind is Kind.GRADIENT:
                gradient = peer.backward(envelope.batch_id, tensor, envelope.backbone_version)
                transport.send(owner_id, envelope.reply(Kind.INPUT_GRADIENT, encode_tensor(gradient)))
            else:
                raise RelayError(f"compute peer cannot handle {envelope.kind}")
    except Exception:
        peer.discard_pending()
        raise
    return peer.backbone_update()


def _expect(envelope: Envelope, kind: Kind, batch_id: str) -> Envelope:
    if envelope.kind is not kind or envelope.batch_id != batch_id:
        raise RelayError(
            f"expected {kind} for batch {batch_id!r}, got {envelope.kind} for {envelope.batch_id!r}"
        )
    return envelope


def _tell_peer_done(transport: Transport, peer_id: str, owner_id: str, backbone_version: str) -> None:
    """Best effort: on failure the peer still stops when its receive times out."""
    try:
        transport.send(peer_id, Envelope(Kind.DONE, owner_id, "", backbone_version))
    except RelayError:
        pass
