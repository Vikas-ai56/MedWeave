"""Run the U-shaped exchange in one process: simulation, tests and local debugging.

Each call on `peer` marks a point where, in a distributed run, a payload
crosses the network through the coordinator relay.
"""

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor

from split_learning.compute_peer import ComputePeer
from split_learning.institution import Institution, StepMetrics

Batch = tuple[Tensor, Tensor]
Perturb = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class Summary:
    loss: float
    accuracy: float
    num_batches: int
    num_samples: int


def train_step(
    institution: Institution,
    peer: ComputePeer,
    inputs: Tensor,
    labels: Tensor,
    batch_id: str = "batch",
    perturb: Perturb | None = None,
) -> StepMetrics:
    """One batch through Head, Backbone and Tail and back, updating all three.

    If any step fails, both parties drop the batch so the next step starts clean.
    """
    version = peer.version or ""  # an unloaded peer rejects the call itself
    try:
        activations = institution.head_forward(batch_id, inputs)
        if perturb is not None:
            with torch.no_grad():
                activations = perturb(activations)

        output = peer.forward(batch_id, activations, version)
        result = institution.tail_step(batch_id, output, labels)
        input_gradient = peer.backward(batch_id, result.gradient, version)
        institution.head_backward(batch_id, input_gradient)
    except Exception:
        institution.discard_pending()
        peer.discard_pending()
        raise
    return result.metrics


def train_batches(
    institution: Institution,
    peer: ComputePeer,
    batches: Iterable[Batch],
    perturb: Perturb | None = None,
) -> Summary:
    institution.train()
    peer.train()
    steps = [
        train_step(institution, peer, inputs, labels, f"{institution.institution_id}-{index}", perturb)
        for index, (inputs, labels) in enumerate(batches)
    ]
    return summarize(steps)


def evaluate(
    institution: Institution, peer: ComputePeer, batches: Iterable[Batch], perturb: Perturb | None = None
) -> Summary:
    """Score batches in eval mode without updating any weights; `perturb` noises activations as in training."""
    version = peer.version or ""  # an unloaded peer rejects the call itself
    steps = []
    with evaluation_mode(institution, peer):
        for inputs, labels in batches:
            activations = institution.encode(inputs)
            if perturb is not None:
                activations = perturb(activations)
            steps.append(institution.evaluate_tail(peer.infer(activations, version), labels))
    return summarize(steps)


def summarize(steps: list[StepMetrics]) -> Summary:
    """Sample-weighted mean loss and accuracy over a list of batches."""
    num_samples = sum(step.total for step in steps)
    if num_samples == 0:
        raise ValueError("cannot summarize an empty list of batches")
    return Summary(
        loss=sum(step.loss * step.total for step in steps) / num_samples,
        accuracy=sum(step.correct for step in steps) / num_samples,
        num_batches=len(steps),
        num_samples=num_samples,
    )


@contextmanager
def evaluation_mode(institution: Institution, peer: ComputePeer) -> Iterator[None]:
    modules = [institution.head, institution.tail, peer.backbone]
    previous = [module.training for module in modules]
    institution.eval()
    peer.eval()
    try:
        yield
    finally:
        for module, was_training in zip(modules, previous):
            module.train(was_training)
