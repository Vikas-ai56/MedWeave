from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from split_learning.backward import head_backward, tail_backward
from split_learning.config import OptimizerFactory, SplitModelConfig, default_optimizer
from split_learning.forward import HeadPass, head_forward, tail_forward
from split_learning.head import build_head
from split_learning.tail import build_tail
from split_learning.validation import PendingBatch, check_batch, check_exact, check_labels


@dataclass(frozen=True)
class StepMetrics:
    loss: float
    correct: int
    total: int


@dataclass(frozen=True)
class TailResult:
    gradient: Tensor
    metrics: StepMetrics


class Institution:
    """A data owner's local half of the U: Head, Tail and their own optimizers.

    Lifecycle per batch: `head_forward`, `tail_step`, `head_backward`. Inputs
    and labels stay inside this object; only activations and gradients leave.
    """

    def __init__(
        self,
        institution_id: str,
        config: SplitModelConfig,
        optimizer_factory: OptimizerFactory = default_optimizer,
    ) -> None:
        self.institution_id = institution_id
        self.config = config
        self.head = build_head(config)
        self.tail = build_tail(config)
        self.head_optimizer = optimizer_factory(self.head.parameters())
        self.tail_optimizer = optimizer_factory(self.tail.parameters())
        self._pending: PendingBatch[HeadPass] = PendingBatch()

    def head_forward(self, batch_id: str, inputs: Tensor) -> Tensor:
        """Return detached activations to send to the compute peer."""
        check_batch(inputs, self.config.input_shape, "inputs")
        activations, head_pass = head_forward(self.head, inputs)
        self._pending.put(batch_id, head_pass)
        return activations

    def tail_step(self, batch_id: str, backbone_output: Tensor, labels: Tensor) -> TailResult:
        """Compute the loss, update the Tail and return the gradient for the peer."""
        head_pass = self._pending.peek(batch_id)
        batch_size = head_pass.activations.shape[0]
        expected_output = torch.Size((batch_size, *self.config.backbone_output_shape))
        check_exact(backbone_output, expected_output, "backbone output")
        check_labels(labels, batch_size, self.config.num_classes)
        tail_pass = tail_forward(self.tail, backbone_output, labels)
        gradient = tail_backward(tail_pass, self.tail_optimizer)
        return TailResult(gradient, metrics(tail_pass.loss, tail_pass.logits, labels))

    def head_backward(self, batch_id: str, gradient: Tensor) -> None:
        """Apply the gradient returned by the peer and update the Head."""
        head_pass = self._pending.peek(batch_id)
        check_exact(gradient, head_pass.activations.shape, "gradient")
        self._pending.take(batch_id)
        head_backward(head_pass, gradient, self.head_optimizer)

    def encode(self, inputs: Tensor) -> Tensor:
        """Head forward for evaluation: no graph kept, nothing pending."""
        check_batch(inputs, self.config.input_shape, "inputs")
        with torch.no_grad():
            return self.head(inputs)

    def evaluate_tail(self, backbone_output: Tensor, labels: Tensor) -> StepMetrics:
        check_batch(backbone_output, self.config.backbone_output_shape, "backbone output")
        check_labels(labels, backbone_output.shape[0], self.config.num_classes)
        with torch.no_grad():
            logits = self.tail(backbone_output)
            return metrics(F.cross_entropy(logits, labels), logits, labels)

    def discard_pending(self) -> None:
        self._pending.clear()

    def train(self, mode: bool = True) -> "Institution":
        self.head.train(mode)
        self.tail.train(mode)
        return self

    def eval(self) -> "Institution":
        return self.train(False)


def metrics(loss: Tensor, logits: Tensor, labels: Tensor) -> StepMetrics:
    correct = int((logits.detach().argmax(dim=1) == labels).sum())
    return StepMetrics(loss.item(), correct, len(labels))


def create_institutions(
    institution_ids: Sequence[str],
    config: SplitModelConfig,
    optimizer_factory: OptimizerFactory = default_optimizer,
) -> dict[str, Institution]:
    """Institutions with identical initial weights but fully independent parameters."""
    if len(set(institution_ids)) != len(institution_ids):
        raise ValueError(f"institution ids must be unique, got {list(institution_ids)}")
    return {id_: Institution(id_, config, optimizer_factory) for id_ in institution_ids}
