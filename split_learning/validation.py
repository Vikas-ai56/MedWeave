"""Checks applied to every tensor that crosses a split boundary."""

from typing import Generic, TypeVar

import torch
from torch import Tensor

from split_learning.config import Shape
from split_learning.errors import ProtocolError, ShapeError

T = TypeVar("T")


def check_batch(tensor: Tensor, sample_shape: Shape, name: str) -> None:
    """`tensor` must be a non-empty float batch of shape (batch, *sample_shape)."""
    if not isinstance(tensor, Tensor):
        raise ShapeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if not tensor.is_floating_point():
        raise ShapeError(f"{name} must be a floating-point tensor, got {tensor.dtype}")
    if tensor.ndim != len(sample_shape) + 1 or tuple(tensor.shape[1:]) != sample_shape:
        expected = ", ".join(map(str, sample_shape))
        raise ShapeError(f"{name} must have shape (batch, {expected}), got {tuple(tensor.shape)}")
    if tensor.shape[0] == 0:
        raise ShapeError(f"{name} must contain at least one sample")


def check_gradient(gradient: Tensor, expected: torch.Size, name: str) -> None:
    """A returned gradient must match the tensor it was computed for exactly."""
    check_batch(gradient, tuple(expected[1:]), name)
    if gradient.shape != expected:
        raise ShapeError(f"{name} must have shape {tuple(expected)}, got {tuple(gradient.shape)}")


def check_labels(labels: Tensor, batch_size: int, num_classes: int) -> None:
    if not isinstance(labels, Tensor) or labels.dtype != torch.long:
        raise ShapeError("labels must be a torch.long tensor of class indices")
    if labels.shape != (batch_size,):
        raise ShapeError(f"labels must have shape ({batch_size},), got {tuple(labels.shape)}")
    low, high = int(labels.min()), int(labels.max())
    if low < 0 or high >= num_classes:
        raise ShapeError(f"labels must be in [0, {num_classes}), got values from {low} to {high}")


class PendingBatch(Generic[T]):
    """Holds the one batch whose forward pass is waiting for its gradient.

    Training is sequential, so each party has at most one batch in flight.
    Taking the batch drops the reference, which frees its autograd graph.
    """

    def __init__(self) -> None:
        self._batch: tuple[str, T] | None = None

    @property
    def batch_id(self) -> str | None:
        return None if self._batch is None else self._batch[0]

    def put(self, batch_id: str, value: T) -> None:
        if self._batch is not None:
            raise ProtocolError(f"batch {self._batch[0]!r} is still waiting for its gradient")
        self._batch = (batch_id, value)

    def peek(self, batch_id: str) -> T:
        if self._batch is None:
            raise ProtocolError(f"no forward pass is pending for batch {batch_id!r}")
        if self._batch[0] != batch_id:
            raise ProtocolError(f"expected batch {self._batch[0]!r}, got {batch_id!r}")
        return self._batch[1]

    def take(self, batch_id: str) -> T:
        value = self.peek(batch_id)
        self._batch = None
        return value

    def clear(self) -> None:
        self._batch = None
