"""Optimizer and learning-rate schedule for the PathMNIST experiments.

The schedule is applied per batch by wrapping the batch iterator, so it works
with the engine's `train_batches` and with optimizers that a compute peer
rebuilds at every handoff: each optimizer records its own `base_lr`, and the
schedule sets `lr = base_lr * factor` before every batch.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TypeVar

import torch
from torch.optim import Optimizer

from split_learning.config import OptimizerFactory
from split_learning.schedule import SCHEDULES, LRSchedule  # noqa: F401  (re-exported)

T = TypeVar("T")


def adamw(lr: float, weight_decay: float = 0.0) -> OptimizerFactory:
    """AdamW factory; with `weight_decay=0` it is exactly Adam."""

    def build(parameters: Iterable[torch.nn.Parameter]) -> Optimizer:
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        for group in optimizer.param_groups:
            group["base_lr"] = lr
        return optimizer

    return build


def set_lr(optimizer: Optimizer, factor: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor


class StepClock:
    """Counts training batches across the whole run and applies the schedule to each one."""

    def __init__(self, schedule: LRSchedule) -> None:
        self.schedule = schedule
        self.step = 0

    def drive(self, batches: Iterable[T], optimizers: Callable[[], Sequence[Optimizer]]) -> Iterator[T]:
        """Yield `batches`, setting the learning rate of `optimizers()` just before each one is used."""
        for batch in batches:
            factor = self.schedule.factor(self.step)
            for optimizer in optimizers():
                set_lr(optimizer, factor)
            self.step += 1
            yield batch
