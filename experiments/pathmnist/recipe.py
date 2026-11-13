"""Optimizer and learning-rate schedule for the PathMNIST experiments.

The schedule is applied per batch by wrapping the batch iterator, so it works
with the engine's `train_batches` and with optimizers that a compute peer
rebuilds at every handoff: each optimizer records its own `base_lr`, and the
schedule sets `lr = base_lr * factor` before every batch.
"""

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TypeVar

import torch
from torch.optim import Optimizer

from split_learning.config import OptimizerFactory

T = TypeVar("T")
SCHEDULES = ("constant", "cosine")


def adamw(lr: float, weight_decay: float = 0.0) -> OptimizerFactory:
    """AdamW factory; with `weight_decay=0` it is exactly Adam."""

    def build(parameters: Iterable[torch.nn.Parameter]) -> Optimizer:
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        for group in optimizer.param_groups:
            group["base_lr"] = lr
        return optimizer

    return build


@dataclass(frozen=True)
class LRSchedule:
    """`constant`: factor 1. `cosine`: linear warm-up over `warmup` of the steps, then cosine decay to 0."""

    kind: str
    total_steps: int
    warmup: float = 0.05

    def __post_init__(self) -> None:
        if self.kind not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {self.kind!r}")
        if self.total_steps < 1 or not 0.0 <= self.warmup < 1.0:
            raise ValueError(f"invalid schedule: {self.total_steps} steps, warmup {self.warmup}")

    def factor(self, step: int) -> float:
        if self.kind == "constant":
            return 1.0
        warmup_steps = max(1, round(self.total_steps * self.warmup))
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, self.total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


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
