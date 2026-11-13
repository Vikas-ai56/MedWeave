"""Learning-rate schedules, as a factor of the base learning rate at each step.

Shared by the PathMNIST experiments, which step once per batch, and the
coordinator, which steps once per turn and sends each turn its learning rate.
"""

import math
from dataclasses import dataclass

SCHEDULES = ("constant", "cosine")


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
