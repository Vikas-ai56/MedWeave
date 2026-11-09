from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SplitModelConfig:
    in_channels: int = 1
    head_channels: int = 16
    backbone_channels: int = 32
    num_classes: int = 2
    seed: int = 0


@contextmanager
def seeded(seed: int) -> Iterator[None]:
    """Seed module construction without disturbing the global RNG state."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        yield
