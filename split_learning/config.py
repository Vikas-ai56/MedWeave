from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer

Shape = tuple[int, ...]
OptimizerFactory = Callable[[Iterable[nn.Parameter]], Optimizer]


def default_optimizer(parameters: Iterable[nn.Parameter]) -> Optimizer:
    return torch.optim.SGD(parameters, lr=0.01)


@dataclass(frozen=True)
class SplitModelConfig:
    """Architecture of the split model.

    Each entry in `head_channels` and `backbone_channels` adds one conv block
    (3x3 conv, ReLU, 2x2 max-pool) with that many output channels, so moving an
    entry between the two tuples moves the split point. `tail_hidden` adds
    hidden linear layers to the Tail.

    `batch_norm` inserts BatchNorm2d after every conv, `backbone_final_pool=False`
    drops the max-pool from the last Backbone block, and `tail_dropout` adds
    Dropout before the Tail's classifier. Their defaults leave the layer
    indices, and so the checkpoint keys, unchanged.
    """

    in_channels: int = 1
    image_size: tuple[int, int] = (28, 28)
    head_channels: tuple[int, ...] = (16,)
    backbone_channels: tuple[int, ...] = (32, 32)
    tail_hidden: tuple[int, ...] = ()
    num_classes: int = 2
    seed: int = 0
    batch_norm: bool = False
    backbone_final_pool: bool = True
    tail_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.in_channels < 1:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {self.num_classes}")
        if not 0.0 <= self.tail_dropout < 1.0:
            raise ValueError(f"tail_dropout must be in [0, 1), got {self.tail_dropout}")
        if not self.head_channels or not self.backbone_channels:
            raise ValueError("head_channels and backbone_channels need at least one block each")
        widths = (*self.head_channels, *self.backbone_channels, *self.tail_hidden)
        if any(width < 1 for width in widths):
            raise ValueError(f"channel and hidden widths must be positive, got {widths}")
        if min(self.backbone_output_shape[1:]) < 1:
            blocks = len(self.head_channels) + len(self.backbone_channels)
            raise ValueError(f"image_size {self.image_size} is too small for {blocks} pooling blocks")

    @property
    def input_shape(self) -> Shape:
        return (self.in_channels, *self.image_size)

    @property
    def head_output_shape(self) -> Shape:
        return conv_blocks_output_shape(self.input_shape, self.head_channels)

    @property
    def backbone_output_shape(self) -> Shape:
        return conv_blocks_output_shape(
            self.head_output_shape, self.backbone_channels, pool_last=self.backbone_final_pool
        )


def conv_blocks_output_shape(input_shape: Shape, channels: tuple[int, ...], pool_last: bool = True) -> Shape:
    _, height, width = input_shape
    pools = len(channels) if pool_last else len(channels) - 1
    for _ in range(pools):
        height, width = height // 2, width // 2
    return (channels[-1], height, width)


@contextmanager
def seeded(seed: int) -> Iterator[None]:
    """Seed module construction without disturbing the global RNG state."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        yield
