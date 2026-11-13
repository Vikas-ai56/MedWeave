from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer

Shape = tuple[int, ...]
NORM_GROUPS = 8
OptimizerFactory = Callable[[Iterable[nn.Parameter]], Optimizer]


OPTIMIZERS: dict[str, type[Optimizer]] = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}


def default_optimizer(parameters: Iterable[nn.Parameter]) -> Optimizer:
    return torch.optim.SGD(parameters, lr=0.01)


def optimizer_factory(name: str, learning_rate: float, weight_decay: float = 0.0) -> OptimizerFactory:
    """Named optimizer at `learning_rate`; `weight_decay` is decoupled for AdamW and L2 for Adam and SGD."""
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {sorted(OPTIMIZERS)}, got {name!r}")
    if not learning_rate > 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if not weight_decay >= 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    optimizer_class = OPTIMIZERS[name]
    return lambda parameters: optimizer_class(parameters, lr=learning_rate, weight_decay=weight_decay)


def set_learning_rate(optimizer: Optimizer, learning_rate: float) -> None:
    """Set every parameter group's learning rate, e.g. after loading optimizer state that restored an old one."""
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


@dataclass(frozen=True)
class SplitModelConfig:
    """Architecture of the split model.

    Each entry in `head_channels` and `backbone_channels` adds one conv block
    (3x3 conv, ReLU, 2x2 max-pool) with that many output channels, so moving an
    entry between the two tuples moves the split point. `tail_hidden` adds
    hidden linear layers to the Tail.

    `batch_norm` inserts BatchNorm2d after every conv; `group_norm` inserts
    GroupNorm with `NORM_GROUPS` groups instead, which keeps no running
    statistics. `convs_per_block` repeats the (conv, norm, ReLU) unit inside
    each block before its pool. `backbone_final_pool=False` drops the max-pool
    from the last Backbone block, and `tail_dropout` adds Dropout before the
    Tail's classifier. The defaults leave the layer indices, and so the
    checkpoint keys, unchanged.
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
    group_norm: bool = False
    convs_per_block: int = 1

    def __post_init__(self) -> None:
        if self.in_channels < 1:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {self.num_classes}")
        if not 0.0 <= self.tail_dropout < 1.0:
            raise ValueError(f"tail_dropout must be in [0, 1), got {self.tail_dropout}")
        if self.convs_per_block < 1:
            raise ValueError(f"convs_per_block must be positive, got {self.convs_per_block}")
        if self.batch_norm and self.group_norm:
            raise ValueError("choose at most one of batch_norm and group_norm")
        if not self.head_channels or not self.backbone_channels:
            raise ValueError("head_channels and backbone_channels need at least one block each")
        widths = (*self.head_channels, *self.backbone_channels, *self.tail_hidden)
        if any(width < 1 for width in widths):
            raise ValueError(f"channel and hidden widths must be positive, got {widths}")
        conv_widths = (*self.head_channels, *self.backbone_channels)
        if self.group_norm and any(width % NORM_GROUPS for width in conv_widths):
            raise ValueError(f"group_norm needs channel widths divisible by {NORM_GROUPS}, got {conv_widths}")
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
