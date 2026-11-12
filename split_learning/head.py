from collections.abc import Sequence

from torch import Tensor, nn

from split_learning.config import SplitModelConfig, seeded
from split_learning.layers import conv_blocks


class Head(nn.Module):
    """Institution-local front of the U: raw input -> smashed activations."""

    def __init__(self, in_channels: int, channels: Sequence[int], group_norm: bool = False) -> None:
        super().__init__()
        self.layers = conv_blocks(in_channels, channels, group_norm)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def build_head(config: SplitModelConfig) -> Head:
    with seeded(config.seed):
        return Head(config.in_channels, config.head_channels, config.group_norm)
