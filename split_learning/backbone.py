from collections.abc import Sequence

from torch import Tensor, nn

from split_learning.config import SplitModelConfig, seeded
from split_learning.layers import conv_blocks


class Backbone(nn.Module):
    """Shared middle of the U, executed by the assigned compute peer."""

    def __init__(self, in_channels: int, channels: Sequence[int]) -> None:
        super().__init__()
        self.layers = conv_blocks(in_channels, channels)

    def forward(self, activations: Tensor) -> Tensor:
        return self.layers(activations)


def build_backbone(config: SplitModelConfig) -> Backbone:
    with seeded(config.seed + 1):
        return Backbone(config.head_channels[-1], config.backbone_channels)
