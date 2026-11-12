from collections.abc import Sequence

from torch import Tensor, nn

from split_learning.config import SplitModelConfig, seeded
from split_learning.layers import conv_blocks


class Head(nn.Module):
    """Institution-local front of the U: raw input -> smashed activations."""

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        batch_norm: bool = False,
        group_norm: bool = False,
        convs_per_block: int = 1,
    ) -> None:
        super().__init__()
        self.layers = conv_blocks(
            in_channels, channels, batch_norm=batch_norm, group_norm=group_norm, convs_per_block=convs_per_block
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def build_head(config: SplitModelConfig) -> Head:
    with seeded(config.seed):
        return Head(
            config.in_channels,
            config.head_channels,
            config.batch_norm,
            config.group_norm,
            config.convs_per_block,
        )
