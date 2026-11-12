from collections.abc import Sequence

from torch import Tensor, nn

from split_learning.config import SplitModelConfig, seeded
from split_learning.layers import conv_blocks


class Backbone(nn.Module):
    """Shared middle of the U, executed by the assigned compute peer."""

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        batch_norm: bool = False,
        final_pool: bool = True,
        group_norm: bool = False,
        convs_per_block: int = 1,
    ) -> None:
        super().__init__()
        self.layers = conv_blocks(
            in_channels,
            channels,
            batch_norm=batch_norm,
            pool_last=final_pool,
            group_norm=group_norm,
            convs_per_block=convs_per_block,
        )

    def forward(self, activations: Tensor) -> Tensor:
        return self.layers(activations)


def build_backbone(config: SplitModelConfig) -> Backbone:
    with seeded(config.seed + 1):
        return Backbone(
            config.head_channels[-1],
            config.backbone_channels,
            config.batch_norm,
            config.backbone_final_pool,
            config.group_norm,
            config.convs_per_block,
        )
