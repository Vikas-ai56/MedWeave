from torch import Tensor, nn

from split_learning.config import SplitModelConfig, seeded


class Backbone(nn.Module):
    """Shared middle of the U, executed by the assigned compute peer."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def build_backbone(config: SplitModelConfig) -> Backbone:
    with seeded(config.seed + 1):
        return Backbone(config.head_channels, config.backbone_channels)
