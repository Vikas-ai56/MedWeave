from torch import Tensor, nn

from split_learning.config import SplitModelConfig, seeded


class Tail(nn.Module):
    """Institution-local end of the U: Backbone output -> class logits."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def build_tail(config: SplitModelConfig) -> Tail:
    with seeded(config.seed + 2):
        return Tail(config.backbone_channels, config.num_classes)
