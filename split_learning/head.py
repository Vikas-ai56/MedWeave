from torch import Tensor, nn

from split_learning.config import SplitModelConfig, seeded


class Head(nn.Module):
    """Institution-local front of the U: raw input -> smashed activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def build_head(config: SplitModelConfig) -> Head:
    with seeded(config.seed):
        return Head(config.in_channels, config.head_channels)
