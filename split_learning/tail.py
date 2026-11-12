from collections.abc import Sequence

from torch import Tensor, nn

from split_learning.config import SplitModelConfig, seeded


class Tail(nn.Module):
    """Institution-local end of the U: Backbone output -> class logits."""

    def __init__(self, in_channels: int, hidden: Sequence[int], num_classes: int) -> None:
        super().__init__()
        widths = [in_channels, *hidden]
        layers: list[nn.Module] = [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        for in_features, out_features in zip(widths, widths[1:]):
            layers += [nn.Linear(in_features, out_features), nn.ReLU()]
        layers.append(nn.Linear(widths[-1], num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, activations: Tensor) -> Tensor:
        return self.layers(activations)


def build_tail(config: SplitModelConfig) -> Tail:
    with seeded(config.seed + 2):
        return Tail(config.backbone_channels[-1], config.tail_hidden, config.num_classes)
