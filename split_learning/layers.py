from collections.abc import Sequence

from torch import nn


def conv_blocks(in_channels: int, channels: Sequence[int]) -> nn.Sequential:
    """One (3x3 conv, ReLU, 2x2 max-pool) block per entry in `channels`."""
    layers: list[nn.Module] = []
    for out_channels in channels:
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ]
        in_channels = out_channels
    return nn.Sequential(*layers)
