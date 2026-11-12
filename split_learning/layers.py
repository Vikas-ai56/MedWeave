from collections.abc import Sequence

from torch import nn


def conv_blocks(in_channels: int, channels: Sequence[int], group_norm: bool = False) -> nn.Sequential:
    """One (3x3 conv, [GroupNorm], ReLU, 2x2 max-pool) block per entry in `channels`."""
    layers: list[nn.Module] = []
    for out_channels in channels:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if group_norm:
            layers.append(nn.GroupNorm(norm_groups(out_channels), out_channels))
        layers += [nn.ReLU(), nn.MaxPool2d(2)]
        in_channels = out_channels
    return nn.Sequential(*layers)


def norm_groups(channels: int) -> int:
    return channels // 4 if channels % 4 == 0 else 1
