from collections.abc import Sequence

from torch import nn


def conv_blocks(
    in_channels: int,
    channels: Sequence[int],
    batch_norm: bool = False,
    pool_last: bool = True,
) -> nn.Sequential:
    """One (3x3 conv, [BatchNorm], ReLU, 2x2 max-pool) block per entry in `channels`.

    With `pool_last=False` the final block keeps its spatial size.
    """
    layers: list[nn.Module] = []
    for index, out_channels in enumerate(channels):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        if pool_last or index < len(channels) - 1:
            layers.append(nn.MaxPool2d(2))
        in_channels = out_channels
    return nn.Sequential(*layers)
