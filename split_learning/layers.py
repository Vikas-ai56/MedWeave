from collections.abc import Sequence

from torch import nn

from split_learning.config import NORM_GROUPS


def conv_blocks(
    in_channels: int,
    channels: Sequence[int],
    *,
    batch_norm: bool = False,
    pool_last: bool = True,
    group_norm: bool = False,
    convs_per_block: int = 1,
) -> nn.Sequential:
    """One block per entry in `channels`: `convs_per_block` x (3x3 conv, [norm], ReLU), then 2x2 max-pool.

    The norm is BatchNorm2d with `batch_norm`, GroupNorm with `group_norm`.
    With `pool_last=False` the final block keeps its spatial size.
    """
    layers: list[nn.Module] = []
    for index, out_channels in enumerate(channels):
        for _ in range(convs_per_block):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            if group_norm:
                layers.append(nn.GroupNorm(NORM_GROUPS, out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        if pool_last or index < len(channels) - 1:
            layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

