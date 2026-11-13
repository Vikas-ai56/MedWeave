"""Split CNNs for PathMNIST (3x28x28 RGB, 9 classes).

MedWeavePathCNN-v1:

    stage 1   Conv(3->32)   BN ReLU MaxPool   [B, 32, 14, 14]
    stage 2   Conv(32->64)  BN ReLU MaxPool   [B, 64, 7, 7]
    stage 3   Conv(64->128) BN ReLU           [B, 128, 7, 7]
    tail      AvgPool(1) Flatten Dropout(0.2) Linear(128->9) -> logits

MedWeavePathCNN-v2 has the same stages with two (Conv, BN, ReLU) units each,
about 0.29M parameters, and the same boundary shapes as v1.

By default stage 1 is the Head and the remaining stages are the Backbone.
`norm="group"` swaps every BatchNorm for GroupNorm with `NORM_GROUPS` groups,
which keeps no running statistics; the Flower system uses it, because a shared
Backbone's BatchNorm statistics follow whichever institution trained last.
"""

from dataclasses import replace
from typing import Literal

from split_learning.config import SplitModelConfig

PATH_CNN_V1_STAGES = (32, 64, 128)
PATH_CNN_V1_HEAD_STAGES = 1
PATHMNIST_CHANNELS = 3
PATHMNIST_IMAGE_SIZE = (28, 28)
PATHMNIST_CLASSES = 9


Norm = Literal["batch", "group"]


def path_cnn_v1_config(
    head_stages: int = PATH_CNN_V1_HEAD_STAGES, seed: int = 0, norm: Norm = "batch"
) -> SplitModelConfig:
    """Config for MedWeavePathCNN-v1 with the first `head_stages` conv stages in the Head.

    `head_stages` can be 1 or 2; the Backbone keeps the rest, so its output is
    always [B, 128, 7, 7] and the Tail is the same at either split point.
    """
    if not 1 <= head_stages < len(PATH_CNN_V1_STAGES):
        raise ValueError(
            f"head_stages must be between 1 and {len(PATH_CNN_V1_STAGES) - 1}, got {head_stages}"
        )
    if norm not in ("batch", "group"):
        raise ValueError(f"norm must be 'batch' or 'group', got {norm!r}")
    return SplitModelConfig(
        in_channels=PATHMNIST_CHANNELS,
        image_size=PATHMNIST_IMAGE_SIZE,
        head_channels=PATH_CNN_V1_STAGES[:head_stages],
        backbone_channels=PATH_CNN_V1_STAGES[head_stages:],
        num_classes=PATHMNIST_CLASSES,
        seed=seed,
        batch_norm=norm == "batch",
        group_norm=norm == "group",
        backbone_final_pool=False,
        tail_dropout=0.2,
    )


def path_cnn_v2_config(
    head_stages: int = PATH_CNN_V1_HEAD_STAGES, seed: int = 0, norm: Norm = "batch"
) -> SplitModelConfig:
    """MedWeavePathCNN-v2: v1 with two conv units per stage. Not checkpoint-compatible with v1."""
    return replace(path_cnn_v1_config(head_stages, seed, norm), convs_per_block=2)
