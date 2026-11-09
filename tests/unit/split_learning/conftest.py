import pytest
import torch
from torch import Tensor

from split_learning.config import SplitModelConfig

BATCH_SIZE = 4
IMAGE_SIZE = 8


@pytest.fixture
def config() -> SplitModelConfig:
    return SplitModelConfig(
        in_channels=1,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        head_channels=(4,),
        backbone_channels=(6,),
        num_classes=3,
    )


@pytest.fixture
def batch(config: SplitModelConfig) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(42)
    inputs = torch.randn(BATCH_SIZE, config.in_channels, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    labels = torch.randint(0, config.num_classes, (BATCH_SIZE,), generator=generator)
    return inputs, labels
