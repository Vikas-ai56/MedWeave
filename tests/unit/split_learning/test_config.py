import pytest
import torch

from split_learning.backbone import build_backbone
from split_learning.config import SplitModelConfig
from split_learning.head import build_head
from split_learning.tail import build_tail


@pytest.mark.parametrize(
    "config",
    [
        SplitModelConfig(),
        SplitModelConfig(in_channels=3, image_size=(32, 24), num_classes=10),
        SplitModelConfig(head_channels=(8, 16), backbone_channels=(32,), tail_hidden=(64, 32)),
        SplitModelConfig(image_size=(64, 64), head_channels=(4,), backbone_channels=(8, 8, 8, 8)),
    ],
)
def test_models_follow_configured_shapes(config: SplitModelConfig) -> None:
    inputs = torch.randn(2, *config.input_shape)
    smashed = build_head(config)(inputs)
    output = build_backbone(config)(smashed)
    logits = build_tail(config)(output)

    assert smashed.shape[1:] == config.head_output_shape
    assert output.shape[1:] == config.backbone_output_shape
    assert logits.shape == (2, config.num_classes)


def test_moving_a_block_moves_the_split_point() -> None:
    shallow = SplitModelConfig(head_channels=(16,), backbone_channels=(32, 32))
    deep = SplitModelConfig(head_channels=(16, 32), backbone_channels=(32,))

    assert shallow.head_output_shape == (16, 14, 14)
    assert deep.head_output_shape == (32, 7, 7)
    assert shallow.backbone_output_shape == deep.backbone_output_shape


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"in_channels": 0}, "in_channels"),
        ({"num_classes": 1}, "num_classes"),
        ({"head_channels": ()}, "at least one block"),
        ({"backbone_channels": (32, 0)}, "positive"),
        ({"tail_hidden": (-1,)}, "positive"),
        ({"image_size": (4, 4), "backbone_channels": (8, 8, 8)}, "too small"),
        ({"tail_dropout": 1.0}, "tail_dropout"),
        ({"tail_dropout": -0.1}, "tail_dropout"),
    ],
)
def test_invalid_config_is_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SplitModelConfig(**kwargs)
