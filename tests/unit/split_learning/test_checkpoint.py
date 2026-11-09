import numpy as np
import pytest
import torch
from torch import nn

from split_learning.backbone import Backbone, build_backbone
from split_learning.checkpoint import get_weights, load_checkpoint, save_checkpoint, set_weights
from split_learning.config import SplitModelConfig


def perturbed(config: SplitModelConfig) -> Backbone:
    backbone = build_backbone(config)
    with torch.no_grad():
        for param in backbone.parameters():
            param.add_(1.0)
    return backbone


def assert_same_state(a: nn.Module, b: nn.Module) -> None:
    for (key, x), y in zip(a.state_dict().items(), b.state_dict().values()):
        assert torch.equal(x, y), key


def test_checkpoint_file_round_trip(config: SplitModelConfig, tmp_path) -> None:
    trained = perturbed(config)
    path = tmp_path / "backbone_v3.pt"
    save_checkpoint(trained, path, version=3)

    fresh = build_backbone(config)
    version = load_checkpoint(fresh, path)

    assert version == 3
    assert_same_state(fresh, trained)


def test_weights_round_trip(config: SplitModelConfig) -> None:
    trained = perturbed(config)
    weights = get_weights(trained)

    fresh = build_backbone(config)
    set_weights(fresh, weights)

    assert all(isinstance(array, np.ndarray) for array in weights)
    assert_same_state(fresh, trained)


def test_get_weights_is_a_snapshot(config: SplitModelConfig) -> None:
    backbone = build_backbone(config)
    weights = get_weights(backbone)
    with torch.no_grad():
        next(backbone.parameters()).add_(1.0)

    assert not np.array_equal(weights[0], next(backbone.parameters()).detach().numpy())


def test_set_weights_rejects_wrong_length(config: SplitModelConfig) -> None:
    backbone = build_backbone(config)
    with pytest.raises(ValueError):
        set_weights(backbone, get_weights(backbone)[:-1])


def test_loading_into_mismatched_architecture_fails(config: SplitModelConfig, tmp_path) -> None:
    path = tmp_path / "backbone.pt"
    save_checkpoint(build_backbone(config), path, version=1)
    wider = Backbone(config.head_channels[-1], [width * 2 for width in config.backbone_channels])

    with pytest.raises(RuntimeError):
        load_checkpoint(wider, path)
