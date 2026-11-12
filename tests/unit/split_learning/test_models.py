import torch

from split_learning.backbone import build_backbone
from split_learning.config import SplitModelConfig
from split_learning.head import build_head
from split_learning.tail import build_tail


def test_components_chain_to_class_logits(config: SplitModelConfig, batch) -> None:
    inputs, _ = batch
    logits = build_tail(config)(build_backbone(config)(build_head(config)(inputs)))
    assert logits.shape == (inputs.shape[0], config.num_classes)


def test_same_config_gives_identical_initial_weights(config: SplitModelConfig) -> None:
    for build in (build_head, build_backbone, build_tail):
        first, second = build(config), build(config)
        for a, b in zip(first.parameters(), second.parameters()):
            assert torch.equal(a, b)


def test_default_layers_keep_their_checkpoint_keys() -> None:
    config = SplitModelConfig()

    assert list(build_head(config).state_dict()) == ["layers.0.weight", "layers.0.bias"]
    assert list(build_backbone(config).state_dict()) == [
        "layers.0.weight", "layers.0.bias", "layers.3.weight", "layers.3.bias",
    ]
    assert list(build_tail(config).state_dict()) == ["layers.2.weight", "layers.2.bias"]


def test_building_does_not_disturb_global_rng(config: SplitModelConfig) -> None:
    torch.manual_seed(123)
    expected = torch.rand(1)
    torch.manual_seed(123)
    build_head(config)
    assert torch.equal(torch.rand(1), expected)
