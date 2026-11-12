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


def test_building_does_not_disturb_global_rng(config: SplitModelConfig) -> None:
    torch.manual_seed(123)
    expected = torch.rand(1)
    torch.manual_seed(123)
    build_head(config)
    assert torch.equal(torch.rand(1), expected)


def test_group_norm_keeps_shapes_and_normalises_each_block() -> None:
    config = SplitModelConfig(in_channels=3, image_size=(28, 28), num_classes=9, group_norm=True)
    head, backbone = build_head(config), build_backbone(config)

    activations = head(torch.rand(2, *config.input_shape))
    output = backbone(activations)

    assert activations.shape[1:] == config.head_output_shape and output.shape[1:] == config.backbone_output_shape
    norms = [m for m in [*head.modules(), *backbone.modules()] if isinstance(m, torch.nn.GroupNorm)]
    assert [m.num_groups for m in norms] == [4, 8, 8]
