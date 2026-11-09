from split_learning.backbone import build_backbone
from split_learning.config import SplitModelConfig
from split_learning.forward import backbone_forward, head_forward, tail_forward
from split_learning.head import build_head
from split_learning.tail import build_tail


def test_head_forward_sends_detached_activations(config: SplitModelConfig, batch) -> None:
    inputs, _ = batch
    smashed, head_pass = head_forward(build_head(config), inputs)

    assert not smashed.requires_grad
    assert head_pass.activations.requires_grad
    assert smashed.shape == (inputs.shape[0], config.head_channels, 4, 4)


def test_backbone_forward_returns_detached_output(config: SplitModelConfig, batch) -> None:
    inputs, _ = batch
    smashed, _ = head_forward(build_head(config), inputs)
    output, backbone_pass = backbone_forward(build_backbone(config), smashed)

    assert not output.requires_grad
    assert backbone_pass.received.requires_grad and backbone_pass.received.is_leaf
    assert output.shape == (inputs.shape[0], config.backbone_channels, 2, 2)


def test_tail_forward_computes_loss_from_local_labels(config: SplitModelConfig, batch) -> None:
    inputs, labels = batch
    smashed, _ = head_forward(build_head(config), inputs)
    output, _ = backbone_forward(build_backbone(config), smashed)
    tail_pass = tail_forward(build_tail(config), output, labels)

    assert tail_pass.logits.shape == (inputs.shape[0], config.num_classes)
    assert tail_pass.loss.ndim == 0 and tail_pass.loss.item() > 0
