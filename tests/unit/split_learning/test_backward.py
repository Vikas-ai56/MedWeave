import copy

import torch
import torch.nn.functional as F
from torch import nn

from split_learning.backbone import build_backbone
from split_learning.backward import backbone_backward, head_backward, tail_backward
from split_learning.config import SplitModelConfig
from split_learning.forward import backbone_forward, head_forward, tail_forward
from split_learning.head import build_head
from split_learning.tail import build_tail


def sgd(module: nn.Module) -> torch.optim.SGD:
    return torch.optim.SGD(module.parameters(), lr=0.1)


def run_split_backward(head, backbone, tail, inputs, labels, perturb=lambda t: t):
    smashed, head_pass = head_forward(head, inputs)
    output, backbone_pass = backbone_forward(backbone, perturb(smashed))
    tail_pass = tail_forward(tail, output, labels)

    grad_output = tail_backward(tail_pass, sgd(tail))
    grad_smashed = backbone_backward(backbone_pass, grad_output, sgd(backbone))
    head_backward(head_pass, grad_smashed, sgd(head))
    return smashed, output, grad_output, grad_smashed


def assert_same_grads(split: nn.Module, reference: nn.Module) -> None:
    for (name, a), (_, b) in zip(split.named_parameters(), reference.named_parameters()):
        assert a.grad is not None, name
        assert torch.allclose(a.grad, b.grad, atol=1e-6), name


def test_split_gradients_match_monolithic_model(config: SplitModelConfig, batch) -> None:
    inputs, labels = batch
    head, backbone, tail = build_head(config), build_backbone(config), build_tail(config)
    reference = nn.Sequential(copy.deepcopy(head), copy.deepcopy(backbone), copy.deepcopy(tail))

    run_split_backward(head, backbone, tail, inputs, labels)
    F.cross_entropy(reference(inputs), labels).backward()

    assert_same_grads(head, reference[0])
    assert_same_grads(backbone, reference[1])
    assert_same_grads(tail, reference[2])


def test_exchanged_gradients_match_activation_shapes(config: SplitModelConfig, batch) -> None:
    inputs, labels = batch
    smashed, output, grad_output, grad_smashed = run_split_backward(
        build_head(config), build_backbone(config), build_tail(config), inputs, labels
    )

    assert grad_output.shape == output.shape
    assert grad_smashed.shape == smashed.shape
    assert not grad_output.requires_grad and not grad_smashed.requires_grad


def test_additive_noise_passes_gradient_straight_through(config: SplitModelConfig, batch) -> None:
    inputs, labels = batch
    noise = torch.randn(inputs.shape[0], *config.head_output_shape) * 0.1
    head, backbone, tail = build_head(config), build_backbone(config), build_tail(config)
    reference = nn.Sequential(copy.deepcopy(head), copy.deepcopy(backbone), copy.deepcopy(tail))

    run_split_backward(head, backbone, tail, inputs, labels, perturb=lambda t: t + noise)
    noisy = reference[0](inputs) + noise
    F.cross_entropy(reference[2](reference[1](noisy)), labels).backward()

    assert_same_grads(head, reference[0])


def test_backward_updates_every_component(config: SplitModelConfig, batch) -> None:
    inputs, labels = batch
    head, backbone, tail = build_head(config), build_backbone(config), build_tail(config)
    before = [copy.deepcopy(m.state_dict()) for m in (head, backbone, tail)]

    run_split_backward(head, backbone, tail, inputs, labels)

    for module, state in zip((head, backbone, tail), before):
        changed = any(not torch.equal(p, state[k]) for k, p in module.state_dict().items())
        assert changed, type(module).__name__
