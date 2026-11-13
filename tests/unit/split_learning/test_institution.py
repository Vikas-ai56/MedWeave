import copy
from functools import partial

import pytest
import torch
from torch import Tensor

from split_learning.config import SplitModelConfig
from split_learning.errors import ProtocolError, ShapeError
from split_learning.institution import Institution, create_institutions

IDS = ["inst-1", "inst-2", "inst-3", "inst-4"]


@pytest.fixture
def institution(config: SplitModelConfig) -> Institution:
    return Institution("inst-1", config)


def fake_backbone_output(config: SplitModelConfig, batch_size: int = 4) -> Tensor:
    return torch.randn(batch_size, *config.backbone_output_shape)


def run_local_cycle(institution: Institution, config: SplitModelConfig, batch) -> None:
    inputs, labels = batch
    activations = institution.head_forward("b0", inputs)
    institution.tail_step("b0", fake_backbone_output(config), labels)
    institution.head_backward("b0", torch.randn_like(activations))


def parameters_of(institution: Institution) -> list[Tensor]:
    return [*institution.head.parameters(), *institution.tail.parameters()]


def test_four_institutions_start_identical_but_share_no_storage(config: SplitModelConfig) -> None:
    institutions = list(create_institutions(IDS, config).values())

    for other in institutions[1:]:
        for a, b in zip(parameters_of(institutions[0]), parameters_of(other)):
            assert torch.equal(a, b)
            assert a.data_ptr() != b.data_ptr()


def test_training_one_institution_leaves_the_others_unchanged(config: SplitModelConfig, batch) -> None:
    institutions = create_institutions(IDS, config)
    before = {id_: copy.deepcopy(parameters_of(inst)) for id_, inst in institutions.items()}

    run_local_cycle(institutions["inst-2"], config, batch)

    for id_, inst in institutions.items():
        changed = any(not torch.equal(a, b) for a, b in zip(parameters_of(inst), before[id_]))
        assert changed == (id_ == "inst-2"), id_


def test_each_optimizer_owns_only_its_component(institution: Institution) -> None:
    def owned(optimizer: torch.optim.Optimizer) -> set[int]:
        return {id(p) for group in optimizer.param_groups for p in group["params"]}

    assert owned(institution.head_optimizer) == {id(p) for p in institution.head.parameters()}
    assert owned(institution.tail_optimizer) == {id(p) for p in institution.tail.parameters()}


def test_duplicate_institution_ids_are_rejected(config: SplitModelConfig) -> None:
    with pytest.raises(ValueError, match="unique"):
        create_institutions(["a", "a"], config)


def test_tail_step_returns_gradient_and_metrics(institution, config, batch) -> None:
    inputs, labels = batch
    institution.head_forward("b0", inputs)
    output = fake_backbone_output(config)
    result = institution.tail_step("b0", output, labels)

    assert result.gradient.shape == output.shape
    assert result.metrics.total == len(labels)
    assert 0 <= result.metrics.correct <= len(labels)
    assert result.metrics.loss > 0


def test_cycle_leaves_no_pending_batch(institution, config, batch) -> None:
    run_local_cycle(institution, config, batch)
    with pytest.raises(ProtocolError, match="no forward pass"):
        institution.head_backward("b0", torch.randn(4, *config.head_output_shape))


def test_out_of_order_calls_are_rejected(institution, config, batch) -> None:
    inputs, labels = batch
    with pytest.raises(ProtocolError):
        institution.tail_step("b0", fake_backbone_output(config), labels)

    institution.head_forward("b0", inputs)
    with pytest.raises(ProtocolError, match="still waiting"):
        institution.head_forward("b1", inputs)
    with pytest.raises(ProtocolError, match="expected batch 'b0'"):
        institution.tail_step("b1", fake_backbone_output(config), labels)

    institution.discard_pending()
    institution.head_forward("b1", inputs)


@pytest.mark.parametrize(
    "make_inputs",
    [
        lambda c: torch.randn(4, c.in_channels + 1, *c.image_size),
        lambda c: torch.randn(4, c.in_channels, 16, 16),
        lambda c: torch.randn(c.in_channels, *c.image_size),
    ],
    ids=["wrong-channels", "wrong-size", "missing-batch-dim"],
)
def test_invalid_inputs_are_rejected(institution, config, make_inputs) -> None:
    with pytest.raises(ShapeError, match="inputs"):
        institution.head_forward("b0", make_inputs(config))


def test_backbone_output_with_wrong_batch_size_is_rejected(institution, config, batch) -> None:
    inputs, labels = batch
    institution.head_forward("b0", inputs)
    with pytest.raises(ShapeError, match="backbone output"):
        institution.tail_step("b0", fake_backbone_output(config, batch_size=3), labels[:3])


@pytest.mark.parametrize(
    "bad_labels",
    [
        torch.tensor([0, 1, 2]),
        torch.tensor([0.0, 1.0, 2.0, 0.0]),
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([-1, 0, 1, 2]),
    ],
    ids=["wrong-length", "float", "too-large", "negative"],
)
def test_invalid_labels_are_rejected(institution, config, batch, bad_labels) -> None:
    institution.head_forward("b0", batch[0])
    tail_before = copy.deepcopy(institution.tail.state_dict())

    with pytest.raises(ShapeError, match="labels"):
        institution.tail_step("b0", fake_backbone_output(config), bad_labels)
    for key, value in institution.tail.state_dict().items():
        assert torch.equal(value, tail_before[key])


def test_wrong_gradient_shape_keeps_pending_batch(institution, config, batch) -> None:
    inputs, labels = batch
    activations = institution.head_forward("b0", inputs)
    institution.tail_step("b0", fake_backbone_output(config), labels)

    with pytest.raises(ShapeError, match="gradient"):
        institution.head_backward("b0", torch.randn(4, 1, 2, 2))
    institution.head_backward("b0", torch.randn_like(activations))


def test_evaluation_helpers_keep_no_graph(institution, config, batch) -> None:
    inputs, labels = batch
    activations = institution.encode(inputs)
    metrics = institution.evaluate_tail(fake_backbone_output(config), labels)

    assert not activations.requires_grad
    assert metrics.total == len(labels)
    institution.head_forward("b0", inputs)


def test_custom_loss_function_is_used_for_the_tail_step(config: SplitModelConfig, batch) -> None:
    inputs, labels = batch
    output = fake_backbone_output(config)
    losses = []
    for loss_fn in (torch.nn.functional.cross_entropy, partial(torch.nn.functional.cross_entropy, label_smoothing=0.5)):
        institution = Institution("inst-1", config, loss_fn=loss_fn)
        institution.head_forward("b0", inputs)
        losses.append(institution.tail_step("b0", output, labels).metrics.loss)

    assert losses[0] != pytest.approx(losses[1])
    assert create_institutions(["a"], config, loss_fn=losses.append)["a"].loss_fn == losses.append
