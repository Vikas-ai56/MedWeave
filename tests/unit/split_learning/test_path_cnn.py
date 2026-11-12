"""MedWeavePathCNN-v1 on synthetic PathMNIST-shaped tensors (no dataset download)."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from split_learning.backbone import build_backbone
from split_learning.checkpoint import (
    StateDict,
    get_weights,
    read_backbone_checkpoint,
    save_backbone_checkpoint,
    weights_to_state,
)
from split_learning.compute_peer import ComputePeer
from split_learning.config import SplitModelConfig
from split_learning.errors import CheckpointError
from split_learning.head import build_head
from split_learning.institution import Institution, create_institutions
from split_learning.path_cnn import path_cnn_v1_config
from split_learning.tail import build_tail
from split_learning.training import evaluate, train_batches, train_step

IDS = ["inst-1", "inst-2", "inst-3", "inst-4"]
BATCH_SIZE = 8


@pytest.fixture
def v1() -> SplitModelConfig:
    return path_cnn_v1_config()


def sgd(parameters) -> torch.optim.Optimizer:
    return torch.optim.SGD(parameters, lr=0.1)


def synthetic_batches(config: SplitModelConfig, count: int, seed: int = 0) -> list[tuple[Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(BATCH_SIZE, *config.input_shape, generator=generator),
            torch.randint(0, config.num_classes, (BATCH_SIZE,), generator=generator),
        )
        for _ in range(count)
    ]


def loaded_peer(config: SplitModelConfig, state: StateDict | None = None, version: str = "0") -> ComputePeer:
    peer = ComputePeer(config, sgd)
    peer.load_backbone(state if state is not None else build_backbone(config).state_dict(), version)
    return peer


def layer_names(module: nn.Module) -> list[str]:
    return [type(layer).__name__ for layer in module.layers]


def snapshot(module: nn.Module) -> StateDict:
    return copy.deepcopy(module.state_dict())


def changed_keys(module: nn.Module, before: StateDict) -> set[str]:
    return {key for key, value in module.state_dict().items() if not torch.equal(value, before[key])}


def test_layers_follow_the_v1_specification(v1: SplitModelConfig) -> None:
    head, backbone, tail = build_head(v1), build_backbone(v1), build_tail(v1)

    assert layer_names(head) == ["Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d"]
    assert layer_names(backbone) == [
        "Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d",
        "Conv2d", "BatchNorm2d", "ReLU",
    ]
    assert layer_names(tail) == ["AdaptiveAvgPool2d", "Flatten", "Dropout", "Linear"]

    convs = [layer for module in (head, backbone) for layer in module.layers if isinstance(layer, nn.Conv2d)]
    assert [(c.in_channels, c.out_channels, c.kernel_size, c.padding) for c in convs] == [
        (3, 32, (3, 3), (1, 1)),
        (32, 64, (3, 3), (1, 1)),
        (64, 128, (3, 3), (1, 1)),
    ]
    norms = [layer.num_features for module in (head, backbone) for layer in module.layers if isinstance(layer, nn.BatchNorm2d)]
    assert norms == [32, 64, 128]
    dropout, classifier = tail.layers[2], tail.layers[3]
    assert dropout.p == 0.2
    assert (classifier.in_features, classifier.out_features) == (128, 9)


@pytest.mark.parametrize("batch_size", [1, 16])
def test_tensor_shapes_at_every_boundary(v1: SplitModelConfig, batch_size: int) -> None:
    inputs = torch.randn(batch_size, 3, 28, 28)

    smashed = build_head(v1)(inputs)
    output = build_backbone(v1)(smashed)
    logits = build_tail(v1)(output)

    assert v1.input_shape == (3, 28, 28)
    assert smashed.shape == (batch_size, 32, 14, 14) == (batch_size, *v1.head_output_shape)
    assert output.shape == (batch_size, 128, 7, 7) == (batch_size, *v1.backbone_output_shape)
    assert logits.shape == (batch_size, 9)


def test_cross_entropy_on_raw_logits(v1: SplitModelConfig) -> None:
    (inputs, labels), = synthetic_batches(v1, 1)
    model = nn.Sequential(build_head(v1), build_backbone(v1), build_tail(v1)).eval()

    logits = model(inputs)
    loss = F.cross_entropy(logits, labels)

    assert not torch.allclose(logits.sum(dim=1), torch.ones(BATCH_SIZE)), "model must not apply softmax"
    assert torch.isfinite(loss) and loss.item() > 0


def test_split_step_matches_monolithic_gradients(v1: SplitModelConfig) -> None:
    (inputs, labels), = synthetic_batches(v1, 1)
    institution, peer = Institution("inst-1", v1, sgd), loaded_peer(v1)
    reference = nn.Sequential(
        copy.deepcopy(institution.head), copy.deepcopy(peer.backbone), copy.deepcopy(institution.tail)
    )

    torch.manual_seed(0)  # same dropout mask in both runs
    train_step(institution, peer, inputs, labels)
    torch.manual_seed(0)
    F.cross_entropy(reference(inputs), labels).backward()

    split = nn.ModuleList([institution.head, peer.backbone, institution.tail])
    for (name, a), b in zip(split.named_parameters(), reference.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-6), name
    for (name, a), b in zip(split.named_buffers(), reference.buffers()):
        assert torch.allclose(a, b), name


def test_train_step_updates_parameters_and_batchnorm_statistics(v1: SplitModelConfig) -> None:
    institution, peer = Institution("inst-1", v1, sgd), loaded_peer(v1)
    modules = {"head": institution.head, "backbone": peer.backbone, "tail": institution.tail}
    before = {name: snapshot(module) for name, module in modules.items()}

    metrics = train_step(institution, peer, *synthetic_batches(v1, 1)[0])

    assert metrics.total == BATCH_SIZE and metrics.loss > 0
    for name, module in modules.items():
        changed = changed_keys(module, before[name])
        weights = {key for key, _ in module.named_parameters() if key.endswith("weight")}
        assert weights <= changed, name
        running = {key for key, _ in module.named_buffers()}
        assert running <= changed, name
    assert int(institution.head.layers[1].num_batches_tracked) == 1


def test_four_institutions_have_independent_heads_and_tails(v1: SplitModelConfig) -> None:
    institutions = create_institutions(IDS, v1, sgd)

    def state_of(institution: Institution) -> dict[str, Tensor]:
        return {
            **{f"head.{k}": v for k, v in institution.head.state_dict().items()},
            **{f"tail.{k}": v for k, v in institution.tail.state_dict().items()},
        }

    reference = state_of(institutions["inst-1"])
    for id_ in IDS[1:]:
        for key, value in state_of(institutions[id_]).items():
            assert torch.equal(value, reference[key]), (id_, key)
            assert value.data_ptr() != reference[key].data_ptr(), (id_, key)

    optimized = [
        {id(p) for opt in (inst.head_optimizer, inst.tail_optimizer) for g in opt.param_groups for p in g["params"]}
        for inst in institutions.values()
    ]
    assert sum(map(len, optimized)) == len(set().union(*optimized))

    before = {id_: copy.deepcopy(state_of(inst)) for id_, inst in institutions.items()}
    train_step(institutions["inst-3"], loaded_peer(v1), *synthetic_batches(v1, 1)[0])

    for id_, inst in institutions.items():
        after = state_of(inst)
        changed = any(not torch.equal(after[key], before[id_][key]) for key in after)
        assert changed == (id_ == "inst-3"), id_


def test_backbone_checkpoint_round_trip_keeps_batchnorm_statistics(v1: SplitModelConfig, tmp_path) -> None:
    trainer = loaded_peer(v1)
    train_batches(Institution("inst-1", v1, sgd), trainer, synthetic_batches(v1, 2))
    update = trainer.backbone_update()
    save_backbone_checkpoint(tmp_path / "backbone_v1.pt", update.state_dict, version="1")

    from_file = loaded_peer(v1, *read_backbone_checkpoint(tmp_path / "backbone_v1.pt"))
    from_arrays = loaded_peer(v1, weights_to_state(trainer.backbone, get_weights(trainer.backbone)), "1")

    assert {"layers.1.running_mean", "layers.1.running_var", "layers.1.num_batches_tracked"} <= set(update.state_dict)
    for peer in (from_file, from_arrays):
        assert peer.version == "1"
        for key, value in trainer.backbone.state_dict().items():
            loaded = peer.backbone.state_dict()[key]
            assert loaded.dtype == value.dtype and torch.equal(loaded, value), key
    assert int(from_file.backbone.state_dict()["layers.1.num_batches_tracked"]) == 2


@pytest.mark.parametrize(
    "other",
    [SplitModelConfig(in_channels=3, num_classes=9), path_cnn_v1_config(head_stages=2)],
    ids=["placeholder-cnn", "other-split-point"],
)
def test_backbone_checkpoint_from_another_architecture_is_rejected(
    v1: SplitModelConfig, other: SplitModelConfig, tmp_path
) -> None:
    save_backbone_checkpoint(tmp_path / "other.pt", build_backbone(other).state_dict(), version="1")

    with pytest.raises(CheckpointError, match="does not match"):
        loaded_peer(v1, *read_backbone_checkpoint(tmp_path / "other.pt"))


def test_four_sequential_turns_share_one_canonical_backbone(v1: SplitModelConfig, tmp_path) -> None:
    institutions = create_institutions(IDS, v1, sgd)
    peers = {id_: ComputePeer(v1, sgd) for id_ in IDS}
    path = tmp_path / "canonical.pt"
    save_backbone_checkpoint(path, build_backbone(v1).state_dict(), version="0")

    for turn, owner_id in enumerate(IDS):
        compute_peer = peers[IDS[(turn + 1) % len(IDS)]]
        compute_peer.load_backbone(*read_backbone_checkpoint(path))
        train_batches(institutions[owner_id], compute_peer, synthetic_batches(v1, 2, seed=turn))
        update = compute_peer.backbone_update()

        assert update.base_version == str(turn) and update.num_batches == 2
        save_backbone_checkpoint(path, update.state_dict, version=str(turn + 1))

    canonical, version = read_backbone_checkpoint(path)
    assert version == "4"
    assert int(canonical["layers.1.num_batches_tracked"]) == 8
    tails = [institutions[id_].tail.state_dict()["layers.3.weight"] for id_ in IDS]
    assert all(not torch.equal(tails[0], other) for other in tails[1:])


def test_split_point_can_move_after_stage_two(v1: SplitModelConfig) -> None:
    deep = path_cnn_v1_config(head_stages=2)

    assert deep.head_output_shape == (64, 7, 7)
    assert deep.backbone_output_shape == v1.backbone_output_shape == (128, 7, 7)
    assert layer_names(build_head(deep)) == ["Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d"] * 2
    assert layer_names(build_backbone(deep)) == ["Conv2d", "BatchNorm2d", "ReLU"]
    for key, value in build_tail(v1).state_dict().items():
        assert torch.equal(build_tail(deep).state_dict()[key], value), key

    metrics = train_step(Institution("inst-1", deep, sgd), loaded_peer(deep), *synthetic_batches(deep, 1)[0])
    assert metrics.total == BATCH_SIZE


@pytest.mark.parametrize("head_stages", [0, 3])
def test_invalid_split_point_is_rejected(head_stages: int) -> None:
    with pytest.raises(ValueError, match="head_stages"):
        path_cnn_v1_config(head_stages=head_stages)


def test_training_lowers_loss_and_evaluation_is_deterministic(v1: SplitModelConfig) -> None:
    torch.manual_seed(0)
    institution, peer = Institution("inst-1", v1, sgd), loaded_peer(v1)
    batches = synthetic_batches(v1, 2)
    train_batches(institution, peer, batches)  # warm up BatchNorm running statistics
    first = evaluate(institution, peer, batches)

    for _ in range(10):
        train_batches(institution, peer, batches)

    assert evaluate(institution, peer, batches) == evaluate(institution, peer, batches)
    assert evaluate(institution, peer, batches).loss < first.loss
