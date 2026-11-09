import copy

import numpy as np
import pytest
import torch
from torch import nn

from split_learning.backbone import Backbone, build_backbone
from split_learning.checkpoint import (
    get_weights,
    load_institution_checkpoint,
    read_backbone_checkpoint,
    save_backbone_checkpoint,
    save_institution_checkpoint,
    weights_to_state,
)
from split_learning.compute_peer import ComputePeer
from split_learning.config import SplitModelConfig
from split_learning.errors import CheckpointError
from split_learning.institution import Institution


def trained_backbone(config: SplitModelConfig) -> Backbone:
    backbone = build_backbone(config)
    with torch.no_grad():
        for param in backbone.parameters():
            param.add_(1.0)
    return backbone


def assert_same_state(a: nn.Module, b: nn.Module) -> None:
    for (key, x), y in zip(a.state_dict().items(), b.state_dict().values()):
        assert torch.equal(x, y), key


def test_backbone_file_round_trip_onto_a_new_peer(config: SplitModelConfig, tmp_path) -> None:
    trained = trained_backbone(config)
    path = tmp_path / "backbone_v3.pt"
    save_backbone_checkpoint(path, trained.state_dict(), version="3")

    peer = ComputePeer(config)
    peer.load_backbone(*read_backbone_checkpoint(path))

    assert peer.version == "3"
    assert_same_state(peer.backbone, trained)


def test_peer_update_round_trips_through_coordinator_file(config: SplitModelConfig, tmp_path) -> None:
    first = ComputePeer(config)
    first.load_backbone(trained_backbone(config).state_dict(), "3")
    update = first.backbone_update()
    save_backbone_checkpoint(tmp_path / "backbone_v4.pt", update.state_dict, version="4")

    second = ComputePeer(config)
    second.load_backbone(*read_backbone_checkpoint(tmp_path / "backbone_v4.pt"))

    assert second.version == "4"
    assert_same_state(second.backbone, first.backbone)


def test_weights_round_trip(config: SplitModelConfig) -> None:
    trained = trained_backbone(config)
    weights = get_weights(trained)

    peer = ComputePeer(config)
    peer.load_backbone(weights_to_state(peer.backbone, weights), "1")

    assert all(isinstance(array, np.ndarray) for array in weights)
    assert_same_state(peer.backbone, trained)


def test_get_weights_is_a_snapshot(config: SplitModelConfig) -> None:
    backbone = build_backbone(config)
    weights = get_weights(backbone)
    with torch.no_grad():
        next(backbone.parameters()).add_(1.0)

    assert not np.array_equal(weights[0], next(backbone.parameters()).detach().numpy())


def test_wrong_number_of_weight_arrays_is_rejected(config: SplitModelConfig) -> None:
    backbone = build_backbone(config)
    with pytest.raises(CheckpointError, match="weight arrays"):
        weights_to_state(backbone, get_weights(backbone)[:-1])


def test_incompatible_backbone_file_is_rejected(config: SplitModelConfig, tmp_path) -> None:
    wider = Backbone(config.head_channels[-1], [width * 2 for width in config.backbone_channels])
    save_backbone_checkpoint(tmp_path / "wider.pt", wider.state_dict(), version="1")

    with pytest.raises(CheckpointError, match="does not match"):
        ComputePeer(config).load_backbone(*read_backbone_checkpoint(tmp_path / "wider.pt"))


def test_corrupt_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="cannot read"):
        read_backbone_checkpoint(path)


def test_file_missing_fields_is_rejected(tmp_path) -> None:
    path = tmp_path / "partial.pt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(CheckpointError, match="version"):
        read_backbone_checkpoint(path)


def momentum_sgd(parameters) -> torch.optim.Optimizer:
    return torch.optim.SGD(parameters, lr=0.1, momentum=0.9)


def train_locally(institution: Institution, config: SplitModelConfig, batch) -> None:
    inputs, labels = batch
    activations = institution.head_forward("b0", inputs)
    institution.tail_step("b0", torch.randn(4, *config.backbone_output_shape), labels)
    institution.head_backward("b0", torch.randn_like(activations))


def test_institution_round_trip_restores_models_and_optimizers(config, batch, tmp_path) -> None:
    trained = Institution("inst-1", config, momentum_sgd)
    train_locally(trained, config, batch)
    save_institution_checkpoint(tmp_path / "inst-1.pt", trained)

    restored = Institution("inst-1", config, momentum_sgd)
    load_institution_checkpoint(tmp_path / "inst-1.pt", restored)

    assert_same_state(restored.head, trained.head)
    assert_same_state(restored.tail, trained.tail)
    for original, loaded in [
        (trained.head_optimizer, restored.head_optimizer),
        (trained.tail_optimizer, restored.tail_optimizer),
    ]:
        original_buffers = [s["momentum_buffer"] for s in original.state_dict()["state"].values()]
        loaded_buffers = [s["momentum_buffer"] for s in loaded.state_dict()["state"].values()]
        assert len(loaded_buffers) == len(original_buffers) > 0
        assert all(torch.equal(a, b) for a, b in zip(original_buffers, loaded_buffers))


def test_institution_checkpoint_for_another_institution_is_rejected(config, tmp_path) -> None:
    save_institution_checkpoint(tmp_path / "inst-1.pt", Institution("inst-1", config))
    with pytest.raises(CheckpointError, match="belongs to 'inst-1'"):
        load_institution_checkpoint(tmp_path / "inst-1.pt", Institution("inst-2", config))


def test_institution_checkpoint_with_other_architecture_is_rejected(config, tmp_path) -> None:
    save_institution_checkpoint(tmp_path / "inst-1.pt", Institution("inst-1", config))
    other = SplitModelConfig(
        image_size=config.image_size,
        head_channels=(8,),
        backbone_channels=config.backbone_channels,
        num_classes=config.num_classes,
    )
    with pytest.raises(CheckpointError, match="Head checkpoint does not match"):
        load_institution_checkpoint(tmp_path / "inst-1.pt", Institution("inst-1", other))


def test_failed_load_leaves_institution_unchanged(config, tmp_path) -> None:
    source = Institution("inst-1", config)
    with torch.no_grad():
        for param in source.head.parameters():
            param.add_(1.0)
    save_institution_checkpoint(tmp_path / "inst-1.pt", source)
    checkpoint = torch.load(tmp_path / "inst-1.pt", weights_only=True)
    checkpoint["tail"] = {"layers.2.weight": torch.zeros(1)}
    torch.save(checkpoint, tmp_path / "broken.pt")

    target = Institution("inst-1", config)
    head_before = copy.deepcopy(target.head.state_dict())
    with pytest.raises(CheckpointError, match="Tail"):
        load_institution_checkpoint(tmp_path / "broken.pt", target)

    for key, value in target.head.state_dict().items():
        assert torch.equal(value, head_before[key]), key


def test_failed_backbone_load_leaves_weights_unchanged(config) -> None:
    peer = ComputePeer(config)
    before = copy.deepcopy(peer.backbone.state_dict())
    partial = {**trained_backbone(config).state_dict(), "layers.0.bias": torch.zeros(1)}

    with pytest.raises(CheckpointError):
        peer.load_backbone(partial, "1")
    for key, value in peer.backbone.state_dict().items():
        assert torch.equal(value, before[key]), key
