import numpy as np
import pytest
import torch
from torch import nn

from split_learning.backbone import Backbone, build_backbone
from split_learning.checkpoint import (
    get_weights,
    read_backbone_checkpoint,
    save_backbone_checkpoint,
    weights_to_state,
)
from split_learning.compute_peer import ComputePeer
from split_learning.config import SplitModelConfig
from split_learning.errors import CheckpointError


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
