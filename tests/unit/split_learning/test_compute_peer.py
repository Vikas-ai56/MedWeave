import pytest
import torch
from torch import Tensor

from split_learning.backbone import build_backbone
from split_learning.checkpoint import StateDict
from split_learning.compute_peer import ComputePeer
from split_learning.config import SplitModelConfig
from split_learning.errors import CheckpointError, ProtocolError, ShapeError

VERSION = "v1"


@pytest.fixture
def initial_state(config: SplitModelConfig) -> StateDict:
    return build_backbone(config).state_dict()


@pytest.fixture
def peer(config: SplitModelConfig, initial_state: StateDict) -> ComputePeer:
    peer = ComputePeer(config)
    peer.load_backbone(initial_state, VERSION)
    return peer


@pytest.fixture
def activations(config: SplitModelConfig) -> Tensor:
    return torch.randn(4, *config.head_output_shape)


def output_gradient(config: SplitModelConfig) -> Tensor:
    return torch.randn(4, *config.backbone_output_shape)


def test_forward_returns_detached_backbone_output(peer, config, activations) -> None:
    output = peer.forward("b0", activations, VERSION)
    assert output.shape == (4, *config.backbone_output_shape)
    assert not output.requires_grad


def test_backward_returns_input_gradient_and_updates_backbone(peer, config, activations) -> None:
    before = peer.backbone_update().state_dict
    peer.forward("b0", activations, VERSION)
    input_gradient = peer.backward("b0", output_gradient(config), VERSION)

    assert input_gradient.shape == activations.shape
    after = peer.backbone_update()
    assert after.num_batches == 1 and after.base_version == VERSION
    assert any(not torch.equal(before[k], after.state_dict[k]) for k in before)


def test_backbone_update_is_a_snapshot(peer, config, activations) -> None:
    update = peer.backbone_update()
    peer.forward("b0", activations, VERSION)
    peer.backward("b0", output_gradient(config), VERSION)
    assert any(not torch.equal(v, peer.backbone.state_dict()[k]) for k, v in update.state_dict.items())


def test_same_checkpoint_gives_identical_peers(config, initial_state, activations) -> None:
    first, second = ComputePeer(config), ComputePeer(config)
    first.load_backbone(initial_state, VERSION)
    second.load_backbone(initial_state, VERSION)

    assert torch.equal(first.infer(activations, VERSION), second.infer(activations, VERSION))


def test_loading_resets_turn_state(peer, config, initial_state, activations) -> None:
    peer.forward("b0", activations, VERSION)
    peer.load_backbone(initial_state, "v2")

    update = peer.backbone_update()
    assert update.base_version == "v2" and update.num_batches == 0


def test_operations_require_a_loaded_backbone(config, activations) -> None:
    peer = ComputePeer(config)
    with pytest.raises(ProtocolError, match="load_backbone"):
        peer.forward("b0", activations, VERSION)
    with pytest.raises(ProtocolError, match="load_backbone"):
        peer.backbone_update()


def test_version_mismatch_is_rejected(peer, activations) -> None:
    with pytest.raises(ProtocolError, match="version"):
        peer.forward("b0", activations, "v0")


def test_second_forward_before_backward_is_rejected(peer, activations) -> None:
    peer.forward("b0", activations, VERSION)
    with pytest.raises(ProtocolError, match="b0"):
        peer.forward("b1", activations, VERSION)


def test_backward_for_unknown_batch_is_rejected(peer, config, activations) -> None:
    with pytest.raises(ProtocolError):
        peer.backward("b0", output_gradient(config), VERSION)
    peer.forward("b0", activations, VERSION)
    with pytest.raises(ProtocolError, match="expected batch 'b0'"):
        peer.backward("b1", output_gradient(config), VERSION)


def test_update_with_unfinished_batch_is_rejected(peer, activations) -> None:
    peer.forward("b0", activations, VERSION)
    with pytest.raises(ProtocolError, match="b0"):
        peer.backbone_update()
    peer.discard_pending()
    assert peer.backbone_update().num_batches == 0


@pytest.mark.parametrize(
    "bad",
    [
        torch.randn(4, 3, 4, 4),
        torch.randn(4, 4, 4),
        torch.randn(0, 4, 4, 4),
        torch.ones(4, 4, 4, 4, dtype=torch.long),
    ],
    ids=["wrong-channels", "missing-batch-dim", "empty", "integer"],
)
def test_invalid_activations_are_rejected(peer, bad) -> None:
    with pytest.raises(ShapeError):
        peer.forward("b0", bad, VERSION)


def test_bad_gradient_keeps_pending_batch(peer, config, activations) -> None:
    peer.forward("b0", activations, VERSION)
    with pytest.raises(ShapeError):
        peer.backward("b0", torch.randn(3, *config.backbone_output_shape), VERSION)

    input_gradient = peer.backward("b0", output_gradient(config), VERSION)
    assert input_gradient.shape == activations.shape


def test_incompatible_checkpoint_is_rejected(config) -> None:
    other = SplitModelConfig(
        image_size=config.image_size, head_channels=config.head_channels, backbone_channels=(12,)
    )
    peer = ComputePeer(config)
    with pytest.raises(CheckpointError, match="Backbone checkpoint does not match"):
        peer.load_backbone(build_backbone(other).state_dict(), VERSION)
    assert peer.version is None


def test_infer_keeps_no_graph_and_no_pending_batch(peer, activations) -> None:
    output = peer.infer(activations, VERSION)
    assert not output.requires_grad
    peer.forward("b0", activations, VERSION)
