import pytest
import torch

from coordinator.checkpoint_manager.manager import (
    COORDINATOR,
    CheckpointManager,
    StaleUpdateError,
)
from split_learning.backbone import build_backbone
from split_learning.checkpoint import save_backbone_checkpoint
from split_learning.compute_peer import BackboneUpdate, ComputePeer
from split_learning.config import SplitModelConfig
from split_learning.errors import CheckpointError
from split_learning.institution import create_institutions
from split_learning.training import train_batches


@pytest.fixture
def config() -> SplitModelConfig:
    return SplitModelConfig(
        in_channels=1, image_size=(8, 8), head_channels=(4,), backbone_channels=(6,), num_classes=3
    )


@pytest.fixture
def initial_state(config):
    return build_backbone(config).state_dict()


@pytest.fixture
def manager(tmp_path, initial_state) -> CheckpointManager:
    manager = CheckpointManager(tmp_path / "checkpoints")
    manager.initialize(initial_state)
    return manager


def shifted(state, amount=1.0):
    return {key: tensor + amount for key, tensor in state.items()}


def assert_same_weights(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_empty_directory_has_no_version(tmp_path):
    manager = CheckpointManager(tmp_path)
    assert manager.latest_version is None
    with pytest.raises(CheckpointError):
        manager.latest()


def test_initialize_stores_version_zero(tmp_path, initial_state):
    manager = CheckpointManager(tmp_path)
    record = manager.initialize(initial_state)

    assert record.version == 0
    assert record.created_by == COORDINATOR
    assert record.storage_path.is_file()
    state, version = manager.latest()
    assert version == "0"
    assert_same_weights(state, initial_state)


def test_initialize_twice_is_rejected(manager, initial_state):
    with pytest.raises(CheckpointError):
        manager.initialize(initial_state)
    assert manager.latest_version == 0


def test_commit_stores_the_next_version(manager, initial_state):
    new_state = shifted(initial_state)
    record = manager.commit(BackboneUpdate("0", new_state, num_batches=3), created_by="inst-2")

    assert record.version == 1
    assert record.created_by == "inst-2"
    state, version = manager.latest()
    assert version == "1"
    assert_same_weights(state, new_state)


def test_versions_increase_by_one(manager, initial_state):
    for expected in (1, 2, 3):
        update = BackboneUpdate(str(expected - 1), shifted(initial_state, expected), num_batches=1)
        assert manager.commit(update, "inst-2").version == expected
    assert manager.latest_version == 3


@pytest.mark.parametrize("base_version", ["0", "5", "abc"])
def test_stale_update_is_rejected(manager, initial_state, base_version):
    manager.commit(BackboneUpdate("0", shifted(initial_state), 1), "inst-2")

    with pytest.raises(StaleUpdateError):
        manager.commit(BackboneUpdate(base_version, shifted(initial_state, 2), 1), "inst-3")
    assert manager.latest_version == 1
    assert_same_weights(manager.latest()[0], shifted(initial_state))


def test_update_with_missing_key_is_rejected(manager, initial_state):
    state = dict(initial_state)
    state.pop(next(iter(state)))
    with pytest.raises(CheckpointError):
        manager.commit(BackboneUpdate("0", state, 1), "inst-2")
    assert manager.latest_version == 0


def test_update_with_wrong_shape_is_rejected(manager, initial_state):
    key = next(iter(initial_state))
    state = {**initial_state, key: torch.zeros(1)}
    with pytest.raises(CheckpointError):
        manager.commit(BackboneUpdate("0", state, 1), "inst-2")
    assert manager.latest_version == 0


def test_restarted_manager_continues_from_latest(manager, initial_state):
    manager.commit(BackboneUpdate("0", shifted(initial_state), 1), "inst-2")
    manager.commit(BackboneUpdate("1", shifted(initial_state, 2), 1), "inst-3")

    restarted = CheckpointManager(manager.directory)
    assert restarted.latest_version == 2
    assert_same_weights(restarted.latest()[0], shifted(initial_state, 2))


def test_unrelated_and_temporary_files_are_ignored(tmp_path, initial_state):
    (tmp_path / "backbone_v9.pt.tmp").write_bytes(b"half written")
    (tmp_path / "notes.txt").write_text("hello")
    manager = CheckpointManager(tmp_path)
    assert manager.latest_version is None
    manager.initialize(initial_state)
    assert manager.latest_version == 0


def test_corrupt_latest_file_raises(manager):
    (manager.directory / "backbone_v0.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError):
        manager.latest()


def test_file_with_wrong_version_inside_raises(tmp_path, initial_state):
    save_backbone_checkpoint(tmp_path / "backbone_v0.pt", initial_state, version="7")
    with pytest.raises(CheckpointError):
        CheckpointManager(tmp_path).latest()


def test_turn_with_real_compute_peer(manager, config):
    """Latest checkpoint -> compute peer trains -> update committed as version 1."""
    institution = create_institutions(["inst-1"], config)["inst-1"]
    peer = ComputePeer(config)
    generator = torch.Generator().manual_seed(0)
    batches = [
        (torch.randn(4, 1, 8, 8, generator=generator), torch.randint(0, 3, (4,), generator=generator))
        for _ in range(3)
    ]

    before, version = manager.latest()
    peer.load_backbone(before, version)
    train_batches(institution, peer, batches)
    record = manager.commit(peer.backbone_update(), created_by="inst-2")

    assert record.version == 1
    after, _ = manager.latest()
    assert any(not torch.equal(before[key], after[key]) for key in before)
