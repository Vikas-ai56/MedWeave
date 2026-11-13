"""PathMNIST experiment helpers on synthetic data (no download)."""

import numpy as np
import pytest
import torch

from experiments.pathmnist.data import CLASS_NAMES, Split, batches, ensure_pathmnist, iid_partitions, load_splits
from experiments.pathmnist.metrics import classification_metrics, confusion
from experiments.pathmnist.run import flatten, predict
from split_learning.backbone import build_backbone
from split_learning.compute_peer import ComputePeer
from split_learning.institution import Institution
from split_learning.path_cnn import path_cnn_v1_config


def synthetic_split(size: int, seed: int = 0) -> Split:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randint(0, 256, (size, 3, 28, 28), dtype=torch.uint8, generator=generator)
    return Split(images, torch.arange(size) % 9)


def test_partitions_are_disjoint_complete_and_balanced() -> None:
    parts = iid_partitions(10, 4, seed=0)

    assert sorted(len(part) for part in parts) == [2, 2, 3, 3]
    assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(10))
    assert all(np.array_equal(a, b) for a, b in zip(parts, iid_partitions(10, 4, seed=0)))
    assert not all(np.array_equal(a, b) for a, b in zip(parts, iid_partitions(10, 4, seed=1)))


def test_partitioning_more_parts_than_samples_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot split"):
        iid_partitions(3, 4, seed=0)


def test_batches_are_scaled_floats_with_long_labels() -> None:
    split = synthetic_split(10)
    produced = list(batches(split, batch_size=4))

    assert [len(labels) for _, labels in produced] == [4, 4, 2]
    inputs, labels = produced[0]
    assert inputs.shape == (4, 3, 28, 28) and inputs.dtype == torch.float32 and labels.dtype == torch.long
    assert inputs.min() >= -1.0 and inputs.max() <= 1.0
    assert torch.equal(torch.cat([labels for _, labels in produced]), split.labels)


def test_shuffled_batches_cover_every_sample_once() -> None:
    split = Split(torch.zeros(10, 3, 28, 28, dtype=torch.uint8), torch.arange(10))
    labels = torch.cat([y for _, y in batches(split, 3, torch.Generator().manual_seed(0))])

    assert sorted(labels.tolist()) == list(range(10))
    assert not torch.equal(labels, split.labels)


def test_load_splits_reads_the_medmnist_layout(tmp_path) -> None:
    arrays = {}
    for name, size in [("train", 5), ("val", 2), ("test", 3)]:
        arrays[f"{name}_images"] = np.zeros((size, 28, 28, 3), dtype=np.uint8)
        arrays[f"{name}_labels"] = np.arange(size, dtype=np.uint8).reshape(-1, 1)
    np.savez(tmp_path / "pathmnist.npz", **arrays)

    splits = load_splits(tmp_path / "pathmnist.npz")

    assert splits["train"].images.shape == (5, 3, 28, 28)
    assert splits["test"].labels.tolist() == [0, 1, 2] and splits["test"].labels.dtype == torch.long


def test_file_with_wrong_checksum_is_rejected(tmp_path) -> None:
    (tmp_path / "pathmnist.npz").write_bytes(b"not the dataset")
    with pytest.raises(ValueError, match="MD5"):
        ensure_pathmnist(tmp_path)


def test_metrics_for_perfect_and_inverted_predictions() -> None:
    labels = torch.arange(18) % 9
    perfect = torch.nn.functional.one_hot(labels, 9).float() * 10

    good = classification_metrics(perfect, labels)
    bad = classification_metrics(-perfect, labels)

    assert good["accuracy"] == good["macro_f1"] == good["balanced_accuracy"] == good["auc"] == 1.0
    assert bad["accuracy"] == 0.0 and bad["auc"] == 0.0
    assert good["loss"] < 0.01 < bad["loss"]
    assert confusion(perfect, labels) == (np.eye(9, dtype=int) * 2).tolist()


def test_metrics_include_per_class_recall_when_names_are_given() -> None:
    labels = torch.arange(18) % 9
    logits = torch.nn.functional.one_hot(labels, 9).float()
    logits[labels == 2] = torch.nn.functional.one_hot(torch.tensor(3), 9).float()

    named = classification_metrics(logits, labels, CLASS_NAMES)

    assert "recall_ADI" not in classification_metrics(logits, labels)
    assert named["recall_DEB"] == 0.0 and named["recall_LYM"] == 1.0 and named["recall_TUM"] == 1.0


def test_flatten_adds_per_institution_and_mean_keys() -> None:
    flat = flatten("val", {"a": {"accuracy": 0.5}, "b": {"accuracy": 1.0}})

    assert flat == {"val/accuracy/a": 0.5, "val/accuracy/b": 1.0, "val/accuracy/mean": 0.75}


def test_predict_returns_ordered_logits_and_restores_train_mode() -> None:
    config = path_cnn_v1_config()
    institution = Institution("inst-1", config)
    peer = ComputePeer(config)
    peer.load_backbone(build_backbone(config).state_dict(), "0")
    split = synthetic_split(10)

    logits = predict(institution, peer, split, batch_size=4)
    single = predict(institution, peer, Split(split.images[7:8], split.labels[7:8]), batch_size=4)

    assert logits.shape == (10, 9) and not logits.requires_grad
    assert torch.allclose(logits[7], single[0], atol=1e-5)
    assert institution.head.training and institution.tail.training and peer.backbone.training
