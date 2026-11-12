"""Stratified partitioning: complete, disjoint, equal, class-balanced and reproducible."""

import numpy as np
import pytest

from institution_data.partition import PartitionManifest, build_manifest, check_partitions, stratified_partition
from institution_data.pathmnist import DatasetError

IDS = ["inst-1", "inst-2", "inst-3", "inst-4"]
PATHMNIST_TRAIN_SIZE = 89_996


def imbalanced_labels(size: int, seed: int = 0) -> np.ndarray:
    weights = np.array([10, 11, 12, 10, 9, 14, 9, 11, 14], dtype=float)
    return np.random.default_rng(seed).choice(9, size=size, p=weights / weights.sum())


def test_pathmnist_sized_training_set_splits_into_four_equal_disjoint_partitions():
    labels = imbalanced_labels(PATHMNIST_TRAIN_SIZE)
    partitions = stratified_partition(labels, IDS, seed=0)

    assert [len(partitions[i]) for i in IDS] == [22_499] * 4
    combined = np.concatenate(list(partitions.values()))
    assert np.array_equal(np.sort(combined), np.arange(PATHMNIST_TRAIN_SIZE))
    check_partitions(partitions, PATHMNIST_TRAIN_SIZE)


def test_each_class_is_split_to_within_one_sample():
    labels = imbalanced_labels(PATHMNIST_TRAIN_SIZE)
    partitions = stratified_partition(labels, IDS, seed=0)

    per_institution = np.stack([np.bincount(labels[partitions[i]], minlength=9) for i in IDS])
    assert (per_institution.max(axis=0) - per_institution.min(axis=0) <= 1).all()


def test_same_seed_reproduces_and_another_seed_differs():
    labels = imbalanced_labels(1_000)
    first, again = stratified_partition(labels, IDS, 7), stratified_partition(labels, IDS, 7)
    other = stratified_partition(labels, IDS, 8)

    assert all(np.array_equal(first[i], again[i]) for i in IDS)
    assert any(not np.array_equal(first[i], other[i]) for i in IDS)


def test_sizes_differ_by_at_most_one_when_count_does_not_divide():
    partitions = stratified_partition(imbalanced_labels(1_003), IDS, 0)
    sizes = sorted(len(indices) for indices in partitions.values())
    assert sizes == [250, 251, 251, 251]


@pytest.mark.parametrize("ids", [[], ["inst-1", "inst-1"]])
def test_invalid_institution_ids_are_rejected(ids):
    with pytest.raises(ValueError):
        stratified_partition(imbalanced_labels(10), ids, 0)


def test_overlapping_partitions_fail_the_check():
    with pytest.raises(DatasetError, match="more than one"):
        check_partitions({"inst-1": np.array([0, 1]), "inst-2": np.array([1, 2])}, 3)


def test_manifest_round_trips_through_json(tmp_path):
    labels = imbalanced_labels(100)
    partitions = stratified_partition(labels, IDS, 3)
    manifest = build_manifest("pathmnist", "abc", 3, 9, labels, partitions)
    manifest.write(tmp_path / "manifest.json")

    restored = PartitionManifest.read(tmp_path / "manifest.json")
    assert restored == manifest
    assert sum(restored.institution("inst-2").class_counts) == len(partitions["inst-2"])
    with pytest.raises(DatasetError, match="no partition"):
        restored.institution("inst-9")
