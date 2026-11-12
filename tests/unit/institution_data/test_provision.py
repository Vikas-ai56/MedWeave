"""Provisioned shards: each institution opens only its own data, and a mismatched shard is refused."""

import numpy as np
import pytest

from institution_data import (
    DEFAULT_INSTITUTION_IDS,
    DatasetError,
    audit_partitions,
    open_evaluation,
    open_institution_training,
    prepare_pathmnist,
)
from institution_data.pathmnist import FILE_NAME, load_split
from institution_data.provision import dataset_dir, partition_dir


def test_shards_are_disjoint_complete_and_match_the_manifest(provisioned_data_dir):
    audit = audit_partitions(provisioned_data_dir, seed=0)

    assert audit.passed
    assert [entry.num_samples for entry in audit.manifest.institutions] == [10, 10, 10, 10]


def test_institution_shard_holds_its_official_training_samples(provisioned_data_dir):
    official = load_split(dataset_dir(provisioned_data_dir) / FILE_NAME, "train")
    with np.load(partition_dir(provisioned_data_dir, 0) / "inst-2.npz") as shard:
        indices = shard["indices"]

    local = open_institution_training(provisioned_data_dir, 0, "inst-2")

    assert np.array_equal(local.split.images, official.images[indices])
    assert np.array_equal(local.split.labels, official.labels[indices])


def test_shard_that_disagrees_with_the_manifest_is_refused(provisioned_data_dir):
    shard = partition_dir(provisioned_data_dir, 0) / "inst-1.npz"
    with np.load(shard) as archive:
        images, labels, indices = archive["images"], archive["labels"], archive["indices"]
    with open(shard, "wb") as stream:
        np.savez(stream, images=images, labels=labels, indices=indices[::-1] + 1)

    with pytest.raises(DatasetError, match="does not match"):
        open_institution_training(provisioned_data_dir, 0, "inst-1")


def test_unknown_institution_and_unprovisioned_seed_are_refused(provisioned_data_dir):
    with pytest.raises(DatasetError, match="no partition"):
        open_institution_training(provisioned_data_dir, 0, "inst-9")
    with pytest.raises(DatasetError, match="prepare"):
        open_institution_training(provisioned_data_dir, 5, "inst-1")


def test_evaluation_reads_official_splits_and_never_train(provisioned_data_dir):
    assert len(open_evaluation(provisioned_data_dir, "val")) == 12
    assert len(open_evaluation(provisioned_data_dir, "test")) == 8
    with pytest.raises(ValueError, match="never 'train'"):
        open_evaluation(provisioned_data_dir, "train")


def test_prepare_reuses_an_existing_provisioning_without_downloading(provisioned_data_dir):
    manifest = prepare_pathmnist(provisioned_data_dir, DEFAULT_INSTITUTION_IDS, seed=0)
    assert manifest.total_samples == 40
