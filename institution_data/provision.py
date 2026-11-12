"""Provision the simulated institutions' local PathMNIST data on this machine, and open it.

Layout under the data root (`$MEDWEAVE_DATA_DIR`, default `~/.medweave/data`, kept outside the
session state so resetting `.medweave_local/` never re-downloads):

    pathmnist/pathmnist.npz                         official file, MD5-checked
    pathmnist/partition-seed<seed>/manifest.json    PartitionManifest
    pathmnist/partition-seed<seed>/<id>.npz         one institution's images, labels, official indices

`prepare_pathmnist` downloads, checks the official split sizes, partitions the training split and writes
one shard per institution; it is idempotent and is what `python -m institution_data prepare` and
`run_local.sh` call before a session. A ClientApp only ever opens its own shard
(`open_institution_training`), mirroring the deployment where each laptop holds only its own data, and
the shard is rejected if its size or index digest disagrees with the manifest. Validation and test are
never partitioned or used for training; `open_evaluation` reads them from the official file.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from institution_data.batches import ImageLabelDataset
from institution_data.partition import (
    PartitionManifest,
    build_manifest,
    check_partitions,
    index_digest,
    stratified_partition,
)
from institution_data.pathmnist import (
    DATASET_NAME,
    FILE_NAME,
    NUM_CLASSES,
    OFFICIAL_SPLIT_SIZES,
    DatasetError,
    as_split,
    download_pathmnist,
    file_md5,
    load_split,
)

DEFAULT_INSTITUTION_IDS = ("inst-1", "inst-2", "inst-3", "inst-4")
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class PartitionAudit:
    manifest: PartitionManifest
    disjoint: bool
    complete: bool
    shards_match_manifest: bool

    @property
    def passed(self) -> bool:
        return self.disjoint and self.complete and self.shards_match_manifest


def default_data_dir() -> Path:
    return Path(os.environ.get("MEDWEAVE_DATA_DIR", Path.home() / ".medweave" / "data"))


def dataset_dir(data_dir: Path) -> Path:
    return data_dir / DATASET_NAME


def partition_dir(data_dir: Path, seed: int) -> Path:
    return dataset_dir(data_dir) / f"partition-seed{seed}"


def provision_partitions(
    official_file: Path, data_dir: Path, institution_ids: Sequence[str], seed: int, source_md5: str
) -> PartitionManifest:
    train = load_split(official_file, "train")
    partitions = stratified_partition(train.labels, institution_ids, seed)
    check_partitions(partitions, len(train))
    target = partition_dir(data_dir, seed)
    target.mkdir(parents=True, exist_ok=True)
    for institution_id, indices in partitions.items():
        shard = target / f"{institution_id}.npz"
        staging = shard.with_name(shard.name + ".part")
        with open(staging, "wb") as stream:
            np.savez(stream, images=train.images[indices], labels=train.labels[indices], indices=indices)
        staging.replace(shard)
    manifest = build_manifest(DATASET_NAME, source_md5, seed, NUM_CLASSES, train.labels, partitions)
    manifest.write(target / MANIFEST_FILE)
    return manifest


def prepare_pathmnist(
    data_dir: Path, institution_ids: Sequence[str] = DEFAULT_INSTITUTION_IDS, seed: int = 0
) -> PartitionManifest:
    target = partition_dir(data_dir, seed)
    manifest_path = target / MANIFEST_FILE
    if manifest_path.is_file():
        manifest = PartitionManifest.read(manifest_path)
        shards_present = all((target / f"{institution_id}.npz").is_file() for institution_id in institution_ids)
        if manifest.institution_ids == tuple(institution_ids) and shards_present:
            return manifest
    official = download_pathmnist(dataset_dir(data_dir))
    with np.load(official) as archive:
        sizes = {split: len(archive[f"{split}_labels"]) for split in OFFICIAL_SPLIT_SIZES}
    if sizes != OFFICIAL_SPLIT_SIZES:
        raise DatasetError(f"{official} split sizes {sizes} differ from the official {OFFICIAL_SPLIT_SIZES}")
    return provision_partitions(official, data_dir, institution_ids, seed, file_md5(official))


def read_shard(data_dir: Path, seed: int, institution_id: str) -> tuple[PartitionManifest, ImageLabelDataset, np.ndarray]:
    target = partition_dir(data_dir, seed)
    manifest = PartitionManifest.read(target / MANIFEST_FILE)
    entry = manifest.institution(institution_id)
    shard = target / f"{institution_id}.npz"
    if not shard.is_file():
        raise DatasetError(f"no shard for {institution_id!r} at {shard}; run: python -m institution_data prepare")
    with np.load(shard) as archive:
        split = as_split(archive["images"], archive["labels"], shard.name)
        indices = archive["indices"]
    if len(split) != entry.num_samples or index_digest(indices) != entry.indices_sha256:
        raise DatasetError(f"{shard} does not match {target / MANIFEST_FILE}; run prepare again")
    return manifest, ImageLabelDataset(split), indices


def open_institution_training(data_dir: Path, seed: int, institution_id: str) -> ImageLabelDataset:
    _, dataset, _ = read_shard(data_dir, seed, institution_id)
    return dataset


def open_evaluation(data_dir: Path, split: str) -> ImageLabelDataset:
    if split == "train":
        raise ValueError("evaluation reads the official 'val' or 'test' split, never 'train'")
    official = dataset_dir(data_dir) / FILE_NAME
    if not official.is_file():
        raise DatasetError(f"no dataset at {official}; run: python -m institution_data prepare")
    return ImageLabelDataset(load_split(official, split))


def audit_partitions(data_dir: Path, seed: int) -> PartitionAudit:
    manifest = PartitionManifest.read(partition_dir(data_dir, seed) / MANIFEST_FILE)
    shard_indices = [read_shard(data_dir, seed, institution_id)[2] for institution_id in manifest.institution_ids]
    combined = np.concatenate(shard_indices)
    return PartitionAudit(
        manifest=manifest,
        disjoint=len(np.unique(combined)) == len(combined),
        complete=np.array_equal(np.sort(combined), np.arange(manifest.total_samples)),
        shards_match_manifest=sum(len(indices) for indices in shard_indices) == manifest.total_samples,
    )
