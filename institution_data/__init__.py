"""Institutional training data for MedWeave (Member 3): PathMNIST in, four local DataLoaders out.

Public interface (everything else is internal):
- `prepare_pathmnist(data_dir, institution_ids, seed)`: download the pinned file, provision one
  stratified, disjoint, equal shard per institution plus a manifest; idempotent.
  CLI: `python -m institution_data prepare`.
- `open_institution_training(data_dir, seed, institution_id)` and `training_loader(dataset, batch_size,
  seed, round_number, num_batches)`: the batches one institution trains on in one turn.
- `open_evaluation(data_dir, "val" | "test")` and `evaluation_loader(dataset, batch_size)`.
- `default_data_dir()`: `$MEDWEAVE_DATA_DIR`, default `~/.medweave/data`.

Batches are `(images, labels)`: float32 `[B, 3, 28, 28]` in [0, 1] and int64 `[B]` in 0..8. Both stay
with the data-owning institution. Design and rules: docs/features/data-and-privacy.md.
"""

from institution_data.batches import ImageLabelDataset, evaluation_loader, training_loader
from institution_data.partition import PartitionManifest, stratified_partition
from institution_data.pathmnist import CLASS_NAMES, NUM_CLASSES, OFFICIAL_SPLIT_SIZES, DatasetError
from institution_data.provision import (
    DEFAULT_INSTITUTION_IDS,
    PartitionAudit,
    audit_partitions,
    default_data_dir,
    open_evaluation,
    open_institution_training,
    prepare_pathmnist,
)

__all__ = [
    "CLASS_NAMES",
    "DEFAULT_INSTITUTION_IDS",
    "NUM_CLASSES",
    "OFFICIAL_SPLIT_SIZES",
    "DatasetError",
    "ImageLabelDataset",
    "PartitionAudit",
    "PartitionManifest",
    "audit_partitions",
    "default_data_dir",
    "evaluation_loader",
    "open_evaluation",
    "open_institution_training",
    "prepare_pathmnist",
    "stratified_partition",
    "training_loader",
]
