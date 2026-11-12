"""Deterministic, stratified split of a training set into equal institutional partitions, and its manifest.

`stratified_partition` shuffles each class with a generator seeded from `seed`, lays the classes end to
end and deals that sequence round-robin over the institutions. So every index is assigned exactly once,
partition sizes differ by at most one (exactly equal when the count divides: 89,996 / 4 = 22,499), each
class is split across institutions to within one sample, and the same labels, ids and seed always give
the same partitions. This simulates IID institutions; it is not evidence that real hospitals look alike.

`PartitionManifest` records what was built (dataset, source MD5, seed, per-institution size and class
counts, SHA-256 of each sorted index list) as JSON, so a run can be reproduced and a shard checked.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from institution_data.pathmnist import DatasetError


@dataclass(frozen=True)
class InstitutionPartition:
    institution_id: str
    num_samples: int
    class_counts: tuple[int, ...]
    indices_sha256: str


@dataclass(frozen=True)
class PartitionManifest:
    dataset: str
    source_md5: str
    seed: int
    num_classes: int
    total_samples: int
    institutions: tuple[InstitutionPartition, ...]

    @property
    def institution_ids(self) -> tuple[str, ...]:
        return tuple(entry.institution_id for entry in self.institutions)

    def institution(self, institution_id: str) -> InstitutionPartition:
        for entry in self.institutions:
            if entry.institution_id == institution_id:
                return entry
        raise DatasetError(f"{institution_id!r} has no partition; provisioned ids are {list(self.institution_ids)}")

    def write(self, path: Path) -> None:
        staging = path.with_name(path.name + ".part")
        staging.write_text(json.dumps(asdict(self), indent=2) + "\n")
        staging.replace(path)

    @classmethod
    def read(cls, path: Path) -> "PartitionManifest":
        if not path.is_file():
            raise DatasetError(f"no partition manifest at {path}; run: python -m institution_data prepare")
        raw = json.loads(path.read_text())
        institutions = tuple(
            InstitutionPartition(
                institution_id=entry["institution_id"],
                num_samples=entry["num_samples"],
                class_counts=tuple(entry["class_counts"]),
                indices_sha256=entry["indices_sha256"],
            )
            for entry in raw["institutions"]
        )
        return cls(
            dataset=raw["dataset"],
            source_md5=raw["source_md5"],
            seed=raw["seed"],
            num_classes=raw["num_classes"],
            total_samples=raw["total_samples"],
            institutions=institutions,
        )


def stratified_partition(labels: np.ndarray, institution_ids: Sequence[str], seed: int) -> dict[str, np.ndarray]:
    if not institution_ids:
        raise ValueError("need at least one institution id")
    if len(set(institution_ids)) != len(institution_ids):
        raise ValueError(f"institution ids must be unique, got {list(institution_ids)}")
    if len(labels) < len(institution_ids):
        raise ValueError(f"cannot split {len(labels)} samples over {len(institution_ids)} institutions")
    generator = np.random.default_rng(seed)
    dealing_order = np.concatenate(
        [generator.permutation(np.flatnonzero(labels == label)) for label in np.unique(labels)]
    )
    count = len(institution_ids)
    return {
        institution_id: np.sort(dealing_order[position::count])
        for position, institution_id in enumerate(institution_ids)
    }


def check_partitions(partitions: Mapping[str, np.ndarray], total_samples: int) -> None:
    combined = np.concatenate(list(partitions.values()))
    if len(np.unique(combined)) != len(combined):
        raise DatasetError("an index is assigned to more than one institution")
    if len(combined) != total_samples or not np.array_equal(np.sort(combined), np.arange(total_samples)):
        raise DatasetError(f"partitions cover {len(combined)} indices, expected exactly 0..{total_samples - 1}")
    sizes = [len(indices) for indices in partitions.values()]
    if max(sizes) - min(sizes) > 1:
        raise DatasetError(f"partition sizes differ by more than one: {sizes}")


def index_digest(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def class_counts(labels: np.ndarray, num_classes: int) -> tuple[int, ...]:
    return tuple(int(count) for count in np.bincount(labels, minlength=num_classes))


def build_manifest(
    dataset: str,
    source_md5: str,
    seed: int,
    num_classes: int,
    labels: np.ndarray,
    partitions: Mapping[str, np.ndarray],
) -> PartitionManifest:
    return PartitionManifest(
        dataset=dataset,
        source_md5=source_md5,
        seed=seed,
        num_classes=num_classes,
        total_samples=len(labels),
        institutions=tuple(
            InstitutionPartition(
                institution_id=institution_id,
                num_samples=len(indices),
                class_counts=class_counts(labels[indices], num_classes),
                indices_sha256=index_digest(indices),
            )
            for institution_id, indices in partitions.items()
        ),
    )
