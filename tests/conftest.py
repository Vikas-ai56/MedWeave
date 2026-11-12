"""Shared fixtures: a tiny dataset in the official PathMNIST file format, provisioned like the real one.

`provisioned_data_dir` writes a 40/12/8-sample `pathmnist.npz` (random pixels, labels cycling over the
9 classes, labels stored `[N, 1]` uint8 as in MedMNIST) and provisions inst-1..inst-4 shards with seed 0
through the production `provision_partitions`, so tests exercise the real loading path offline.
"""

from pathlib import Path

import numpy as np
import pytest

from institution_data.pathmnist import FILE_NAME, NUM_CLASSES, file_md5
from institution_data.provision import DEFAULT_INSTITUTION_IDS, dataset_dir, provision_partitions

SMALL_SPLIT_SIZES = {"train": 40, "val": 12, "test": 8}


def write_official_format(path: Path, sizes: dict[str, int], seed: int = 0) -> None:
    generator = np.random.default_rng(seed)
    arrays = {}
    for split, size in sizes.items():
        arrays[f"{split}_images"] = generator.integers(0, 256, (size, 28, 28, 3), dtype=np.uint8)
        arrays[f"{split}_labels"] = (np.arange(size) % NUM_CLASSES).reshape(-1, 1).astype(np.uint8)
    with open(path, "wb") as stream:
        np.savez(stream, **arrays)


@pytest.fixture
def provisioned_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    official = dataset_dir(data_dir) / FILE_NAME
    official.parent.mkdir(parents=True)
    write_official_format(official, SMALL_SPLIT_SIZES)
    provision_partitions(official, data_dir, DEFAULT_INSTITUTION_IDS, seed=0, source_md5=file_md5(official))
    return data_dir
