"""PathMNIST loading, IID partitioning and batching for local experiments.

The `.npz` file is the official MedMNIST release on Zenodo. Images stay
`uint8` in memory and are converted per batch to float in [-1, 1].
"""

import hashlib
import shutil
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

PATHMNIST_URL = "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1"
PATHMNIST_MD5 = "a8b06965200029087d5bd730944a56c1"
SPLITS = ("train", "val", "test")
CLASS_NAMES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
"""Label order of PathMNIST: adipose, background, debris, lymphocytes, mucus,
smooth muscle, normal colon mucosa, cancer-associated stroma, adenocarcinoma epithelium."""


@dataclass(frozen=True)
class Split:
    """Images as uint8 `[N, 3, 28, 28]`, labels as int64 `[N]`."""

    images: Tensor
    labels: Tensor

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, indices: np.ndarray) -> "Split":
        index = torch.from_numpy(indices)
        return Split(self.images[index], self.labels[index])


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_pathmnist(data_dir: Path) -> Path:
    """Return the path to a verified `pathmnist.npz`, downloading it if missing."""
    path = data_dir / "pathmnist.npz"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".npz.part")
        with urllib.request.urlopen(PATHMNIST_URL) as response, partial.open("wb") as file:
            shutil.copyfileobj(response, file)
        partial.replace(path)
    checksum = md5_of(path)
    if checksum != PATHMNIST_MD5:
        raise ValueError(f"{path} has MD5 {checksum}, expected {PATHMNIST_MD5}; delete it and retry")
    return path


def load_splits(path: Path) -> dict[str, Split]:
    with np.load(path) as archive:
        return {
            name: Split(
                images=torch.from_numpy(archive[f"{name}_images"]).permute(0, 3, 1, 2).contiguous(),
                labels=torch.from_numpy(archive[f"{name}_labels"].reshape(-1).astype(np.int64)),
            )
            for name in SPLITS
        }


def iid_partitions(num_samples: int, num_parts: int, seed: int) -> list[np.ndarray]:
    """Disjoint, shuffled index sets whose sizes differ by at most one."""
    if not 1 <= num_parts <= num_samples:
        raise ValueError(f"cannot split {num_samples} samples into {num_parts} parts")
    order = np.random.default_rng(seed).permutation(num_samples)
    return [np.sort(part) for part in np.array_split(order, num_parts)]


def to_model_input(images: Tensor) -> Tensor:
    return images.float().div_(127.5).sub_(1.0)


Augmentation = Callable[[Tensor, torch.Generator | None], Tensor]


def batches(
    split: Split,
    batch_size: int,
    generator: torch.Generator | None = None,
    augment: Augmentation | None = None,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Float inputs and long labels; shuffled when a `generator` is given.

    `augment` is applied to each input batch, drawing from the same generator.
    """
    order = torch.randperm(len(split), generator=generator) if generator else torch.arange(len(split))
    for start in range(0, len(split), batch_size):
        index = order[start : start + batch_size]
        inputs = to_model_input(split.images[index])
        yield (augment(inputs, generator) if augment else inputs), split.labels[index]
