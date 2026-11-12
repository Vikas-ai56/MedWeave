"""PathMNIST source: the pinned MedMNIST file, its integrity check, and its official splits.

Owns where the dataset comes from and how the `.npz` is read. Source pinned to the `medmnist` 3.0.2
metadata: Zenodo record 10519652, MD5 a8b06965200029087d5bd730944a56c1, 28x28 RGB, 9 tissue classes,
official splits 89,996 train / 10,004 validation / 7,180 test, licence CC BY 4.0. The test split
(CRC-VAL-HE-7K) comes from a different clinical centre than train/validation, so test accuracy is
expected to sit below validation accuracy.

Images stay uint8 NHWC here and labels become int64 `[N]`; conversion to model inputs happens only in
`institution_data.batches`. A download is written to a `.part` file and renamed only after its MD5
matches, so a partial or corrupted file is never mistaken for the dataset.
"""

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASET_NAME = "pathmnist"
URL = "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1"
MD5 = "a8b06965200029087d5bd730944a56c1"
FILE_NAME = "pathmnist.npz"
NUM_CLASSES = 9
IMAGE_SHAPE = (28, 28, 3)
SPLITS = ("train", "val", "test")
OFFICIAL_SPLIT_SIZES = {"train": 89_996, "val": 10_004, "test": 7_180}
CLASS_NAMES = (
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
)
DOWNLOAD_TIMEOUT_SECONDS = 60


class DatasetError(Exception):
    pass


@dataclass(frozen=True)
class Split:
    images: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_pathmnist(directory: Path) -> Path:
    target = directory / FILE_NAME
    if target.exists():
        actual = file_md5(target)
        if actual != MD5:
            raise DatasetError(f"{target} has MD5 {actual}, expected {MD5}; delete it and run prepare again")
        return target
    directory.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(FILE_NAME + ".part")
    with urllib.request.urlopen(URL, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, open(partial, "wb") as stream:
        shutil.copyfileobj(response, stream, length=1 << 20)
    actual = file_md5(partial)
    if actual != MD5:
        partial.unlink()
        raise DatasetError(f"downloaded {URL} has MD5 {actual}, expected {MD5}")
    partial.replace(target)
    return target


def load_split(path: Path, split: str) -> Split:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    with np.load(path) as archive:
        return as_split(archive[f"{split}_images"], archive[f"{split}_labels"], f"{path.name}:{split}")


def as_split(images: np.ndarray, labels: np.ndarray, name: str) -> Split:
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != IMAGE_SHAPE:
        raise DatasetError(f"{name} images must be uint8 (N, 28, 28, 3), got {images.dtype} {images.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise DatasetError(f"{name} labels must be integers, got {labels.dtype}")
    labels = labels.reshape(-1).astype(np.int64)
    if len(labels) != len(images):
        raise DatasetError(f"{name} has {len(images)} images but {len(labels)} labels")
    if len(labels) and (labels.min() < 0 or labels.max() >= NUM_CLASSES):
        raise DatasetError(f"{name} labels must be in [0, {NUM_CLASSES}), got {labels.min()}..{labels.max()}")
    return Split(images, labels)
