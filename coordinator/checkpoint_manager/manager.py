"""Canonical Backbone checkpoint store.

The coordinator keeps one canonical Backbone (SPEC §2.3). Each version is a
file `backbone_v{N}.pt` in the session's checkpoint directory. Version 0 is the
initial Backbone, and each accepted training turn adds the next version.

Versions are integers here and travel as `str(version)` in payloads, which is
what `ComputePeer` expects. File reading and writing is done by
`split_learning.checkpoint`.
"""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from torch import Tensor

from split_learning.checkpoint import StateDict, read_backbone_checkpoint, save_backbone_checkpoint
from split_learning.compute_peer import BackboneUpdate
from split_learning.errors import CheckpointError

COORDINATOR = "coordinator"
_FILE_PATTERN = re.compile(r"backbone_v(\d+)\.pt")


class StaleUpdateError(CheckpointError):
    """An update was trained from a Backbone version that is no longer the latest."""


@dataclass(frozen=True)
class CheckpointRecord:
    """Metadata for one stored version, matching the `model_checkpoints` table."""

    version: int
    storage_path: Path
    created_by: str
    created_at: datetime


class CheckpointManager:
    """Stores Backbone versions and accepts only updates built on the latest one.

    The latest version is found from the files in `directory`, so a restarted
    coordinator carries on from where it stopped.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        versions = [
            int(match.group(1))
            for path in self.directory.iterdir()
            if (match := _FILE_PATTERN.fullmatch(path.name))
        ]
        self._latest_version: int | None = max(versions, default=None)

    @property
    def latest_version(self) -> int | None:
        """The newest stored version, or None before `initialize`."""
        return self._latest_version

    def initialize(self, state_dict: Mapping[str, Tensor]) -> CheckpointRecord:
        """Store the initial Backbone as version 0."""
        if self._latest_version is not None:
            raise CheckpointError(
                f"{self.directory} already holds Backbone version {self._latest_version}"
            )
        return self._store(0, state_dict, COORDINATOR)

    def latest(self) -> tuple[StateDict, str]:
        """Return `(state_dict, version)` to pass to `ComputePeer.load_backbone`."""
        version = self._require_latest()
        state_dict, stored_version = read_backbone_checkpoint(self._path(version))
        if stored_version != str(version):
            raise CheckpointError(
                f"{self._path(version).name} contains version {stored_version!r}, expected {version}"
            )
        return state_dict, stored_version

    def commit(self, update: BackboneUpdate, created_by: str) -> CheckpointRecord:
        """Store a compute peer's update as the next version.

        Raises `StaleUpdateError` if the update was not trained from the latest
        version, and `CheckpointError` if its weights do not match the stored
        Backbone's keys and shapes. Nothing is stored in either case.
        """
        current_state, current_version = self.latest()
        if update.base_version != current_version:
            raise StaleUpdateError(
                f"update from {created_by!r} is based on version {update.base_version!r}, "
                f"latest is {current_version!r}"
            )
        _check_same_layout(update.state_dict, current_state)
        return self._store(int(current_version) + 1, update.state_dict, created_by)

    def _store(self, version: int, state_dict: Mapping[str, Tensor], created_by: str) -> CheckpointRecord:
        path = self._path(version)
        temporary = path.with_name(path.name + ".tmp")
        save_backbone_checkpoint(temporary, state_dict, str(version))
        os.replace(temporary, path)  # a crash never leaves a half-written latest version
        self._latest_version = version
        return CheckpointRecord(version, path, created_by, datetime.now(UTC))

    def _require_latest(self) -> int:
        if self._latest_version is None:
            raise CheckpointError(f"no Backbone checkpoint in {self.directory}; call initialize first")
        return self._latest_version

    def _path(self, version: int) -> Path:
        return self.directory / f"backbone_v{version}.pt"


def _check_same_layout(new: Mapping[str, Tensor], current: Mapping[str, Tensor]) -> None:
    if new.keys() != current.keys():
        missing = sorted(current.keys() - new.keys())
        extra = sorted(new.keys() - current.keys())
        raise CheckpointError(f"update does not match the Backbone: missing {missing}, unexpected {extra}")
    for key, tensor in new.items():
        if tensor.shape != current[key].shape:
            raise CheckpointError(
                f"update has shape {tuple(tensor.shape)} for {key!r}, "
                f"expected {tuple(current[key].shape)}"
            )
