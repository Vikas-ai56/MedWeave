"""Checkpoint I/O.

The Backbone checkpoint is canonical and owned by the coordinator; Head and
Tail checkpoints stay with their institution. Files are plain `torch.save`
dictionaries and are always read with `weights_only=True`.
"""

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from split_learning.errors import CheckpointError

if TYPE_CHECKING:
    from split_learning.institution import Institution

StateDict = dict[str, Tensor]


def load_module_state(module: nn.Module, state_dict: Mapping[str, Tensor], name: str) -> None:
    try:
        module.load_state_dict(state_dict)
    except RuntimeError as error:
        raise CheckpointError(f"{name} checkpoint does not match the configured architecture: {error}") from error


def load_optimizer_state(optimizer: Optimizer, state_dict: dict[str, Any], name: str) -> None:
    try:
        optimizer.load_state_dict(state_dict)
    except (ValueError, KeyError) as error:
        raise CheckpointError(f"{name} optimizer state does not match its parameters: {error}") from error


def clone_state(module: nn.Module) -> StateDict:
    return {key: tensor.detach().clone() for key, tensor in module.state_dict().items()}


def read_checkpoint(path: str | Path, required_keys: tuple[str, ...]) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError) as error:
        raise CheckpointError(f"cannot read checkpoint {path}: {error}") from error
    if not isinstance(checkpoint, dict):
        raise CheckpointError(f"checkpoint {path} is not a dictionary")
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise CheckpointError(f"checkpoint {path} is missing {missing}")
    return checkpoint


def save_backbone_checkpoint(path: str | Path, state_dict: Mapping[str, Tensor], version: str) -> None:
    torch.save({"version": version, "state_dict": dict(state_dict)}, path)


def read_backbone_checkpoint(path: str | Path) -> tuple[StateDict, str]:
    """Return `(state_dict, version)`; load it with `ComputePeer.load_backbone`."""
    checkpoint = read_checkpoint(path, ("version", "state_dict"))
    return checkpoint["state_dict"], str(checkpoint["version"])


def save_institution_checkpoint(path: str | Path, institution: "Institution") -> None:
    torch.save(
        {
            "institution_id": institution.institution_id,
            "head": institution.head.state_dict(),
            "tail": institution.tail.state_dict(),
            "head_optimizer": institution.head_optimizer.state_dict(),
            "tail_optimizer": institution.tail_optimizer.state_dict(),
        },
        path,
    )


def load_institution_checkpoint(path: str | Path, institution: "Institution") -> None:
    checkpoint = read_checkpoint(
        path, ("institution_id", "head", "tail", "head_optimizer", "tail_optimizer")
    )
    if checkpoint["institution_id"] != institution.institution_id:
        raise CheckpointError(
            f"checkpoint belongs to {checkpoint['institution_id']!r}, "
            f"not {institution.institution_id!r}"
        )
    load_module_state(institution.head, checkpoint["head"], "Head")
    load_module_state(institution.tail, checkpoint["tail"], "Tail")
    load_optimizer_state(institution.head_optimizer, checkpoint["head_optimizer"], "Head")
    load_optimizer_state(institution.tail_optimizer, checkpoint["tail_optimizer"], "Tail")


def get_weights(module: nn.Module) -> list[np.ndarray]:
    """Module weights as NumPy arrays in state_dict order (Flower's NDArrays format)."""
    return [tensor.detach().cpu().numpy().copy() for tensor in module.state_dict().values()]


def weights_to_state(module: nn.Module, weights: list[np.ndarray]) -> StateDict:
    """Pair NumPy arrays from `get_weights` back up with `module`'s state_dict keys."""
    keys = list(module.state_dict().keys())
    if len(weights) != len(keys):
        raise CheckpointError(f"expected {len(keys)} weight arrays, got {len(weights)}")
    return {key: torch.from_numpy(np.asarray(array)) for key, array in zip(keys, weights)}
