"""Shared Backbone checkpoint I/O: files for the coordinator's store, NDArrays for Flower."""

from pathlib import Path

import numpy as np
import torch

from split_learning.backbone import Backbone


def save_checkpoint(backbone: Backbone, path: str | Path, version: int) -> None:
    torch.save({"version": version, "state_dict": backbone.state_dict()}, path)


def load_checkpoint(backbone: Backbone, path: str | Path) -> int:
    """Load weights into `backbone` in place and return the checkpoint version."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(checkpoint["state_dict"])
    return int(checkpoint["version"])


def get_weights(backbone: Backbone) -> list[np.ndarray]:
    """Backbone weights as NumPy arrays in state_dict order."""
    return [tensor.detach().cpu().numpy().copy() for tensor in backbone.state_dict().values()]


def set_weights(backbone: Backbone, weights: list[np.ndarray]) -> None:
    keys = list(backbone.state_dict().keys())
    if len(weights) != len(keys):
        raise ValueError(f"expected {len(keys)} weight arrays, got {len(weights)}")
    state_dict = {key: torch.from_numpy(np.asarray(array)) for key, array in zip(keys, weights)}
    backbone.load_state_dict(state_dict)
