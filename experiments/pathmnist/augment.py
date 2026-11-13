"""Augmentation for the experiment harness, whose loader produces inputs in [-1, 1].

The implementation lives in `institution_data.augment` (shared with the Flower system, which uses
[0, 1] inputs); this module keeps the harness's original names and value range.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from institution_data import augment as shared
from institution_data.augment import dihedral, dihedral_views, random_dihedral  # noqa: F401  (re-exported)

HARNESS_RANGE = (-1.0, 1.0)


def colour_jitter(images: Tensor, strength: float, generator: torch.Generator | None = None) -> Tensor:
    return shared.colour_jitter(images, strength, generator, HARNESS_RANGE)


@dataclass(frozen=True)
class Augment(shared.Augment):
    """Random orientation, then colour jitter, on [-1, 1] inputs."""

    value_range: shared.ValueRange = HARNESS_RANGE
