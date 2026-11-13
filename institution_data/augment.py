"""Training augmentation for histology patches (moved from experiments/pathmnist/augment.py).

Patches have no canonical orientation, so the 8 symmetries of the square (4 rotations, each optionally
mirrored) preserve the label. Colour jitter imitates staining and scanner differences between sites.
Images are float `[B, 3, H, W]` in `value_range`: [0, 1] for `institution_data` batches (the default),
[-1, 1] for the experiment harness's own loader.

`augmented_batches` applies an `Augment` to a turn's batch stream with a generator seeded by the caller,
so a turn stays a pure function of (seed, institution, round) like the batch order itself. Labels and
evaluation data are never augmented.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

LUMA = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
ValueRange = tuple[float, float]


def dihedral(images: Tensor, rotations: int, mirror: bool) -> Tensor:
    images = torch.rot90(images, rotations, dims=(2, 3))
    return images.flip(3) if mirror else images


def dihedral_views(images: Tensor) -> list[Tensor]:
    """All 8 orientations; the first is the identity."""
    return [dihedral(images, rotations, mirror) for mirror in (False, True) for rotations in range(4)]


def random_dihedral(images: Tensor, generator: torch.Generator | None = None) -> Tensor:
    choice = torch.randint(0, 8, (len(images),), generator=generator)
    out = torch.empty_like(images)
    for index in range(8):
        selected = choice == index
        if selected.any():
            out[selected] = dihedral(images[selected], index % 4, index >= 4)
    return out


def colour_jitter(
    images: Tensor, strength: float, generator: torch.Generator | None = None, value_range: ValueRange = (0.0, 1.0)
) -> Tensor:
    """Per-image brightness, contrast and saturation, each scaled by a factor in [1 - strength, 1 + strength]."""
    low, high = value_range

    def factors() -> Tensor:
        return 1 + strength * (2 * torch.rand(len(images), 1, 1, 1, generator=generator) - 1)

    image = (images - low) / (high - low)
    image = image * factors()
    gray = (image * LUMA).sum(dim=1, keepdim=True)
    mean = gray.mean(dim=(2, 3), keepdim=True)
    image = (image - mean) * factors() + mean
    gray = (image * LUMA).sum(dim=1, keepdim=True)
    image = (image - gray) * factors() + gray
    return image.clamp(0, 1) * (high - low) + low


@dataclass(frozen=True)
class Augment:
    """Random orientation, then colour jitter of the given strength (0 disables it)."""

    jitter: float = 0.2
    value_range: ValueRange = (0.0, 1.0)

    def __call__(self, images: Tensor, generator: torch.Generator | None = None) -> Tensor:
        images = random_dihedral(images, generator)
        if self.jitter > 0:
            images = colour_jitter(images, self.jitter, generator, self.value_range)
        return images


def augmented_batches(
    batches: Iterable[tuple[Tensor, Tensor]], augment: Augment, seed: int
) -> Iterator[tuple[Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    for images, labels in batches:
        yield augment(images, generator), labels
