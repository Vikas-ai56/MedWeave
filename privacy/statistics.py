"""How much the noise changed an activation tensor: norms, moments, relative change, and heatmaps.

Relative change (%) = ||A' - A||_2 / (||A||_2 + 1e-12) * 100, so an all-zero A still gives a finite
value. These numbers describe how much the representation changed, for experiment tracking and the
dashboard; they say nothing about whether privacy is achieved. Everything is computed on detached
copies under no_grad and returned as Python floats or small NumPy arrays (one channel of one sample),
never as whole activation tensors.
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import Tensor

STABILITY_DELTA = 1e-12


@dataclass(frozen=True)
class PerturbationStats:
    original_norm: float
    noisy_norm: float
    noise_norm: float
    original_mean: float
    original_std: float
    noisy_mean: float
    noisy_std: float
    relative_change_pct: float


@dataclass(frozen=True)
class ActivationHeatmaps:
    original: np.ndarray
    noisy: np.ndarray
    difference: np.ndarray


def perturbation_stats(original: Tensor, noisy: Tensor) -> PerturbationStats:
    if original.shape != noisy.shape:
        raise ValueError(f"original {tuple(original.shape)} and noisy {tuple(noisy.shape)} shapes differ")
    with torch.no_grad():
        original, noisy = original.detach().double(), noisy.detach().double()
        original_norm = float(torch.linalg.vector_norm(original))
        noise_norm = float(torch.linalg.vector_norm(noisy - original))
        return PerturbationStats(
            original_norm=original_norm,
            noisy_norm=float(torch.linalg.vector_norm(noisy)),
            noise_norm=noise_norm,
            original_mean=float(original.mean()),
            original_std=float(original.std(correction=0)),
            noisy_mean=float(noisy.mean()),
            noisy_std=float(noisy.std(correction=0)),
            relative_change_pct=noise_norm / (original_norm + STABILITY_DELTA) * 100.0,
        )


def mean_stats(stats: Sequence[PerturbationStats]) -> PerturbationStats:
    if not stats:
        raise ValueError("cannot average an empty list of perturbation statistics")
    return PerturbationStats(
        **{field.name: sum(getattr(item, field.name) for item in stats) / len(stats) for field in fields(PerturbationStats)}
    )


def activation_heatmaps(original: Tensor, noisy: Tensor, sample: int = 0, channel: int = 0) -> ActivationHeatmaps:
    if original.ndim != 4 or original.shape != noisy.shape:
        raise ValueError(f"expected two (B, C, H, W) tensors of one shape, got {tuple(original.shape)} and {tuple(noisy.shape)}")
    with torch.no_grad():
        clean = original[sample, channel].detach().cpu().numpy().copy()
        perturbed = noisy[sample, channel].detach().cpu().numpy().copy()
    return ActivationHeatmaps(original=clean, noisy=perturbed, difference=perturbed - clean)
