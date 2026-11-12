"""Additive Gaussian noise on Head activations, applied before they leave the data-owning institution.

A' = A + noise_std * N(0, 1), elementwise. `noise_std` is an absolute standard deviation in activation
units, not a differential-privacy epsilon. This is a configurable perturbation with no formal privacy
guarantee, and it does nothing about the gradient channel (labels are recoverable from returned
gradients; see docs/features/data-and-privacy.md).

Contract: the result has A's shape, dtype and device; A is never modified; gradients flow through the
addition unchanged (the engine applies the returned gradient straight-through, which is exact for
additive noise); noise_std = 0 returns an exact copy; noise_std must be finite and >= 0 and A finite.
Noise is drawn on the CPU from a seeded generator and moved to A's device, so a run is reproducible.
`GaussianActivationNoise` is the engine's `perturb` hook (`Callable[[Tensor], Tensor]`) and keeps the
`PerturbationStats` of every batch it perturbed. `derive_seed` gives each (institution, round) its own
reproducible stream from one base seed.
"""

import hashlib
import math

import torch
from torch import Tensor

from privacy.statistics import PerturbationStats, mean_stats, perturbation_stats


def check_noise_std(noise_std: float) -> float:
    value = float(noise_std)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"noise_std must be finite and >= 0, got {noise_std!r}")
    return value


def add_gaussian_noise(activations: Tensor, noise_std: float, generator: torch.Generator | None = None) -> Tensor:
    noise_std = check_noise_std(noise_std)
    if not isinstance(activations, Tensor) or not activations.is_floating_point():
        raise TypeError("activations must be a floating-point torch.Tensor")
    if not bool(torch.isfinite(activations.detach()).all()):
        raise ValueError("activations contain NaN or infinite values")
    if noise_std == 0:
        return activations.clone()
    noise = torch.randn(activations.shape, generator=generator, dtype=activations.dtype)
    return activations + noise_std * noise.to(activations.device)


def derive_seed(base_seed: int, *parts: object) -> int:
    key = "/".join(str(part) for part in (base_seed, *parts)).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 1


class GaussianActivationNoise:
    def __init__(self, noise_std: float, seed: int) -> None:
        self.noise_std = check_noise_std(noise_std)
        self.seed = seed
        self._generator = torch.Generator().manual_seed(seed)
        self.batch_stats: list[PerturbationStats] = []

    def __call__(self, activations: Tensor) -> Tensor:
        noisy = add_gaussian_noise(activations, self.noise_std, self._generator)
        self.batch_stats.append(perturbation_stats(activations, noisy))
        return noisy

    def summary(self) -> PerturbationStats:
        return mean_stats(self.batch_stats)
