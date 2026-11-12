"""Activation privacy layer for MedWeave (Member 3): Gaussian noise on Head activations and its statistics.

Public interface (everything else is internal):
- `GaussianActivationNoise(noise_std, seed)`: the `perturb` hook passed to `run_owner_turn` /
  `train_batches`; `.summary()` gives the mean `PerturbationStats` of the batches it perturbed.
- `add_gaussian_noise(activations, noise_std, generator)`: the bare transformation.
- `derive_seed(base_seed, *parts)`: one reproducible noise stream per (institution, round).
- `perturbation_stats`, `mean_stats`, `activation_heatmaps`: numbers and small arrays for experiment
  tracking and the dashboard.

Independent of Flower, the coordinator and the model classes. Not differential privacy, and no
protection for gradients or labels: docs/features/data-and-privacy.md.
"""

from privacy.noise import GaussianActivationNoise, add_gaussian_noise, check_noise_std, derive_seed
from privacy.statistics import ActivationHeatmaps, PerturbationStats, activation_heatmaps, mean_stats, perturbation_stats

__all__ = [
    "ActivationHeatmaps",
    "GaussianActivationNoise",
    "PerturbationStats",
    "activation_heatmaps",
    "add_gaussian_noise",
    "check_noise_std",
    "derive_seed",
    "mean_stats",
    "perturbation_stats",
]
