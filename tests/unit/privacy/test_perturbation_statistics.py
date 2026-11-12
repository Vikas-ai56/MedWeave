"""Perturbation statistics and heatmaps: correct values, finite at zero, small outputs only."""

import math

import pytest
import torch

from privacy import PerturbationStats, activation_heatmaps, mean_stats, perturbation_stats


def test_relative_change_and_norms_are_exact_on_a_known_tensor():
    original = torch.tensor([1.0, 1.0, 1.0, 1.0])
    noisy = original + torch.tensor([3.0, 4.0, 0.0, 0.0])

    stats = perturbation_stats(original, noisy)

    assert stats.original_norm == pytest.approx(2.0)
    assert stats.noise_norm == pytest.approx(5.0)
    assert stats.relative_change_pct == pytest.approx(250.0)
    assert stats.original_mean == pytest.approx(1.0) and stats.original_std == pytest.approx(0.0)
    assert stats.noisy_mean == pytest.approx(2.75)


def test_zero_activation_gives_finite_relative_change():
    zeros = torch.zeros(3, 4)
    assert perturbation_stats(zeros, zeros).relative_change_pct == 0.0
    assert math.isfinite(perturbation_stats(zeros, zeros + 0.1).relative_change_pct)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="shapes differ"):
        perturbation_stats(torch.zeros(2), torch.zeros(3))


def test_mean_stats_averages_every_field_and_rejects_empty():
    low = PerturbationStats(1, 1, 1, 1, 1, 1, 1, 10)
    high = PerturbationStats(3, 3, 3, 3, 3, 3, 3, 30)
    assert mean_stats([low, high]) == PerturbationStats(2, 2, 2, 2, 2, 2, 2, 20)
    with pytest.raises(ValueError, match="empty"):
        mean_stats([])


def test_heatmaps_cover_one_channel_of_one_sample():
    original = torch.rand(2, 3, 5, 6)
    noisy = original + 0.25

    maps = activation_heatmaps(original, noisy, sample=1, channel=2)

    assert maps.original.shape == maps.noisy.shape == maps.difference.shape == (5, 6)
    assert maps.difference == pytest.approx(0.25)
    assert maps.original == pytest.approx(original[1, 2].numpy())
