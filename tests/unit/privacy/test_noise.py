"""Gaussian activation noise: identity at zero, shape-preserving, non-mutating, seeded, gradient-transparent."""

import math

import pytest
import torch

from privacy import GaussianActivationNoise, add_gaussian_noise, derive_seed


def activations(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.rand(4, 16, 14, 14, generator=torch.Generator().manual_seed(0), dtype=dtype)


def test_zero_noise_returns_an_equal_copy():
    original = activations()
    result = add_gaussian_noise(original, 0.0)

    assert torch.equal(result, original) and result is not original


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_positive_noise_changes_values_and_keeps_shape_dtype_device(dtype):
    original = activations(dtype)
    result = add_gaussian_noise(original, 0.5, torch.Generator().manual_seed(1))

    assert result.shape == original.shape and result.dtype == dtype and result.device == original.device
    assert not torch.equal(result, original)


def test_input_is_never_modified():
    original = activations()
    before = original.clone()
    add_gaussian_noise(original, 1.0, torch.Generator().manual_seed(1))
    assert torch.equal(original, before)


def test_noise_has_the_requested_standard_deviation():
    original = torch.zeros(200_000)
    result = add_gaussian_noise(original, 0.3, torch.Generator().manual_seed(2))
    assert float(result.std()) == pytest.approx(0.3, rel=0.02)
    assert abs(float(result.mean())) < 0.01


def test_same_seed_reproduces_and_another_seed_differs():
    original = activations()
    first = add_gaussian_noise(original, 0.5, torch.Generator().manual_seed(3))
    again = add_gaussian_noise(original, 0.5, torch.Generator().manual_seed(3))
    other = add_gaussian_noise(original, 0.5, torch.Generator().manual_seed(4))
    assert torch.equal(first, again) and not torch.equal(first, other)


@pytest.mark.parametrize("noise_std", [-0.1, math.nan, math.inf])
def test_invalid_noise_std_is_rejected(noise_std):
    with pytest.raises(ValueError, match="noise_std"):
        add_gaussian_noise(activations(), noise_std)
    with pytest.raises(ValueError, match="noise_std"):
        GaussianActivationNoise(noise_std, seed=0)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf])
def test_non_finite_activations_are_rejected(bad_value):
    tensor = activations()
    tensor[0, 0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="NaN or infinite"):
        add_gaussian_noise(tensor, 0.1)


def test_integer_activations_are_rejected():
    with pytest.raises(TypeError):
        add_gaussian_noise(torch.ones(2, 2, dtype=torch.int64), 0.1)


def test_gradients_flow_through_the_addition_unchanged():
    original = activations().requires_grad_()
    add_gaussian_noise(original, 0.5, torch.Generator().manual_seed(5)).sum().backward()
    assert torch.equal(original.grad, torch.ones_like(original))


def test_hook_is_reproducible_per_seed_and_records_each_batch():
    first, again = GaussianActivationNoise(0.2, seed=9), GaussianActivationNoise(0.2, seed=9)
    batches = [activations(), activations()]

    outputs = [first(batch) for batch in batches]
    assert all(torch.equal(out, again(batch)) for out, batch in zip(outputs, batches))
    assert len(first.batch_stats) == 2 and first.summary().relative_change_pct > 0


def test_derived_seeds_are_stable_and_distinct_per_institution_and_round():
    assert derive_seed(0, "inst-1", 1) == derive_seed(0, "inst-1", 1)
    seeds = {derive_seed(0, institution, round_number) for institution in ("inst-1", "inst-2") for round_number in (1, 2)}
    assert len(seeds) == 4 and all(0 <= seed < 2**63 for seed in seeds)
