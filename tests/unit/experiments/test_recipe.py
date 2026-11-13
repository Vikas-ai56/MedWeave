"""Augmentation, test-time views and the learning-rate schedule."""

import pytest
import torch

from experiments.pathmnist.augment import Augment, colour_jitter, dihedral, dihedral_views, random_dihedral
from experiments.pathmnist.recipe import LRSchedule, StepClock, adamw, set_lr


def images(count: int = 6) -> torch.Tensor:
    return torch.rand(count, 3, 8, 8, generator=torch.Generator().manual_seed(0)) * 2 - 1


def test_dihedral_views_are_eight_distinct_label_free_transforms() -> None:
    x = images(1)
    views = dihedral_views(x)

    assert len(views) == 8 and torch.equal(views[0], x)
    assert len({tuple(view.flatten().tolist()) for view in views}) == 8
    assert all(torch.allclose(view.sum(), x.sum()) for view in views)


def test_random_dihedral_maps_each_image_to_one_of_its_orientations() -> None:
    x = images()
    out = random_dihedral(x, torch.Generator().manual_seed(1))

    for original, transformed in zip(x, out):
        candidates = [dihedral(original[None], k, m)[0] for m in (False, True) for k in range(4)]
        assert any(torch.equal(transformed, candidate) for candidate in candidates)


def test_colour_jitter_stays_in_range_and_zero_strength_is_identity() -> None:
    x = images()
    jittered = colour_jitter(x, 0.4, torch.Generator().manual_seed(2))

    assert jittered.shape == x.shape and jittered.dtype == x.dtype
    assert jittered.min() >= -1.0 and jittered.max() <= 1.0
    assert not torch.allclose(jittered, x)
    assert torch.allclose(colour_jitter(x, 0.0), x, atol=1e-6)


def test_augment_is_reproducible_with_a_seeded_generator() -> None:
    x = images()
    first = Augment(0.2)(x, torch.Generator().manual_seed(3))
    second = Augment(0.2)(x, torch.Generator().manual_seed(3))

    assert torch.equal(first, second) and first.shape == x.shape


def test_cosine_schedule_warms_up_then_decays_to_zero() -> None:
    schedule = LRSchedule("cosine", total_steps=100, warmup=0.1)
    factors = [schedule.factor(step) for step in range(100)]

    assert factors[0] == pytest.approx(0.1) and factors[9] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(factors[9:], factors[10:]))
    assert factors[-1] < 0.01
    assert all(LRSchedule("constant", 10).factor(step) == 1.0 for step in range(10))
    with pytest.raises(ValueError, match="schedule"):
        LRSchedule("linear", 10)


def test_clock_sets_every_optimizer_from_its_own_base_lr() -> None:
    fast = adamw(1e-3)([torch.nn.Parameter(torch.zeros(1))])
    slow = adamw(1e-4, weight_decay=0.1)([torch.nn.Parameter(torch.zeros(1))])
    clock = StepClock(LRSchedule("cosine", total_steps=4, warmup=0.25))
    seen = []

    for _ in clock.drive(range(3), lambda: (fast, slow)):
        seen.append((fast.param_groups[0]["lr"], slow.param_groups[0]["lr"]))

    assert clock.step == 3
    assert seen[0] == pytest.approx((1e-3, 1e-4))
    assert all(f / s == pytest.approx(10.0) for f, s in seen)
    set_lr(slow, 0.5)
    assert slow.param_groups[0]["lr"] == pytest.approx(5e-5)
    assert slow.param_groups[0]["weight_decay"] == 0.1
