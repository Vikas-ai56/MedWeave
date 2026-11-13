"""Training augmentation on institution batches ([0, 1] inputs)."""

import torch

from institution_data.augment import Augment, augmented_batches, dihedral


def batches(count=3):
    generator = torch.Generator().manual_seed(0)
    return [(torch.rand(4, 3, 8, 8, generator=generator), torch.arange(4)) for _ in range(count)]


def test_augmented_batches_are_reproducible_from_the_seed_and_keep_labels():
    data = batches()
    first = list(augmented_batches(data, Augment(0.2), seed=7))
    second = list(augmented_batches(data, Augment(0.2), seed=7))
    other = list(augmented_batches(data, Augment(0.2), seed=8))

    assert all(torch.equal(a[0], b[0]) for a, b in zip(first, second))
    assert any(not torch.equal(a[0], b[0]) for a, b in zip(first, other))
    assert all(torch.equal(out[1], original[1]) for out, original in zip(first, data))


def test_jitter_keeps_images_in_the_unit_range_and_orientation_only_moves_pixels():
    images, _ = batches(1)[0]
    jittered = Augment(0.5)(images, torch.Generator().manual_seed(1))
    assert jittered.shape == images.shape and 0.0 <= jittered.min() and jittered.max() <= 1.0

    turned = Augment(0.0)(images, torch.Generator().manual_seed(1))
    for original, out in zip(images, turned):
        candidates = [dihedral(original[None], k, m)[0] for m in (False, True) for k in range(4)]
        assert any(torch.equal(out, candidate) for candidate in candidates)
