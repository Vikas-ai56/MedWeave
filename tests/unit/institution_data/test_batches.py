"""Model-ready tensors and the stateless, deterministic per-turn training stream."""

import numpy as np
import pytest
import torch

from institution_data.batches import ImageLabelDataset, evaluation_loader, to_model_inputs, training_loader
from institution_data.pathmnist import Split


def dataset(size: int = 30) -> ImageLabelDataset:
    generator = np.random.default_rng(0)
    images = generator.integers(0, 256, (size, 28, 28, 3), dtype=np.uint8)
    return ImageLabelDataset(Split(images, np.arange(size, dtype=np.int64) % 9))


def collect(loader) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(images.clone(), labels.clone()) for images, labels in loader]


def test_images_become_float_nchw_in_unit_range():
    images = np.zeros((2, 28, 28, 3), dtype=np.uint8)
    images[1, 5, 7] = (255, 0, 51)

    tensor = to_model_inputs(images)

    assert tensor.shape == (2, 3, 28, 28) and tensor.dtype == torch.float32
    assert tensor[1, :, 5, 7].tolist() == pytest.approx([1.0, 0.0, 0.2])
    assert tensor.min() >= 0 and tensor.max() <= 1


def test_training_batches_match_the_engine_contract():
    for images, labels in training_loader(dataset(), batch_size=8, seed=0, round_number=1, num_batches=3):
        assert images.shape == (8, 3, 28, 28) and images.dtype == torch.float32
        assert labels.shape == (8,) and labels.dtype == torch.int64
        assert 0 <= int(labels.min()) and int(labels.max()) <= 8


def test_consecutive_rounds_continue_one_stream_without_shared_state():
    data = dataset()
    two_rounds = collect(training_loader(data, 4, 0, round_number=1, num_batches=6))
    round_one = collect(training_loader(data, 4, 0, round_number=1, num_batches=3))
    round_two = collect(training_loader(data, 4, 0, round_number=2, num_batches=3))

    for (images, labels), (expected_images, expected_labels) in zip(round_one + round_two, two_rounds):
        assert torch.equal(images, expected_images) and torch.equal(labels, expected_labels)


def test_same_settings_reproduce_and_another_seed_reorders():
    data = dataset()
    first = collect(training_loader(data, 5, 1, 1, 2))
    again = collect(training_loader(data, 5, 1, 1, 2))
    other = collect(training_loader(data, 5, 2, 1, 2))

    assert all(torch.equal(a[0], b[0]) for a, b in zip(first, again))
    assert any(not torch.equal(a[0], b[0]) for a, b in zip(first, other))


def test_one_epoch_never_repeats_a_sample():
    batches = collect(training_loader(dataset(30), 3, 0, 1, num_batches=10))
    images = torch.cat([images for images, _ in batches])
    assert len({tuple(image.flatten()[:20].tolist()) for image in images}) == 30


def test_batch_larger_than_the_partition_is_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        training_loader(dataset(10), batch_size=11, seed=0, round_number=1, num_batches=1)


def test_evaluation_keeps_order_and_every_sample():
    data = dataset(10)
    batches = collect(evaluation_loader(data, batch_size=4))

    assert [len(labels) for _, labels in batches] == [4, 4, 2]
    assert torch.cat([labels for _, labels in batches]).tolist() == (np.arange(10) % 9).tolist()
