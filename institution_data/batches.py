"""Model-ready tensors and DataLoaders over one institution's images.

The dataset boundary is crossed only here: uint8 NHWC images become float32 NCHW in [0, 1] and labels
int64 `[B]` with class ids 0..8 (the split-learning input contract), with no normalisation or
augmentation.

Training data is a deterministic stream of full batches over seeded epochs. Epoch e is a permutation
drawn from (seed, e); each epoch's trailing partial batch is dropped, and a later epoch's different order
reaches those samples. A turn of `num_batches` in round r reads stream positions
[(r - 1) * num_batches, r * num_batches), so a turn needs nothing carried over from earlier turns: every
ClientApp call can rebuild its loader from the round number alone. Evaluation loaders keep the official
order and every sample.
"""

from collections.abc import Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from institution_data.pathmnist import Split


def to_model_inputs(images: np.ndarray) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div_(255.0)


class ImageLabelDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, split: Split) -> None:
        self.split = split

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image = to_model_inputs(self.split.images[index : index + 1])[0]
        return image, torch.tensor(self.split.labels[index], dtype=torch.int64)

    def __getitems__(self, indices: list[int]) -> tuple[Tensor, Tensor]:
        selection = np.asarray(indices)
        return to_model_inputs(self.split.images[selection]), torch.from_numpy(self.split.labels[selection])


def keep_batch(batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    return batch


class TurnBatchSampler(Sampler[list[int]]):
    def __init__(self, num_samples: int, batch_size: int, seed: int, first_batch: int, num_batches: int) -> None:
        if batch_size < 1 or batch_size > num_samples:
            raise ValueError(f"batch_size must be in [1, {num_samples}], got {batch_size}")
        if first_batch < 0 or num_batches < 1:
            raise ValueError(f"need first_batch >= 0 and num_batches >= 1, got {first_batch}, {num_batches}")
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed
        self.first_batch = first_batch
        self.num_batches = num_batches

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        batches_per_epoch = self.num_samples // self.batch_size
        current_epoch, order = -1, np.empty(0, dtype=np.int64)
        for position in range(self.first_batch, self.first_batch + self.num_batches):
            epoch, offset = divmod(position, batches_per_epoch)
            if epoch != current_epoch:
                current_epoch, order = epoch, np.random.default_rng([self.seed, epoch]).permutation(self.num_samples)
            yield order[offset * self.batch_size : (offset + 1) * self.batch_size].tolist()


def training_loader(
    dataset: ImageLabelDataset, batch_size: int, seed: int, round_number: int, num_batches: int
) -> DataLoader:
    if round_number < 1:
        raise ValueError(f"round_number starts at 1, got {round_number}")
    sampler = TurnBatchSampler(len(dataset), batch_size, seed, (round_number - 1) * num_batches, num_batches)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=keep_batch)


def evaluation_loader(dataset: ImageLabelDataset, batch_size: int) -> DataLoader:
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=keep_batch)
