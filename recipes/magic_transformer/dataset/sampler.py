"""Sampler helpers for the Magic Transformer recipe."""

import math
from collections.abc import Iterator, Sized

import torch
from torch.utils.data import Sampler


class InfiniteDistributedSampler(Sampler[int]):
    """Distributed sampler that keeps iteration-based training loaders alive."""

    def __init__(
        self,
        dataset: Sized,
        num_replicas: int,
        rank: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset_size = len(dataset)
        if self.dataset_size <= 0:
            raise ValueError("InfiniteDistributedSampler requires a non-empty dataset.")

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        if self.drop_last and self.dataset_size % self.num_replicas != 0:
            self.num_samples = math.ceil((self.dataset_size - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(self.dataset_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        cycle = 0
        while True:
            if self.shuffle:
                generator = torch.Generator()
                generator.manual_seed(self.seed + cycle)
                indices = torch.randperm(self.dataset_size, generator=generator).tolist()
            else:
                indices = list(range(self.dataset_size))

            if self.drop_last:
                indices = indices[: self.total_size]
            else:
                padding_size = self.total_size - len(indices)
                if padding_size > 0:
                    repeat = math.ceil(padding_size / len(indices))
                    indices += (indices * repeat)[:padding_size]

            yield from indices[self.rank : self.total_size : self.num_replicas]
            cycle += 1

    def __len__(self) -> int:
        return self.num_samples
