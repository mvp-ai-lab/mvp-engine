"""Recipe-local fake dataset for autoregressive language-model smoke tests."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ..configs.schema import MagicTransformerConfig


class FakeAutoregressiveDataset(Dataset[dict[str, torch.Tensor]]):
    """Generate deterministic token sequences for next-token prediction."""

    def __init__(
        self,
        *,
        size: int,
        seq_len: int,
        vocab_size: int,
        seed: int,
    ) -> None:
        self.size = int(size)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(index))

        tokens = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.seq_len + 1,),
            generator=generator,
            dtype=torch.long,
        )
        return {
            "input_ids": tokens[:-1].clone(),
            "labels": tokens[1:].clone(),
        }


def build_dataset(config: MagicTransformerConfig, workflow: str = "train") -> FakeAutoregressiveDataset:
    """Build the fake dataset split requested by the engine."""
    is_train = workflow == "train"
    size = config.data.fake_train_size if is_train else config.data.fake_eval_size
    seed = config.seed if is_train else config.seed + 10_000

    return FakeAutoregressiveDataset(
        size=size,
        seq_len=config.data.seq_len,
        vocab_size=config.data.vocab_size,
        seed=seed,
    )
