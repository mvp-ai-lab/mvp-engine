"""Batch collation for the Qwen3 LM recipe."""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

from .types import ModelInputs


class Qwen3LMCollator:
    """Pad and merge preprocessed text samples."""

    def __init__(self, pad_token_id: int, *, ignore_index: int = -100) -> None:
        """Store padding values used during batch collation."""
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: list[ModelInputs]) -> ModelInputs:
        """Pad token tensors and optional packed-segment metadata."""
        model_inputs: ModelInputs = {
            "input_ids": pad_sequence(
                [sample["input_ids"] for sample in batch],
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [sample["attention_mask"] for sample in batch],
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                [sample["labels"] for sample in batch],
                batch_first=True,
                padding_value=self.ignore_index,
            ),
        }

        if any("pack_segment_ids" in sample for sample in batch):
            if not all("pack_segment_ids" in sample for sample in batch):
                raise ValueError("Packed and unpacked samples cannot be mixed in the same qwen3_lm batch.")
            model_inputs["pack_segment_ids"] = pad_sequence(
                [sample["pack_segment_ids"] for sample in batch],
                batch_first=True,
                padding_value=0,
            )

        if any("source_sample_num" in sample for sample in batch):
            model_inputs["source_sample_num"] = torch.tensor(
                [int(sample.get("source_sample_num", 1)) for sample in batch],
                dtype=torch.long,
            )

        return model_inputs
