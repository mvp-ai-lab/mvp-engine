"""Batch collation for standard packed text-only LM samples."""

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from .types import ModelInputs


class LLMBatchCollator:
    """Pad packed token fields and add token-count metrics."""

    def __init__(self, pad_token_id: int, *, ignore_index: int = -100) -> None:
        """Store padding values used during batch collation."""
        self.pad_token_id = int(pad_token_id)
        self.ignore_index = int(ignore_index)

    def __call__(self, batch: list[dict[str, Any]]) -> ModelInputs:
        """Pad token tensors to the longest sample and compute token counts."""
        if not all("pack_segment_ids" in sample for sample in batch):
            raise ValueError("Packed LLM samples must include pack_segment_ids.")
        if not all("source_sample_num" in sample for sample in batch):
            raise ValueError("Packed LLM samples must include source_sample_num.")

        input_ids = pad_sequence(
            [sample["input_ids"] for sample in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence([sample["attention_mask"] for sample in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([sample["labels"] for sample in batch], batch_first=True, padding_value=self.ignore_index)
        pack_segment_ids = pad_sequence(
            [sample["pack_segment_ids"] for sample in batch],
            batch_first=True,
            padding_value=0,
        )
        source_sample_num = torch.tensor([int(sample["source_sample_num"]) for sample in batch], dtype=torch.long)

        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pack_segment_ids": pack_segment_ids,
            "source_sample_num": source_sample_num,
        }
        if all("position_ids" in sample for sample in batch):
            model_inputs["position_ids"] = pad_sequence(
                [sample["position_ids"] for sample in batch],
                batch_first=True,
                padding_value=0,
            )

        num_input_tokens = attention_mask.sum(dim=-1)
        shifted_labels = torch.nn.functional.pad(labels, (0, 1), value=self.ignore_index)[..., 1:]
        num_loss_tokens = shifted_labels.ne(self.ignore_index).sum(dim=-1)
        model_inputs["num_input_tokens"] = num_input_tokens
        model_inputs["num_loss_tokens"] = num_loss_tokens
        model_inputs["num_source_samples"] = source_sample_num.clone()
        model_inputs["total_tokens"] = int(num_input_tokens.sum().item())
        model_inputs["effective_tokens"] = int(num_loss_tokens.sum().item())
        return model_inputs
