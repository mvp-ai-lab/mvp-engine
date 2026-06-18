"""Batch collation for standard MLLM packed samples."""

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from .media import MLLMMediaHandler


class MLLMBatchCollator:
    """Pad packed token fields and merge optional multimodal fields into a batch.

    Attributes:
        pad_token_id: Padding value for token ids.
        media_handler: Media handler used for modality-specific batch collation.
        ignore_index: Padding value for labels.
    """

    def __init__(
        self,
        pad_token_id: int,
        media_handler: MLLMMediaHandler,
        *,
        ignore_index: int = -100,
    ) -> None:
        """Store padding values used during batch collation.

        Args:
            pad_token_id: Padding value for ``input_ids``.
            media_handler: Media handler used to collate model-specific media fields.
            ignore_index: Padding value for ``labels``.
        """
        self.pad_token_id = pad_token_id
        self.media_handler = media_handler
        self.ignore_index = ignore_index

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Pad token tensors, collate media tensors, and add token-count metrics.

        Args:
            batch: Packed model-input samples yielded by the dataset pipeline.

        Returns:
            A batched model-input dictionary.

        Raises:
            ValueError: If required packed-sample metadata is missing.
        """
        if not all("pack_segment_ids" in sample for sample in batch):
            raise ValueError("Packed MLLM samples must include pack_segment_ids.")
        if not all("source_sample_num" in sample for sample in batch):
            raise ValueError("Packed MLLM samples must include source_sample_num.")

        model_inputs: dict[str, Any] = {
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
            "pack_segment_ids": pad_sequence(
                [sample["pack_segment_ids"] for sample in batch],
                batch_first=True,
                padding_value=0,
            ),
            "source_sample_num": torch.tensor(
                [int(sample["source_sample_num"]) for sample in batch],
                dtype=torch.long,
            ),
        }
        model_inputs.update(self.media_handler.collate(batch))

        model_inputs["num_input_tokens"] = model_inputs["attention_mask"].sum(dim=-1)
        shifted_labels = torch.nn.functional.pad(model_inputs["labels"], (0, 1), value=self.ignore_index)[..., 1:]
        model_inputs["num_loss_tokens"] = shifted_labels.ne(self.ignore_index).sum(dim=-1)
        model_inputs["num_source_samples"] = model_inputs["source_sample_num"].clone()
        model_inputs["total_tokens"] = int(model_inputs["num_input_tokens"].sum().item())
        model_inputs["effective_tokens"] = int(model_inputs["num_loss_tokens"].sum().item())
        return model_inputs
