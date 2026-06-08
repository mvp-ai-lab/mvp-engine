"""Batch collation for the minimal VLM recipe."""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

from .types import ModelInputs


class MinimalVLMCollator:
    """Pad and merge preprocessed multimodal samples."""

    def __init__(self, pad_token_id: int, *, ignore_index: int = -100) -> None:
        """Store padding values used during batch collation.

        Args:
            pad_token_id: Token id used to pad ``input_ids``.
            ignore_index: Label value used to pad masked loss positions.

        Returns:
            None.
        """
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: list[ModelInputs]) -> ModelInputs:
        """Pad token tensors and concatenate optional vision tensors.

        Args:
            batch: List of processed samples emitted by ``process_sample``.

        Returns:
            A batched tensor dictionary ready for model forward passes.
        """
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

        pixel_values = [sample["pixel_values"] for sample in batch if sample.get("pixel_values") is not None]
        if pixel_values:
            model_inputs["pixel_values"] = torch.cat(pixel_values, dim=0)

        image_grid_thw = [sample["image_grid_thw"] for sample in batch if sample.get("image_grid_thw") is not None]
        if image_grid_thw:
            model_inputs["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

        if any("mm_token_type_ids" in sample for sample in batch):
            mm_token_type_ids = [
                sample.get("mm_token_type_ids", torch.zeros_like(sample["input_ids"])) for sample in batch
            ]
            model_inputs["mm_token_type_ids"] = pad_sequence(
                mm_token_type_ids,
                batch_first=True,
                padding_value=0,
            )

        return model_inputs
