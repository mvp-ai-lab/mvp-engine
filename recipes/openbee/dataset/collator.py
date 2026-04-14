"""Batch collation for the OpenBee recipe."""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

from .types import ModelInputs


class OpenbeeCollator:
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

    def __call__(self, batch: list[ModelInputs]) -> ModelInputs | None:
        """Pad token tensors and concatenate optional vision tensors.

        Args:
            batch: List of processed samples emitted by ``process_sample``.

        Returns:
            A batched tensor dictionary ready for model forward passes.
        """
        batch = [sample for sample in batch if int(sample["input_ids"].numel()) > 0]
        if not batch:
            return None

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
                raise ValueError("Packed and unpacked samples cannot be mixed in the same openbee batch.")
            model_inputs["pack_segment_ids"] = pad_sequence(
                [sample["pack_segment_ids"] for sample in batch],
                batch_first=True,
                padding_value=0,
            )

        pixel_values = [sample["pixel_values"] for sample in batch if sample.get("pixel_values") is not None]
        if pixel_values:
            model_inputs["pixel_values"] = torch.cat(pixel_values, dim=0)

        image_grid_thw = [sample["image_grid_thw"] for sample in batch if sample.get("image_grid_thw") is not None]
        if image_grid_thw:
            model_inputs["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

        return model_inputs
