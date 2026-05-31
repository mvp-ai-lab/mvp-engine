"""Batch collation for the video MLLM recipe."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .types import ModelInputs


class VideoMLLMCollator:
    """Pad token tensors and concatenate per-sample video tensors.

    The chunked token-loss patch makes the model return unreduced per-token CE,
    so each batch must carry ``total_tokens`` and ``effective_tokens`` ints for
    the engine's token-normalized loss accounting.
    """

    def __init__(self, pad_token_id: int, *, ignore_index: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: list[ModelInputs]) -> ModelInputs:
        """Pad ``input_ids``/``attention_mask``/``labels`` and concat video tensors."""
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

        pixel_values_videos = [
            sample["pixel_values_videos"] for sample in batch if sample.get("pixel_values_videos") is not None
        ]
        if pixel_values_videos:
            model_inputs["pixel_values_videos"] = torch.cat(pixel_values_videos, dim=0)

        video_grid_thw = [sample["video_grid_thw"] for sample in batch if sample.get("video_grid_thw") is not None]
        if video_grid_thw:
            model_inputs["video_grid_thw"] = torch.cat(video_grid_thw, dim=0)

        # Token counts required by the engine's per-token loss normalization.
        shifted_labels = F.pad(model_inputs["labels"], (0, 1), value=self.ignore_index)[..., 1:]
        model_inputs["total_tokens"] = int(model_inputs["attention_mask"].sum().item())
        model_inputs["effective_tokens"] = int(shifted_labels.ne(self.ignore_index).sum().item())

        return model_inputs
