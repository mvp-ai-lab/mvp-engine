"""Batch collation for the video MLLM recipe."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .types import ModelInputs


def _collect_optional_tensors(batch: list[ModelInputs], key: str) -> list[torch.Tensor]:
    """Return a field from every sample, or fail if the batch is only partially populated."""
    values = [sample[key] for sample in batch if sample.get(key) is not None]
    if values and len(values) != len(batch):
        raise ValueError(f"video batch has `{key}` for only part of the batch.")
    return values


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

        pixel_values_videos = _collect_optional_tensors(batch, "pixel_values_videos")
        if pixel_values_videos:
            model_inputs["pixel_values_videos"] = torch.cat(pixel_values_videos, dim=0)

        video_grid_thw = _collect_optional_tensors(batch, "video_grid_thw")
        if video_grid_thw:
            model_inputs["video_grid_thw"] = torch.cat(video_grid_thw, dim=0)

        token_positions = _collect_optional_tensors(batch, "video_token_positions")
        if token_positions:
            model_inputs["video_token_positions"] = torch.cat(token_positions, dim=0)

        token_counts = _collect_optional_tensors(batch, "visual_token_count")
        if token_counts:
            model_inputs["video_token_counts"] = torch.stack([count.reshape(()) for count in token_counts]).to(
                dtype=torch.long
            )

        frame_grid_thw = _collect_optional_tensors(batch, "video_frame_grid_thw")
        if frame_grid_thw:
            model_inputs["video_frame_grid_thw"] = torch.cat(frame_grid_thw, dim=0)
            model_inputs["video_frame_counts"] = torch.tensor(
                [int(sample["video_frame_grid_thw"].shape[0]) for sample in batch],
                dtype=torch.long,
            )

        merge_sizes = _collect_optional_tensors(batch, "video_merge_sizes")
        if merge_sizes:
            model_inputs["video_merge_sizes"] = torch.cat(merge_sizes, dim=0)

        if pixel_values_videos:
            for key in ("video_grid_thw", "video_token_positions", "video_token_counts"):
                if key not in model_inputs:
                    raise ValueError(f"video batch is missing required `{key}` layout metadata.")
            visual_token_count = int(model_inputs["pixel_values_videos"].shape[0])
            if int(model_inputs["video_token_positions"].shape[0]) != visual_token_count:
                raise ValueError("video_token_positions length must match pixel_values_videos rows.")
            if int(model_inputs["video_token_counts"].sum().item()) != visual_token_count:
                raise ValueError("video_token_counts must sum to pixel_values_videos rows.")
            if int(model_inputs["video_grid_thw"].prod(dim=-1).sum().item()) != visual_token_count:
                raise ValueError("video_grid_thw must imply the concatenated visual token count.")

        # Token counts required by the engine's per-token loss normalization.
        shifted_labels = F.pad(model_inputs["labels"], (0, 1), value=self.ignore_index)[..., 1:]
        model_inputs["total_tokens"] = int(model_inputs["attention_mask"].sum().item())
        model_inputs["effective_tokens"] = int(shifted_labels.ne(self.ignore_index).sum().item())

        return model_inputs
