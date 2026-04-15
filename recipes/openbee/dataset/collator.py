"""Batch collation for the OpenBee recipe."""

from __future__ import annotations

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

from .types import ModelInputs


class OpenbeeCollator:
    """Pad and merge preprocessed multimodal samples."""

    DUMMY_IMAGE_SIZE = (16, 16)

    def __init__(self, pad_token_id: int, processor, *, ignore_index: int = -100) -> None:
        """Store padding values used during batch collation.

        Args:
            pad_token_id: Token id used to pad ``input_ids``.
            processor: Hugging Face processor used to build batch-level dummy inputs.
            ignore_index: Label value used to pad masked loss positions.

        Returns:
            None.
        """
        self.pad_token_id = pad_token_id
        self.processor = processor
        self.ignore_index = ignore_index
        self._cached_dummy_inputs: dict[str, torch.Tensor] | None = None

    def _get_dummy_image(self) -> Image.Image:
        """Return a cached RGB dummy image for batch-level visual graph participation."""
        cached = getattr(self.processor, "_mvp_openbee_batch_dummy_image", None)
        if isinstance(cached, Image.Image):
            return cached.copy()

        dummy = Image.new("RGB", self.DUMMY_IMAGE_SIZE, color=0)
        setattr(self.processor, "_mvp_openbee_batch_dummy_image", dummy)
        return dummy.copy()

    def _get_dummy_batch_inputs(self) -> dict[str, torch.Tensor]:
        """Build and cache one fully-masked fake multimodal suffix for local-batch injection."""
        if self._cached_dummy_inputs is not None:
            return {key: value.clone() for key, value in self._cached_dummy_inputs.items()}

        fake_messages = [{"role": "user", "content": [{"type": "image", "image": self._get_dummy_image()}]}]
        model_inputs = self.processor.apply_chat_template(
            [fake_messages],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        self._cached_dummy_inputs = {
            "input_ids": model_inputs["input_ids"][0].to(dtype=torch.long),
            "pixel_values": model_inputs["pixel_values"],
            "image_grid_thw": model_inputs["image_grid_thw"],
        }
        return {key: value.clone() for key, value in self._cached_dummy_inputs.items()}

    def _inject_batch_level_dummy_image(self, batch: list[ModelInputs]) -> dict[str, torch.Tensor] | None:
        """Append a fully-masked fake visual suffix when the whole local batch is text-only."""
        if any(sample.get("pixel_values") is not None for sample in batch):
            return None

        dummy_inputs = self._get_dummy_batch_inputs()
        dummy_input_ids = dummy_inputs["input_ids"]
        first_sample = batch[0]

        first_sample["input_ids"] = torch.cat([first_sample["input_ids"], dummy_input_ids], dim=0)
        first_sample["attention_mask"] = torch.cat(
            [
                first_sample["attention_mask"],
                torch.zeros_like(dummy_input_ids, dtype=first_sample["attention_mask"].dtype),
            ],
            dim=0,
        )
        first_sample["labels"] = torch.cat(
            [first_sample["labels"], torch.full_like(dummy_input_ids, self.ignore_index)],
            dim=0,
        )

        if "pack_segment_ids" in first_sample:
            first_sample["pack_segment_ids"] = torch.cat(
                [first_sample["pack_segment_ids"], torch.zeros_like(dummy_input_ids, dtype=torch.long)],
                dim=0,
            )

        return dummy_inputs

    def __call__(self, batch: list[ModelInputs]) -> ModelInputs:
        """Pad token tensors and concatenate optional vision tensors.

        Args:
            batch: List of processed samples emitted by ``process_sample``.

        Returns:
            A batched tensor dictionary ready for model forward passes.
        """
        dummy_inputs = self._inject_batch_level_dummy_image(batch)

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
        elif dummy_inputs is not None:
            model_inputs["image_grid_thw"] = dummy_inputs["image_grid_thw"]
            model_inputs["dummy_image_grid_count"] = int(dummy_inputs["image_grid_thw"].shape[0])

        if dummy_inputs is not None:
            model_inputs["pixel_values"] = dummy_inputs["pixel_values"]

        return model_inputs
