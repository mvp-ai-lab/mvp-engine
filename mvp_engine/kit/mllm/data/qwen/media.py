"""Qwen VL media handlers for MLLM data pipelines."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from mvp_engine.utils.log import simple_info

from ..media import (
    MLLMMediaHandler,
    MLLMMediaTypeHandler,
    RenderedMedia,
    empty_model_sample,
)
from ..types import MLLMMediaSlot

VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"


def read_image(image: str | bytes | dict[str, Any] | Image.Image) -> Image.Image:
    """Normalize one image input into a decoded RGB PIL image.

    Args:
        image: Image path, encoded image bytes, image record with ``bytes`` or ``path``,
            or an existing PIL image.

    Returns:
        A copied RGB PIL image.

    Raises:
        FileNotFoundError: If a path input does not exist.
        ValueError: If the image value or image record is invalid.
        OSError: If PIL cannot decode the image bytes or file.
    """
    if isinstance(image, dict):
        image_bytes = image.get("bytes")
        if isinstance(image_bytes, (bytes, bytearray, memoryview)):
            return read_image(bytes(image_bytes))

        image_path = image.get("path")
        if isinstance(image_path, str) and image_path:
            return read_image(image_path)

        raise ValueError("contains an invalid image record.")

    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()

    if isinstance(image, bytes):
        with Image.open(io.BytesIO(image)) as decoded:
            return decoded.convert("RGB")

    if not isinstance(image, str):
        raise ValueError(f"contains an invalid image value: {type(image).__name__}.")
    if not image:
        raise ValueError(f"contains an invalid image path: {image!r}")

    resolved = Path(image).expanduser()
    resolved = resolved.resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"references missing image: {resolved}")
    with Image.open(resolved) as opened:
        return opened.convert("RGB").copy()


class QwenVLMediaHandler(MLLMMediaHandler):
    """Qwen VL media registry with image support.

    Attributes:
        processor: Qwen VL processor.
        handlers: Registry of media type handlers.
    """

    def __init__(self, processor: Any, handlers: dict[str, MLLMMediaTypeHandler] | None = None) -> None:
        """Initialize Qwen VL image media handling.

        Args:
            processor: Qwen VL processor.
            handlers: Optional media handler registry. When omitted, image support is registered.
        """
        super().__init__(processor=processor, handlers=handlers or {"image": QwenImageHandler()})


class QwenImageHandler(MLLMMediaTypeHandler):
    """Render, load, merge, and collate Qwen-style image inputs.

    Attributes:
        media_type: Registered media type name.
        DUMMY_IMAGE_PIXELS: Pixel budget for the synthetic image used by text-only batch guards.
        OUTPUT_TENSOR_KEYS: Qwen image tensor keys emitted by this handler.
    """

    media_type = "image"
    DUMMY_IMAGE_PIXELS = 32 * 32
    OUTPUT_TENSOR_KEYS = ("pixel_values", "image_grid_thw")

    def render(
        self,
        slot: MLLMMediaSlot,
        *,
        processor: Any,
        tokenizer: Any,
    ) -> RenderedMedia:
        """Render one Qwen image slot into placeholder text.

        Args:
            slot: Image media slot with size metadata in ``[height, width]`` order.
            processor: Qwen VL processor.
            tokenizer: Unused tokenizer argument accepted for the base handler contract.

        Returns:
            Rendered image placeholder text with Qwen vision wrapper tokens.
        """
        del tokenizer
        media_token = self.default_token(processor)
        adjusted_height, adjusted_width = self._adjusted_image_size(slot, processor=processor)
        patch_size, merge_size, _, _ = self._read_image_geometry(processor)
        token_count = (adjusted_height // patch_size) * (adjusted_width // patch_size) // (merge_size**2)
        return RenderedMedia(
            media_id=slot.media_id,
            media_type=self.media_type,
            text=f"{VISION_START_TOKEN}{media_token * token_count}{VISION_END_TOKEN}",
        )

    def default_token(self, processor: Any) -> str:
        """Return the processor image token.

        Args:
            processor: Qwen VL processor.

        Returns:
            Image placeholder token text.

        Raises:
            ValueError: If the processor does not expose a valid image token.
        """
        media_token = getattr(processor, "image_token", DEFAULT_IMAGE_TOKEN)
        if not isinstance(media_token, str) or not media_token:
            raise ValueError("Processor must expose a valid image token.")
        return media_token

    def load(
        self,
        slots: list[MLLMMediaSlot],
        values: list[Any],
        *,
        processor: Any,
    ) -> dict[str, Any]:
        """Materialize resolved image references into Qwen image-processor tensors.

        Args:
            slots: Image media slots.
            values: Current raw image values read from the sample.
            processor: Qwen VL processor.

        Returns:
            ``pixel_values`` and ``image_grid_thw`` tensors, or an empty-sample
            sentinel when required image media cannot be read.
        """
        if not values:
            return {}

        try:
            adjusted_sizes = [self._adjusted_image_size(slot, processor=processor) for slot in slots]
            resized_images = [
                self._load_resized_image(image, target_size)
                for image, target_size in zip(values, adjusted_sizes, strict=True)
            ]
            image_inputs = processor.image_processor(images=resized_images, do_resize=False, return_tensors="pt")
        except FileNotFoundError as exc:
            simple_info(f"dropping sample with missing image media: {exc}", level="warning")
            return empty_model_sample()
        except (OSError, ValueError) as exc:
            simple_info(f"dropping sample with unreadable media: {exc}", level="warning")
            return empty_model_sample()

        return {
            "pixel_values": image_inputs["pixel_values"],
            "image_grid_thw": image_inputs["image_grid_thw"],
        }

    def merge_pack(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge image tensors from source samples into a finalized packed sample.

        Args:
            samples: Per-source-sample model-input dictionaries in one pack.

        Returns:
            Concatenated Qwen image tensors for the pack.
        """
        merged = {}
        for key in self.OUTPUT_TENSOR_KEYS:
            tensors = [sample[key] for sample in samples if sample.get(key) is not None]
            if tensors:
                merged[key] = torch.cat(tensors, dim=0)
        return merged

    def build_dummy_inputs(self, processor: Any) -> dict[str, torch.Tensor]:
        """Build one valid minimal Qwen image suffix for text-only batches.

        Args:
            processor: Qwen VL processor.

        Returns:
            Minimal token and image tensors accepted by Qwen VL models.
        """
        dummy_image = Image.new("RGB", (32, 32), color=0)
        fake_messages = [{"role": "user", "content": [{"type": "image", "image": dummy_image}]}]
        model_inputs = processor.apply_chat_template(
            [fake_messages],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            min_pixels=self.DUMMY_IMAGE_PIXELS,
            max_pixels=self.DUMMY_IMAGE_PIXELS,
        )
        return {
            "input_ids": model_inputs["input_ids"][0].to(dtype=torch.long),
            "attention_mask": model_inputs["attention_mask"][0].to(dtype=torch.long),
            "pixel_values": model_inputs["pixel_values"],
            "image_grid_thw": model_inputs["image_grid_thw"],
        }

    def collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge image tensors into a collated batch.

        Args:
            batch: Packed model-input dictionaries.

        Returns:
            Concatenated Qwen image tensors for the batch.
        """
        collated = {}
        for key in self.OUTPUT_TENSOR_KEYS:
            tensors = [sample[key] for sample in batch if sample.get(key) is not None]
            if tensors:
                collated[key] = torch.cat(tensors, dim=0)
        return collated

    @staticmethod
    def _read_image_geometry(processor: Any) -> tuple[int, int, int, int]:
        """Read Qwen image-processor geometry: (patch_size, merge_size, min_pixels, max_pixels)."""
        image_processor = getattr(processor, "image_processor", processor)
        patch_size = getattr(image_processor, "patch_size", None)
        merge_size = getattr(image_processor, "merge_size", None)
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("Image processor must expose a positive integer `patch_size`.")
        if not isinstance(merge_size, int) or merge_size <= 0:
            raise ValueError("Image processor must expose a positive integer `merge_size`.")

        processor_size = getattr(image_processor, "size", {})
        min_pixels = getattr(processor, "min_image_size", None)
        max_pixels = getattr(processor, "max_image_size", None)
        if hasattr(processor_size, "get"):
            min_pixels = min_pixels if min_pixels is not None else processor_size.get("shortest_edge")
            max_pixels = max_pixels if max_pixels is not None else processor_size.get("longest_edge")
        if not isinstance(min_pixels, int) or min_pixels <= 0:
            raise ValueError("Image processor must expose a positive integer min pixel budget.")
        if not isinstance(max_pixels, int) or max_pixels <= 0:
            raise ValueError("Image processor must expose a positive integer max pixel budget.")
        return patch_size, merge_size, min_pixels, max_pixels

    @staticmethod
    def _adjusted_image_size(slot: MLLMMediaSlot, *, processor: Any) -> list[int]:
        """Return the smart-resized height and width used by Qwen image tokenization."""
        raw_size = slot.metadata.get("size")
        if not isinstance(raw_size, (list, tuple)) or len(raw_size) < 2:
            raise ValueError("image media must include size metadata.")

        patch_size, merge_size, min_pixels, max_pixels = QwenImageHandler._read_image_geometry(processor)
        height, width = int(raw_size[0]), int(raw_size[1])
        adjusted_height, adjusted_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return [adjusted_height, adjusted_width]

    @staticmethod
    def _load_resized_image(image: Any, target_size: list[int]) -> torch.Tensor:
        """Load one image and resize it to its precomputed smart-resize shape."""
        import torchvision.transforms.v2.functional as tvF
        from torchvision.transforms import InterpolationMode

        height, width = int(target_size[0]), int(target_size[1])
        image_tensor = tvF.pil_to_tensor(read_image(image))
        if tuple(image_tensor.shape[-2:]) != (height, width):
            image_tensor = tvF.resize(
                image_tensor,
                [height, width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        return image_tensor
