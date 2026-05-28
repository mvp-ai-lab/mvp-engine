"""Model-family media processing for default MLLM data pipelines."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from mvp_engine.utils.log import simple_info

from .types import CanonicalMedia

IMAGE_TOKEN_PLACEHOLDER = "<|mvp_image_placeholder|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"


def build_empty_sample():
    """Build an empty model-input sentinel for invalid samples."""
    return {
        "input_ids": torch.empty(0, dtype=torch.long),
        "attention_mask": torch.empty(0, dtype=torch.long),
        "labels": torch.empty(0, dtype=torch.long),
    }


@dataclass(slots=True)
class MediaProcessState:
    """Mutable media-token state used while rendering one chat sample."""

    prepared_fields: dict[str, list[Any]]
    sample_fields: dict[str, list[Any]]
    token_counts: list[int]
    media_token: str
    special_token_ids: set[int]
    special_token_id_tensor: torch.Tensor
    cursor: int = 0


def read_image(
    image: str | bytes | dict[str, Any] | Image.Image,
    *,
    image_root: Path | None = None,
) -> Image.Image:
    """Normalize one image input into a decoded RGB PIL image."""
    if isinstance(image, dict):
        image_bytes = image.get("bytes")
        if isinstance(image_bytes, (bytes, bytearray, memoryview)):
            return read_image(bytes(image_bytes), image_root=image_root)

        image_path = image.get("path")
        if isinstance(image_path, str) and image_path:
            return read_image(image_path, image_root=image_root)

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
    if not resolved.is_absolute() and image_root is not None:
        resolved = image_root / resolved
    resolved = resolved.resolve()

    if not resolved.is_file():
        raise FileNotFoundError(f"references missing image: {resolved}")
    with Image.open(resolved) as opened:
        return opened.convert("RGB").copy()


class MLLMMediaKit:
    """Default Qwen-style image media implementation for MLLM data pipelines."""

    DUMMY_IMAGE_SIZE = (32, 32)
    DUMMY_IMAGE_PIXELS = 32 * 32

    def __init__(self) -> None:
        """Initialize per-kit caches used by media collation."""
        self._media_token_id_cache: dict[tuple[int, str], tuple[set[int], torch.Tensor]] = {}
        self._cached_dummy_inputs: dict[int, dict[str, torch.Tensor]] = {}

    def prepare(
        self,
        media: list[CanonicalMedia],
        *,
        processor: Any,
        tokenizer: Any,
    ) -> MediaProcessState:
        """Prepare Qwen image token counts and late-materialization metadata."""
        image_media = [item for item in media if item.type == "image"]
        images = [item.value for item in image_media]
        image_sizes = [item.size for item in image_media]
        if any(size is None for size in image_sizes):
            raise ValueError("image media must include size metadata.")

        media_token = getattr(processor, "image_token", DEFAULT_IMAGE_TOKEN)
        if not isinstance(media_token, str) or not media_token:
            raise ValueError("Processor must expose a valid image token.")
        special_token_ids, special_token_id_tensor = self._get_special_token_ids(tokenizer, media_token)

        if not image_media:
            return MediaProcessState(
                prepared_fields={
                    "images": [],
                    "adjusted_image_size": [],
                },
                sample_fields={
                    "images": [],
                    "adjusted_image_size": [],
                },
                token_counts=[],
                media_token=media_token,
                special_token_ids=special_token_ids,
                special_token_id_tensor=special_token_id_tensor,
            )

        image_processor, patch_size, merge_size, min_pixels, max_pixels = self._resolve_image_processor_config(
            processor
        )
        factor = patch_size * merge_size
        adjusted_image_sizes: list[list[int]] = []
        image_token_counts: list[int] = []
        for size in image_sizes:
            height, width = int(size[0]), int(size[1])
            resized_size = self._smart_resize_image_size(
                height,
                width,
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            if resized_size is None:
                raise ValueError(f"cannot smart-resize image size {height}x{width}.")

            resized_height, resized_width = resized_size
            adjusted_image_sizes.append([int(resized_height), int(resized_width)])
            token_count = self._estimate_image_tokens(
                image_processor,
                height=int(resized_height),
                width=int(resized_width),
                patch_size=patch_size,
                merge_size=merge_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            if token_count is None or token_count <= 0:
                raise ValueError(f"cannot estimate image token count after resize: {resized_height}x{resized_width}.")
            image_token_counts.append(int(token_count))

        return MediaProcessState(
            prepared_fields={
                "images": images,
                "adjusted_image_size": adjusted_image_sizes,
            },
            sample_fields={
                "images": [],
                "adjusted_image_size": [],
            },
            token_counts=image_token_counts,
            media_token=media_token,
            special_token_ids=special_token_ids,
            special_token_id_tensor=special_token_id_tensor,
        )

    def render_text(self, text: str, state: MediaProcessState) -> str:
        """Expand Qwen image placeholders into language-side image token spans."""
        placeholder = IMAGE_TOKEN_PLACEHOLDER if IMAGE_TOKEN_PLACEHOLDER in text else state.media_token
        parts = text.split(placeholder)
        if len(parts) == 1:
            return text

        expanded_parts = [parts[0]]
        for part in parts[1:]:
            if state.cursor >= len(state.token_counts):
                raise ValueError("image size metadata does not match rendered image placeholders.")
            token_count = state.token_counts[state.cursor]
            for key, values in state.prepared_fields.items():
                state.sample_fields[key].append(values[state.cursor])
            state.cursor += 1
            expanded_parts.append(state.media_token * token_count)
            expanded_parts.append(part)
        return "".join(expanded_parts)

    def check_truncation(self, token_ids: list[int], keep_len: int, *, state: MediaProcessState) -> None:
        """Reject truncation that would cut away any media special tokens."""
        if any(token_id in state.special_token_ids for token_id in token_ids[keep_len:]):
            raise ValueError("truncation would cut media tokens.")

    def mask_labels(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        state: MediaProcessState,
        ignore_index: int,
    ) -> torch.Tensor:
        """Mask model-family media tokens out of supervised labels."""
        labels[torch.isin(input_ids, state.special_token_id_tensor)] = ignore_index
        return labels

    def _get_special_token_ids(self, tokenizer: Any, image_token: str) -> tuple[set[int], torch.Tensor]:
        """Resolve tokenizer ids that must always be masked from labels."""
        cache_key = (id(tokenizer), image_token)
        cached = self._media_token_id_cache.get(cache_key)
        if cached is not None:
            return cached

        media_token_ids: set[int] = set()
        for token in (VISION_START_TOKEN, VISION_END_TOKEN, image_token):
            media_token_ids.update(tokenizer(token, add_special_tokens=False)["input_ids"])
        media_token_id_tensor = torch.tensor(sorted(media_token_ids), dtype=torch.long)
        self._media_token_id_cache[cache_key] = (media_token_ids, media_token_id_tensor)
        return self._media_token_id_cache[cache_key]

    def materialize(
        self,
        sample: dict[str, Any] | list[dict[str, Any]],
        *,
        processor: Any,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Materialize resolved media references into Qwen image-processor tensors."""
        if isinstance(sample, list):
            return [self.materialize(s, processor=processor) for s in sample]

        try:
            images = sample.get("images", [])
            adjusted_sizes = sample.get("adjusted_image_size", [])
            if not images:
                sample.pop("images", None)
                sample.pop("adjusted_image_size", None)
                return sample

            resized_images = [
                self._load_resized_image_tensor(image, target_size)
                for image, target_size in zip(images, adjusted_sizes, strict=True)
            ]

            image_inputs = processor.image_processor(
                images=resized_images,
                do_resize=False,
                return_tensors="pt",
            )
            sample.pop("images", None)
            sample.pop("adjusted_image_size", None)
            sample["pixel_values"] = image_inputs["pixel_values"]
            sample["image_grid_thw"] = image_inputs["image_grid_thw"]
            return sample
        except Exception as exc:
            simple_info(exc, level="debug")
            return build_empty_sample()

    def _merge_packed(self, samples: list[dict[str, Any]], packed_sample: dict[str, Any]) -> None:
        """Merge media tensors from source samples into a finalized packed sample."""
        pixel_values = [sample["pixel_values"] for sample in samples if sample.get("pixel_values") is not None]
        packed_sample["pixel_values"] = torch.cat(pixel_values, dim=0) if pixel_values else None

        image_grid_thw = [sample["image_grid_thw"] for sample in samples if sample.get("image_grid_thw") is not None]
        packed_sample["image_grid_thw"] = torch.cat(image_grid_thw, dim=0) if image_grid_thw else None

    def _ensure_text_only_batch_has_dummy_media(
        self,
        batch: list[dict[str, Any]],
        *,
        processor: Any,
        ignore_index: int,
    ) -> dict[str, torch.Tensor] | None:
        """Append one active dummy image suffix when a local batch has no images."""
        if any(sample.get("pixel_values") is not None for sample in batch):
            return None

        dummy_inputs = self._get_dummy_inputs(processor)
        first_sample = batch[0]
        dummy_input_ids = dummy_inputs["input_ids"]

        first_sample["input_ids"] = torch.cat([first_sample["input_ids"], dummy_input_ids], dim=0)
        first_sample["attention_mask"] = torch.cat(
            [
                first_sample["attention_mask"],
                dummy_inputs["attention_mask"].to(first_sample["attention_mask"].dtype),
            ],
            dim=0,
        )
        first_sample["labels"] = torch.cat(
            [
                first_sample["labels"],
                torch.full_like(dummy_input_ids, ignore_index),
            ],
            dim=0,
        )

        next_segment_id = int(first_sample["pack_segment_ids"].max().item()) + 1
        first_sample["pack_segment_ids"] = torch.cat(
            [
                first_sample["pack_segment_ids"],
                torch.full_like(
                    dummy_input_ids,
                    fill_value=next_segment_id,
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )
        return dummy_inputs

    def collate(
        self,
        batch: list[dict[str, Any]],
        model_inputs: dict[str, Any],
        *,
        dummy_inputs: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Merge model-family media tensors into a collated batch."""
        pixel_values = [sample["pixel_values"] for sample in batch if sample.get("pixel_values") is not None]
        if pixel_values:
            model_inputs["pixel_values"] = torch.cat(pixel_values, dim=0)
        elif dummy_inputs is not None:
            model_inputs["pixel_values"] = dummy_inputs["pixel_values"]

        image_grid_thw = [sample["image_grid_thw"] for sample in batch if sample.get("image_grid_thw") is not None]
        if image_grid_thw:
            model_inputs["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
        elif dummy_inputs is not None:
            model_inputs["image_grid_thw"] = dummy_inputs["image_grid_thw"]

    @staticmethod
    def _resolve_image_processor_config(processor: Any) -> tuple[Any, int, int, int, int]:
        """Resolve Qwen image processor geometry needed for token estimation and resizing."""
        image_processor = getattr(processor, "image_processor", processor)
        patch_size = getattr(image_processor, "patch_size", None)
        merge_size = getattr(image_processor, "merge_size", None)
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("Processor image processor must expose a positive integer `patch_size`.")
        if not isinstance(merge_size, int) or merge_size <= 0:
            raise ValueError("Processor image processor must expose a positive integer `merge_size`.")

        image_processor_size = getattr(image_processor, "size", {})
        min_pixels = getattr(processor, "min_image_size", None)
        max_pixels = getattr(processor, "max_image_size", None)
        if min_pixels is None and isinstance(image_processor_size, dict):
            min_pixels = image_processor_size.get("shortest_edge")
        if max_pixels is None and isinstance(image_processor_size, dict):
            max_pixels = image_processor_size.get("longest_edge")
        if not isinstance(min_pixels, int) or min_pixels <= 0:
            raise ValueError("Processor image processor must expose a positive integer min pixel budget.")
        if not isinstance(max_pixels, int) or max_pixels <= 0:
            raise ValueError("Processor image processor must expose a positive integer max pixel budget.")

        return image_processor, patch_size, merge_size, min_pixels, max_pixels

    @staticmethod
    def _smart_resize_image_size(
        height: int,
        width: int,
        *,
        factor: int,
        min_pixels: int,
        max_pixels: int,
    ) -> tuple[int, int] | None:
        """Resize an image size with Qwen2-VL smart-resize semantics."""
        try:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
                smart_resize,
            )

            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            return int(resized_height), int(resized_width)
        except Exception:
            pass

        try:
            height = int(height)
            width = int(width)
            if height <= 0 or width <= 0 or factor <= 0:
                return None
            if max(height, width) / min(height, width) > 200:
                return None

            resized_height = round(height / factor) * factor
            resized_width = round(width / factor) * factor
            if resized_height * resized_width > max_pixels:
                beta = math.sqrt((height * width) / max_pixels)
                resized_height = max(factor, math.floor(height / beta / factor) * factor)
                resized_width = max(factor, math.floor(width / beta / factor) * factor)
            elif resized_height * resized_width < min_pixels:
                beta = math.sqrt(min_pixels / (height * width))
                resized_height = math.ceil(height * beta / factor) * factor
                resized_width = math.ceil(width * beta / factor) * factor
        except Exception:
            return None

        return int(resized_height), int(resized_width)

    def _estimate_image_tokens(
        self,
        image_processor: Any,
        *,
        height: int,
        width: int,
        patch_size: int,
        merge_size: int,
        min_pixels: int,
        max_pixels: int,
    ) -> int | None:
        """Estimate Qwen2-VL language-side image tokens for one raw image size."""
        get_number_of_image_patches = getattr(image_processor, "get_number_of_image_patches", None)

        try:
            if callable(get_number_of_image_patches):
                num_patches = get_number_of_image_patches(
                    height,
                    width,
                    {
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                        "patch_size": patch_size,
                        "merge_size": merge_size,
                    },
                )
                return int(num_patches // (merge_size**2))
        except Exception:
            pass

        resized_size = self._smart_resize_image_size(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        if resized_size is None:
            return None

        resized_height, resized_width = resized_size
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size
        return int((grid_h * grid_w) // (merge_size**2))

    @staticmethod
    def _load_resized_image_tensor(image: Any, target_size: list[int]) -> torch.Tensor:
        """Load one image and resize it to its precomputed smart-resize shape."""
        import torchvision.transforms.v2.functional as tvF
        from torchvision.transforms import InterpolationMode

        height, width = int(target_size[0]), int(target_size[1])
        pil_image = read_image(image)
        image_tensor = tvF.pil_to_tensor(pil_image)
        if tuple(image_tensor.shape[-2:]) != (height, width):
            image_tensor = tvF.resize(
                image_tensor,
                [height, width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        return image_tensor

    def _get_dummy_image(self, processor: Any) -> Image.Image:
        """Return a cached RGB dummy image reused across text-only batches."""
        cached = getattr(processor, "_mvp_mllm_batch_dummy_image", None)
        if isinstance(cached, Image.Image):
            return cached.copy()

        dummy = Image.new("RGB", self.DUMMY_IMAGE_SIZE, color=0)
        setattr(processor, "_mvp_mllm_batch_dummy_image", dummy)
        return dummy.copy()

    def _get_dummy_inputs(self, processor: Any) -> dict[str, torch.Tensor]:
        """Build one valid minimal multimodal suffix for text-only local batches."""
        processor_id = id(processor)
        cached_inputs = self._cached_dummy_inputs.get(processor_id)
        if cached_inputs is not None:
            return {key: value.clone() for key, value in cached_inputs.items()}

        fake_messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": self._get_dummy_image(processor)}],
            }
        ]
        model_inputs = processor.apply_chat_template(
            [fake_messages],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            min_pixels=self.DUMMY_IMAGE_PIXELS,
            max_pixels=self.DUMMY_IMAGE_PIXELS,
        )
        self._cached_dummy_inputs[processor_id] = {
            "input_ids": model_inputs["input_ids"][0].to(dtype=torch.long),
            "attention_mask": model_inputs["attention_mask"][0].to(dtype=torch.long),
            "pixel_values": model_inputs["pixel_values"],
            "image_grid_thw": model_inputs["image_grid_thw"],
        }
        return {key: value.clone() for key, value in self._cached_dummy_inputs[processor_id].items()}
