"""Interleaved image-text sample adapters."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

from mvp_engine.kit import MLLMDataKit
from mvp_engine.kit.mllm.data.guard import CheckResult, DataGuard
from mvp_engine.kit.mllm.data.media import MLLMMediaKit
from mvp_engine.kit.mllm.data.sample import MLLMSampleKit
from mvp_engine.kit.mllm.data.types import CanonicalMedia, CanonicalMLLMSample


class InterleavedDataGuard(DataGuard):
    """Allow inline interleaved rows through the pre-tokenization guard."""

    def check(self, sample: Any) -> CheckResult:
        """Validate one sample while accepting OpenAI content-block media."""
        if not isinstance(sample, dict):
            return CheckResult(is_valid=False, reason="guard.not_dict")

        is_inline_interleaved = (
            InterleavedSampleKit._has_structured_messages(sample)
            or InterleavedSampleKit._has_interleaved_content(sample)
            or "_response" in sample
        )
        if is_inline_interleaved and self.check_basic_formats:
            sample.setdefault("images", [])
            messages = sample.get("messages") or sample.get("conversations")
            if messages is None and "content" in sample:
                sample["messages"] = [{"role": "user", "content": sample["content"]}]
            elif messages is None and "_response" in sample:
                sample["messages"] = []

        if is_inline_interleaved and self.check_image_sizes:
            sample.setdefault("image_size", [])
            return CheckResult(is_valid=True)

        return super().check(sample)


class InterleavedDataKit(MLLMDataKit):
    """MLLM data kit with interleaved-aware raw sample guards."""

    def build_dataguard(
        self,
        assemble_context=None,
        *,
        check_basic_formats: bool,
        check_input_ids: bool,
        check_image_sizes: bool,
        verbose: bool = True,
    ) -> DataGuard:
        """Build the interleaved data guard assembler."""
        del assemble_context
        return InterleavedDataGuard(
            check_basic_formats=check_basic_formats,
            check_input_ids=check_input_ids,
            check_image_sizes=check_image_sizes,
            verbose=verbose,
        )


class InterleavedSampleKit(MLLMSampleKit):
    """Normalize ShareGPT and OpenAI content-block interleaved rows."""

    def normalize(
        self,
        sample: dict[str, Any],
        *,
        image_placeholder: str | None = None,
    ) -> CanonicalMLLMSample:
        """Convert one raw row into canonical chat messages and media refs."""
        if self._has_structured_messages(sample):
            return self._normalize_structured_messages(sample)
        if self._has_interleaved_content(sample):
            return self._normalize_structured_messages(
                {
                    **sample,
                    "messages": [{"role": "user", "content": sample["content"]}],
                }
            )
        if "_response" in sample:
            return self._normalize_llamafactory_converted(sample)
        return super().normalize(sample, image_placeholder=image_placeholder)

    def _normalize_structured_messages(self, sample: dict[str, Any]) -> CanonicalMLLMSample:
        raw_messages = sample.get("messages") or sample.get("conversations")
        messages = [self._normalize_structured_message(message) for message in raw_messages]

        if len(messages) == 1 and messages[0]["role"] == "user":
            messages = [
                {"role": "user", "content": []},
                {"role": "assistant", "content": messages[0]["content"]},
            ]

        media: list[CanonicalMedia] = []
        for message in messages:
            for block in message["content"]:
                if block.get("type") == "image":
                    media.append(
                        CanonicalMedia(
                            type="image",
                            value=block["image"],
                            size=block.get("size"),
                            metadata={"source_index": len(media)},
                        )
                    )

        metadata = {
            key: sample[key] for key in ("id", "source", "__source__", "__key__", "__global_index__") if key in sample
        }
        return CanonicalMLLMSample(messages=messages, media=media, metadata=metadata)

    def _normalize_structured_message(self, message: dict[str, Any]) -> dict[str, Any]:
        role = self.role_map.get(message.get("role") or message.get("from"))
        if role is None:
            raise ValueError(f"contains an invalid role: {message.get('role') or message.get('from')!r}")

        content = message.get("content", message.get("value", ""))
        blocks = self._normalize_content_blocks(content)
        return {"role": role, "content": blocks}

    def _normalize_content_blocks(self, content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            parsed = self._maybe_load_json(content)
            if not isinstance(parsed, list):
                return [{"type": "text", "text": content}] if content else []
            content = parsed

        if not isinstance(content, list):
            raise ValueError("structured interleaved content must be a string or list.")

        blocks: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("structured interleaved content blocks must be dicts.")

            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                text = item.get("text", "")
                if text:
                    blocks.append({"type": "text", "text": str(text)})
                continue
            if item_type in {"image", "input_image", "image_url"}:
                image = self._extract_image_value(item)
                size = self._extract_image_size(item, image)
                blocks.append({"type": "image", "image": image, "size": size})

        return blocks

    def _normalize_llamafactory_converted(self, sample: dict[str, Any]) -> CanonicalMLLMSample:
        images = list(sample.get("_images") or [])
        image_sizes = list(sample.get("image_size") or sample.get("img_size") or [])
        response = sample.get("_response") or [{"role": "assistant", "content": ""}]
        content = response[0].get("content", "") if response else ""

        blocks: list[dict[str, Any]] = []
        media: list[CanonicalMedia] = []
        parts = str(content).split(self.image_placeholder)
        for part_index, part in enumerate(parts):
            if part:
                blocks.append({"type": "text", "text": part})
            if part_index >= len(parts) - 1:
                continue
            if part_index >= len(images):
                raise ValueError("has more image placeholders than images.")

            size = _read_size_metadata(image_sizes[part_index]) if part_index < len(image_sizes) else None
            image = images[part_index]
            blocks.append({"type": "image", "image": image, "size": size})
            media.append(
                CanonicalMedia(
                    type="image",
                    value=image,
                    size=size,
                    metadata={"source_index": len(media)},
                )
            )

        if len(images) != len(media):
            raise ValueError("has more images than image placeholders.")

        return CanonicalMLLMSample(
            messages=[
                {"role": "user", "content": []},
                {"role": "assistant", "content": blocks},
            ],
            media=media,
            metadata={},
        )

    @staticmethod
    def _has_structured_messages(sample: dict[str, Any]) -> bool:
        messages = sample.get("messages") or sample.get("conversations")
        if not isinstance(messages, list):
            return False
        return any(
            isinstance(message, dict) and InterleavedSampleKit._is_structured_content(message.get("content"))
            for message in messages
        )

    @staticmethod
    def _has_interleaved_content(sample: dict[str, Any]) -> bool:
        return "content" in sample and InterleavedSampleKit._is_structured_content(sample.get("content"))

    @staticmethod
    def _is_structured_content(content: Any) -> bool:
        if isinstance(content, list):
            return True
        if isinstance(content, str):
            return isinstance(InterleavedSampleKit._maybe_load_json(content), list)
        return False

    @staticmethod
    def _maybe_load_json(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _extract_image_value(item: dict[str, Any]) -> Any:
        image_file = item.get("image_file")
        if isinstance(image_file, dict) and "image" in image_file:
            return image_file["image"]
        if "image" in item:
            return item["image"]

        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            return image_url.get("url")
        return image_url

    @staticmethod
    def _extract_image_size(item: dict[str, Any], image: Any) -> list[int] | None:
        for candidate in (item.get("size"), item.get("image_size"), image):
            size = _read_size_metadata(candidate)
            if size is not None:
                return size
        return None


class InterleavedMediaKit(MLLMMediaKit):
    """Qwen image media kit with image-size fallback for inline image records."""

    def prepare(self, media: list[CanonicalMedia], *, processor: Any, tokenizer: Any):
        """Fill missing image sizes before using the standard Qwen media path."""
        normalized_media: list[CanonicalMedia] = []
        for item in media:
            if item.type != "image" or item.size is not None:
                normalized_media.append(item)
                continue

            inferred_size = infer_image_size(item.value)
            if inferred_size is None:
                raise ValueError("interleaved image media must provide image_size/img_size or decodable image bytes.")
            normalized_media.append(
                CanonicalMedia(type=item.type, value=item.value, size=inferred_size, metadata=item.metadata)
            )
        return super().prepare(normalized_media, processor=processor, tokenizer=tokenizer)


def build_interleaved_data_kit() -> MLLMDataKit:
    """Build the recipe-local MLLM data kit."""
    return InterleavedDataKit(sample_kit=InterleavedSampleKit(), media_kit=InterleavedMediaKit())


def infer_image_size(image: Any) -> list[int] | None:
    """Infer ``[height, width]`` from an inline image record."""
    size = _read_size_metadata(image)
    if size is not None:
        return size

    if isinstance(image, Image.Image):
        return [int(image.height), int(image.width)]
    if isinstance(image, dict):
        image_bytes = image.get("bytes") or image.get("image_bytes")
        if isinstance(image_bytes, (bytes, bytearray, memoryview)):
            return _infer_size_from_bytes(bytes(image_bytes))

        image_path = image.get("path")
        if isinstance(image_path, str) and image_path:
            return infer_image_size(image_path)
        return None
    if isinstance(image, (bytes, bytearray, memoryview)):
        return _infer_size_from_bytes(bytes(image))
    if isinstance(image, str):
        if image.startswith("data:") and "," in image:
            return _infer_size_from_bytes(base64.b64decode(image.split(",", 1)[1]))
        image_path = Path(image).expanduser()
        if image_path.is_file():
            with Image.open(image_path) as opened:
                return [int(opened.height), int(opened.width)]

    return None


def _infer_size_from_bytes(image_bytes: bytes) -> list[int] | None:
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            return [int(opened.height), int(opened.width)]
    except Exception:
        return None


def _read_size_metadata(value: Any) -> list[int] | None:
    if isinstance(value, dict):
        width = value.get("width") or value.get("w")
        height = value.get("height") or value.get("h")
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            return [int(height), int(width)]
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        width, height = value[0], value[1]
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            return [int(height), int(width)]
    return None
