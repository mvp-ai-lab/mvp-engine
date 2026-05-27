"""Sample-schema normalization for MLLM data pipelines."""

from __future__ import annotations

from typing import Any

from .types import CanonicalMedia, CanonicalMLLMSample

ROLE_MAP = {
    "assistant": "assistant",
    "gpt": "assistant",
    "human": "user",
    "system": "system",
    "tool": "tool",
    "user": "user",
}

IMAGE_PLACEHOLDER = "<image>"


class MLLMSampleKit:
    """Normalize raw dataset rows into canonical MLLM samples."""

    def __init__(
        self,
        *,
        role_map: dict[str, str] | None = None,
        image_placeholder: str = IMAGE_PLACEHOLDER,
    ) -> None:
        """Configure raw role aliases and the image placeholder used by the source schema."""
        self.role_map = dict(role_map or ROLE_MAP)
        self.image_placeholder = image_placeholder

    def normalize(
        self,
        sample: dict[str, Any],
        *,
        image_placeholder: str | None = None,
    ) -> CanonicalMLLMSample:
        """Convert one raw row into canonical chat messages and ordered media references."""
        messages = sample.get("messages") or sample.get("conversations")
        if not isinstance(messages, list):
            raise ValueError("contains invalid messages.")

        images = sample.get("images", [])
        if images is None:
            images = []
        if not isinstance(images, (list, tuple)):
            raise ValueError("contains invalid images.")
        images = list(images)

        raw_image_sizes = sample.get("img_size", []) or sample.get("image_size", [])
        if raw_image_sizes is None:
            raw_image_sizes = []
        if not isinstance(raw_image_sizes, (list, tuple)):
            raise ValueError("contains invalid image size metadata.")
        raw_image_sizes = list(raw_image_sizes)

        image_sizes = [self._normalize_image_size(size) for size in raw_image_sizes]
        if len(images) != len(image_sizes):
            raise ValueError("image count does not match image size metadata count.")

        placeholder = self.image_placeholder if image_placeholder is None else image_placeholder
        image_cursor = 0
        canonical_messages: list[dict[str, Any]] = []
        canonical_media: list[CanonicalMedia] = []

        for message in messages:
            normalized_message = self._normalize_message(message)
            blocks: list[dict[str, Any]] = []
            content = normalized_message["content"]
            segments = content.split(placeholder)

            for segment_index, segment in enumerate(segments):
                if segment:
                    blocks.append({"type": "text", "text": segment})
                if segment_index >= len(segments) - 1:
                    continue
                if image_cursor >= len(images):
                    raise ValueError("has more image placeholders than images.")

                image_value = images[image_cursor]
                image_size = image_sizes[image_cursor]
                blocks.append({"type": "image", "image": image_value})
                canonical_media.append(
                    CanonicalMedia(
                        type="image",
                        value=image_value,
                        size=image_size,
                        metadata={"source_index": image_cursor},
                    )
                )
                image_cursor += 1

            rendered_message: dict[str, Any] = {
                "role": normalized_message["role"],
                "content": blocks,
            }
            if normalized_message["role"] == "assistant" and "tool_calls" in message:
                rendered_message["tool_calls"] = message["tool_calls"]
            canonical_messages.append(rendered_message)

        if image_cursor != len(images):
            raise ValueError("has more images than image placeholders.")

        metadata = {
            key: sample[key] for key in ("id", "source", "__source__", "__key__", "__global_index__") if key in sample
        }
        return CanonicalMLLMSample(messages=canonical_messages, media=canonical_media, metadata=metadata)

    def _normalize_message(self, message: dict[str, Any]) -> dict[str, str]:
        """Normalize source role/content aliases to the canonical chat schema."""
        role = message.get("role")
        content = message.get("content")
        if isinstance(role, str) and isinstance(content, str) and role:
            normalized_role = self.role_map.get(role)
            if normalized_role is None:
                raise ValueError(f"contains an invalid role: {role!r}")
            return {"role": normalized_role, "content": content}

        source_role = message.get("from")
        source_content = message.get("value")
        normalized_role = self.role_map.get(source_role)
        if normalized_role is None:
            raise ValueError(f"contains an invalid role: {source_role!r}")
        if not isinstance(source_content, str):
            raise ValueError("contains non-string content.")
        return {"role": normalized_role, "content": source_content}

    @staticmethod
    def _normalize_image_size(size_entry: Any) -> list[int]:
        """Parse image metadata into ``[height, width]``."""
        if isinstance(size_entry, dict):
            width = size_entry.get("width")
            height = size_entry.get("height")
        elif isinstance(size_entry, (list, tuple)) and len(size_entry) >= 2:
            width = size_entry[0]
            height = size_entry[1]
        else:
            raise ValueError(f"contains invalid image size metadata: {size_entry!r}")

        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError(f"contains invalid image size metadata: {size_entry!r}")
        return [int(height), int(width)]
