"""Qwen VL schema normalization for conversation-style MLLM rows."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from ..schema import ROLE_MAP, MLLMSchemaHandler
from ..types import MLLMMediaSlot, MLLMSegment

THOUGHT_PREFIX = "<think>\n"
THOUGHT_SUFFIX = "\n</think>\n\n"
THOUGHT_PATTERN = re.compile(f"{re.escape(THOUGHT_PREFIX)}(.*?){re.escape(THOUGHT_SUFFIX)}", re.DOTALL)
THOUGHT_MARKERS = (THOUGHT_PREFIX.strip(), THOUGHT_SUFFIX.strip())
MULTIMODAL_PLACEHOLDER = "<|mvp_multimodal_placeholder|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"


class QwenChatSchemaHandler(MLLMSchemaHandler):
    """Normalize Qwen conversation rows into loss-marked text and image segments.

    Attributes:
        processor: Qwen processor used to apply the model chat template.
        thinking_mode: Qwen thinking-block normalization mode.
    """

    def __init__(
        self,
        processor: Any,
        *,
        thinking_mode: bool | None | Literal["non-empty"] = True,
        image_placeholders: tuple[str, ...] | None = None,
    ) -> None:
        """Store Qwen schema options.

        Args:
            processor: Qwen processor whose chat template is used for source/target rendering.
            thinking_mode: Controls Qwen ``<think>`` block normalization. ``True`` keeps
                or inserts thinking blocks, ``False`` removes them from assistant content,
                ``"non-empty"`` keeps only non-empty thinking blocks, and ``None`` leaves
                content unchanged.
            image_placeholders: Extra raw text placeholders that should consume image media slots.

        Raises:
            ValueError: If ``thinking_mode`` is not supported.
        """
        if not (
            thinking_mode is True or thinking_mode is False or thinking_mode is None or thinking_mode == "non-empty"
        ):
            raise ValueError("thinking_mode must be True, False, None, or 'non-empty'.")

        self.processor = processor
        self.thinking_mode = thinking_mode
        image_token = getattr(self.processor, "image_token", DEFAULT_IMAGE_TOKEN)
        wrapped_image_token = f"{VISION_START_TOKEN}{image_token}{VISION_END_TOKEN}"
        self._image_placeholders = tuple(
            dict.fromkeys(
                (*(image_placeholders or ()), wrapped_image_token, "<image>", "<|mvp_image_placeholder|>", image_token)
            )
        )

    def normalize(self, row: Mapping[str, Any]) -> tuple[list[MLLMSegment], list[MLLMMediaSlot], dict[str, Any]]:
        """Normalize one Qwen conversation row.

        Args:
            row: Raw row containing ``messages`` or ``conversations`` plus image media columns.

        Returns:
            A tuple of ``(segments, media_slots, metadata)``. User/system template text
            has ``loss=False``, assistant target text has ``loss=True``, and image
            segments explicitly have ``loss=False``.

        Raises:
            ValueError: If the conversation, media metadata, role mapping, or media
                placeholder count is invalid.
        """
        raw = dict(row)
        messages = raw.get("messages") or raw.get("conversations")
        if not isinstance(messages, list):
            raise ValueError("contains invalid messages.")

        media = self._normalize_media(raw)
        media_cursor = 0
        qwen_messages: list[tuple[dict[str, Any], bool]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("contains invalid message.")

            normalized_message = self._normalize_message(message)
            content = normalized_message["content"]
            skip_think_prefix = False
            if normalized_message["role"] == "assistant":
                content, skip_think_prefix = self._apply_thinking_mode(content)

            parts, media_cursor = self._split_text_and_media(
                content,
                media=media,
                media_cursor=media_cursor,
            )
            qwen_message = {
                "role": normalized_message["role"],
                "content": [
                    {"type": "text", "text": part}
                    if isinstance(part, str)
                    else {"type": part.media_type, part.media_type: MULTIMODAL_PLACEHOLDER}
                    for part in parts
                ],
            }
            if normalized_message["role"] == "assistant" and "tool_calls" in message:
                qwen_message["tool_calls"] = message["tool_calls"]
            qwen_messages.append((qwen_message, skip_think_prefix))

        if media_cursor != len(media):
            raise ValueError("has more media entries than media placeholders.")

        text_segments = self._render_chat_template_segments(qwen_messages)
        media_cursor = 0
        segments = []
        for text, loss in text_segments:
            parts, media_cursor = self._split_text_and_media(text, media=media, media_cursor=media_cursor)
            for part in parts:
                if isinstance(part, str):
                    segments.append(MLLMSegment(type="text", loss=loss, value=part))
                else:
                    segments.append(MLLMSegment(type=part.media_type, loss=False, value=part.media_id))
        if media_cursor != len(media):
            raise ValueError("rendered chat template did not preserve all media placeholders.")
        return segments, media, {}

    def _render_chat_template_segments(
        self,
        messages: list[tuple[dict[str, Any], bool]],
    ) -> list[tuple[str, bool]]:
        """Render Qwen chat turns and split each user/assistant pair into source and target text."""
        segments: list[tuple[str, bool]] = []
        leading_system_messages: list[dict[str, Any]] = []
        message_index = 0
        while message_index < len(messages) and messages[message_index][0].get("role") == "system":
            leading_system_messages.append(messages[message_index][0])
            message_index += 1

        empty_thought = f"{THOUGHT_PREFIX}{THOUGHT_SUFFIX}"
        while message_index < len(messages):
            user_message = messages[message_index][0]
            if user_message.get("role") != "user":
                raise ValueError("conversation must contain user/assistant turn pairs after optional system messages.")
            if message_index + 1 >= len(messages):
                break

            assistant_message, skip_think_prefix = messages[message_index + 1]
            if assistant_message.get("role") != "assistant":
                raise ValueError("conversation must contain user/assistant turn pairs after optional system messages.")

            source_messages = leading_system_messages + [user_message]
            full_messages = source_messages + [assistant_message]
            source_text = self.processor.apply_chat_template(
                source_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if not full_text.startswith(source_text):
                raise ValueError("processor chat template does not preserve source prefix for assistant target split.")

            target_text = full_text[len(source_text) :]
            if skip_think_prefix and target_text.startswith(empty_thought):
                source_text += empty_thought
                target_text = target_text[len(empty_thought) :]

            segments.append((source_text, False))
            segments.append((target_text, True))
            leading_system_messages = []
            message_index += 2

        return segments

    def _apply_thinking_mode(self, content: str) -> tuple[str, bool]:
        """Normalize Qwen thinking text and report whether an inserted empty-think prefix is source-only."""
        thought_match = THOUGHT_PATTERN.search(content)
        thought_is_empty = thought_match is None or not thought_match.group(1).strip()
        modified_text = content
        if self.thinking_mode is False:
            modified_text = THOUGHT_PATTERN.sub("", content).lstrip("\n")
        elif self.thinking_mode == "non-empty" and thought_is_empty:
            modified_text = THOUGHT_PATTERN.sub("", content).lstrip("\n")

        has_thought_block = all(marker in modified_text for marker in THOUGHT_MARKERS)
        if has_thought_block or self.thinking_mode is None:
            return modified_text, False

        skip_prefix = self.thinking_mode is False or (self.thinking_mode == "non-empty" and thought_is_empty)
        return f"{THOUGHT_PREFIX}{THOUGHT_SUFFIX}{modified_text}", skip_prefix

    def _split_text_and_media(
        self,
        text: str,
        *,
        media: list[MLLMMediaSlot],
        media_cursor: int,
    ) -> tuple[list[str | MLLMMediaSlot], int]:
        """Split text on image placeholders and replace them with ordered media slots."""
        parts: list[str | MLLMMediaSlot] = []
        cursor = 0
        while cursor < len(text):
            placeholder: tuple[int, int] | None = None
            for value in self._image_placeholders:
                if not value:
                    continue
                start = text.find(value, cursor)
                if start < 0:
                    continue
                end = start + len(value)
                if placeholder is None or start < placeholder[0] or (start == placeholder[0] and end > placeholder[1]):
                    placeholder = (start, end)

            if placeholder is None:
                break

            start, end = placeholder
            if start > cursor:
                parts.append(text[cursor:start])
            if media_cursor >= len(media):
                raise ValueError("has more media placeholders than media entries.")

            media_item = media[media_cursor]
            if media_item.media_type != "image":
                raise ValueError("media placeholder order does not match media entry order.")
            parts.append(media_item)
            media_cursor += 1
            cursor = end

        if cursor < len(text):
            parts.append(text[cursor:])
        return parts, media_cursor

    @classmethod
    def _normalize_media(cls, row: dict[str, Any]) -> list[MLLMMediaSlot]:
        """Normalize supported raw media columns into image media slots."""
        raw_media = row.get("media")
        if raw_media is not None:
            if not isinstance(raw_media, (list, tuple)):
                raise ValueError("contains invalid media.")
            return [cls._normalize_media_entry(entry, index=index) for index, entry in enumerate(raw_media)]

        images = row.get("images", [])
        if images is None:
            images = []
        if not isinstance(images, (list, tuple)):
            raise ValueError("contains invalid images.")

        raw_image_sizes = row.get("img_size", []) or row.get("image_size", [])
        if raw_image_sizes is None:
            raw_image_sizes = []
        if not isinstance(raw_image_sizes, (list, tuple)):
            raise ValueError("contains invalid image size metadata.")
        if len(images) != len(raw_image_sizes):
            raise ValueError("image count does not match image size metadata count.")

        return [
            MLLMMediaSlot(
                media_id=f"image:{index}",
                media_type="image",
                field="images",
                index=index,
                metadata={"size": cls._normalize_image_size(size)},
            )
            for index, size in enumerate(raw_image_sizes)
        ]

    @classmethod
    def _normalize_media_entry(cls, entry: Any, *, index: int) -> MLLMMediaSlot:
        """Normalize one explicit media entry into an image media slot."""
        if not isinstance(entry, dict):
            raise ValueError("contains invalid media entry.")
        media_type = entry.get("type", "image")
        if media_type != "image":
            raise ValueError(f"contains unsupported media type: {media_type!r}")
        if entry.get("value", entry.get(media_type)) is None:
            raise ValueError("contains media entry without value.")
        raw_size = entry.get("size") or entry.get("image_size") or entry.get("img_size")
        if raw_size is None:
            raise ValueError("image media must include size metadata.")
        return MLLMMediaSlot(
            media_id=f"{media_type}:{index}",
            media_type=media_type,
            field="media",
            index=index,
            metadata={"size": cls._normalize_image_size(raw_size)},
        )

    @staticmethod
    def _normalize_image_size(size_entry: Any) -> list[int]:
        """Return image size as ``[height, width]`` after validation."""
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

    @staticmethod
    def _normalize_message(message: dict[str, Any]) -> dict[str, str]:
        """Normalize one raw conversation message to Qwen role and content strings."""
        role = message.get("role")
        content = message.get("content")
        if isinstance(role, str) and isinstance(content, str) and role:
            normalized_role = ROLE_MAP.get(role)
            if normalized_role is None:
                raise ValueError(f"contains an invalid role: {role!r}")
            return {"role": normalized_role, "content": content}

        source_role = message.get("from")
        source_content = message.get("value")
        normalized_role = ROLE_MAP.get(source_role)
        if normalized_role is None:
            raise ValueError(f"contains an invalid role: {source_role!r}")
        if not isinstance(source_content, str):
            raise ValueError("contains non-string content.")
        return {"role": normalized_role, "content": source_content}
