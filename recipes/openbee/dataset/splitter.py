"""Raw conversation splitting utilities for the OpenBee recipe."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mvp_dataset.core import Assembler, RuntimeContext

IMAGE_PLACEHOLDER = "<image>"
ROLE_MAP = {
    "assistant": "assistant",
    "gpt": "assistant",
    "human": "user",
    "system": "system",
    "tool": "tool",
    "user": "user",
}


def _message_role(message: dict[str, Any]) -> str | None:
    """Return the canonical role for a raw message, when recognized."""
    role = message.get("role")
    if role is None:
        role = message.get("from")
    if not isinstance(role, str):
        return None
    return ROLE_MAP.get(role)


def _message_content(message: dict[str, Any]) -> Any:
    """Return the raw content payload from either supported message schema."""
    if "content" in message:
        return message.get("content")
    return message.get("value")


def _content_length(content: Any) -> int:
    """Estimate text length for string or multimodal block content."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content") or block.get("value")
                if isinstance(text, str):
                    total += len(text)
            elif isinstance(block, str):
                total += len(block)
        return total
    return 0


def _count_image_placeholders(content: Any) -> int:
    """Count image placeholders represented by text markers or image blocks."""
    if isinstance(content, str):
        return content.count(IMAGE_PLACEHOLDER)
    if isinstance(content, list):
        count = 0
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image" or "image" in block or "image_url" in block:
                count += 1
            text = block.get("text") or block.get("content") or block.get("value")
            if isinstance(text, str):
                count += text.count(IMAGE_PLACEHOLDER)
        return count
    return 0


def _image_size_list(sample: dict[str, Any], key: str, image_count: int) -> list[Any] | None:
    """Normalize one image-size metadata field to a list."""
    value = sample.get(key)
    if value is None:
        return None
    if isinstance(value, tuple):
        value = list(value)
    if image_count == 1 and isinstance(value, list) and len(value) == 2 and all(isinstance(x, int) for x in value):
        return [value]
    if isinstance(value, list):
        return value
    return None


class LongConversationSplitAssembler(Assembler[dict[str, Any], dict[str, Any]]):
    """Split very long raw conversations into smaller turn windows."""

    def __init__(
        self,
        *,
        max_turns: int | None = None,
        max_chars: int | None = None,
        overlap_turns: int = 0,
    ) -> None:
        """Configure turn-count, character-count, and overlap split limits."""
        super().__init__()
        if max_turns is not None and max_turns <= 0:
            raise ValueError(f"max_turns must be positive when set, got {max_turns}.")
        if max_chars is not None and max_chars <= 0:
            raise ValueError(f"max_chars must be positive when set, got {max_chars}.")
        if overlap_turns < 0:
            raise ValueError(f"overlap_turns must be non-negative, got {overlap_turns}.")
        if max_turns is not None and overlap_turns >= max_turns:
            raise ValueError("overlap_turns must be smaller than max_turns.")

        self.max_turns = max_turns
        self.max_chars = max_chars
        self.overlap_turns = overlap_turns

    def push(self, sample: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Split one raw conversation sample into smaller windows when needed."""
        if self.max_turns is None and self.max_chars is None:
            return [sample]

        message_key = "messages" if isinstance(sample.get("messages"), list) else "conversations"
        messages = sample.get(message_key)
        if not isinstance(messages, list) or not messages:
            return [sample]

        system_indices, turns = self._split_turns(messages)
        if not turns:
            return [sample]

        sample_chars = self._message_indices_length(messages, range(len(messages)))
        if not self._needs_split(len(turns), sample_chars):
            return [sample]

        windows = self._build_windows(messages, system_indices, turns)
        if len(windows) <= 1:
            return [sample]

        image_indices_by_message = self._image_indices_by_message(messages)
        return [
            self._build_window_sample(
                sample,
                message_key=message_key,
                messages=messages,
                message_indices=message_indices,
                image_indices_by_message=image_indices_by_message,
                window_index=window_index,
                window_count=len(windows),
                turn_start=turn_start,
                turn_end=turn_end,
            )
            for window_index, (turn_start, turn_end, message_indices) in enumerate(windows)
        ]

    def finish(self, *, drop_last: bool = False) -> Iterable[dict[str, Any]]:
        """Emit nothing at stream end because this splitter is stateless."""
        del drop_last
        return []

    def _needs_split(self, turn_count: int, char_count: int) -> bool:
        """Return whether the sample exceeds configured split limits."""
        if self.max_turns is not None and turn_count > self.max_turns:
            return True
        return self.max_chars is not None and char_count > self.max_chars

    def _split_turns(self, messages: list[dict[str, Any]]) -> tuple[list[int], list[list[int]]]:
        """Separate leading system messages from user-started conversation turns."""
        system_indices: list[int] = []
        start_index = 0
        while start_index < len(messages) and _message_role(messages[start_index]) == "system":
            system_indices.append(start_index)
            start_index += 1

        turns: list[list[int]] = []
        current: list[int] = []
        for index in range(start_index, len(messages)):
            role = _message_role(messages[index])
            if role == "user" and current:
                turns.append(current)
                current = [index]
            else:
                current.append(index)
        if current:
            turns.append(current)

        return system_indices, turns

    def _build_windows(
        self,
        messages: list[dict[str, Any]],
        system_indices: list[int],
        turns: list[list[int]],
    ) -> list[tuple[int, int, list[int]]]:
        """Build overlapping turn windows that include leading system messages."""
        windows: list[tuple[int, int, list[int]]] = []
        system_chars = self._message_indices_length(messages, system_indices)
        turn_chars = [self._message_indices_length(messages, turn) for turn in turns]

        start = 0
        while start < len(turns):
            end = start
            char_count = system_chars
            while end < len(turns):
                selected_turn_count = end - start
                next_chars = turn_chars[end]
                too_many_turns = self.max_turns is not None and selected_turn_count >= self.max_turns
                too_many_chars = (
                    self.max_chars is not None
                    and selected_turn_count > 0
                    and (char_count + next_chars > self.max_chars)
                )
                if too_many_turns or too_many_chars:
                    break

                char_count += next_chars
                end += 1

            if end == start:
                end += 1

            message_indices = list(system_indices)
            for turn in turns[start:end]:
                message_indices.extend(turn)
            windows.append((start, end, message_indices))

            if end >= len(turns):
                break

            next_start = end - min(self.overlap_turns, end - start - 1)
            start = max(next_start, start + 1)

        return windows

    def _message_indices_length(self, messages: list[dict[str, Any]], indices: Iterable[int]) -> int:
        """Estimate total content length for selected message indices."""
        return sum(_content_length(_message_content(messages[index])) for index in indices)

    def _image_indices_by_message(self, messages: list[dict[str, Any]]) -> list[list[int]]:
        """Map each message to the image indices consumed by its placeholders."""
        next_image_index = 0
        image_indices_by_message: list[list[int]] = []
        for message in messages:
            placeholder_count = _count_image_placeholders(_message_content(message))
            image_indices = list(range(next_image_index, next_image_index + placeholder_count))
            image_indices_by_message.append(image_indices)
            next_image_index += placeholder_count
        return image_indices_by_message

    def _build_window_sample(
        self,
        sample: dict[str, Any],
        *,
        message_key: str,
        messages: list[dict[str, Any]],
        message_indices: list[int],
        image_indices_by_message: list[list[int]],
        window_index: int,
        window_count: int,
        turn_start: int,
        turn_end: int,
    ) -> dict[str, Any]:
        """Create one split sample with matching messages, images, and metadata."""
        window = dict(sample)
        window[message_key] = [messages[index] for index in message_indices]

        image_indices: list[int] = []
        for message_index in message_indices:
            image_indices.extend(image_indices_by_message[message_index])

        images = sample.get("images", [])
        if isinstance(images, tuple):
            images = list(images)
        if isinstance(images, list):
            window["images"] = [images[index] for index in image_indices if index < len(images)]

        for size_key in ("img_size", "image_size", "adjusted_image_size"):
            image_sizes = _image_size_list(sample, size_key, len(images) if isinstance(images, list) else 0)
            if image_sizes is not None:
                if any(index >= len(image_sizes) for index in image_indices):
                    window.pop(size_key, None)
                    continue
                window[size_key] = [image_sizes[index] for index in image_indices]

        sample_id = sample.get("id")
        if isinstance(sample_id, str) and sample_id:
            window["id"] = f"{sample_id}::split_{window_index:04d}_of_{window_count:04d}"
            window["openbee_split_parent_id"] = sample_id
        window["openbee_split_window_index"] = window_index
        window["openbee_split_window_count"] = window_count
        window["openbee_split_turn_start"] = turn_start
        window["openbee_split_turn_end"] = turn_end
        return window


def build_long_conversation_split_assembler(
    assemble_context: RuntimeContext,
    *,
    max_turns: int | None = None,
    max_chars: int | None = None,
    overlap_turns: int = 0,
) -> LongConversationSplitAssembler:
    """Build a raw-conversation splitter for the OpenBee dataset pipeline."""
    del assemble_context
    return LongConversationSplitAssembler(max_turns=max_turns, max_chars=max_chars, overlap_turns=overlap_turns)
