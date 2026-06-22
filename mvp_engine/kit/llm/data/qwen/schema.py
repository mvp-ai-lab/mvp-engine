"""Qwen text-chat schema normalization for LLM data pipelines."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from ..schema import LLMSchemaHandler
from ..types import LLMSegment

ROLE_MAP = {
    "assistant": "assistant",
    "gpt": "assistant",
    "human": "user",
    "system": "system",
    "tool": "tool",
    "user": "user",
}

THOUGHT_PREFIX = "<think>\n"
THOUGHT_SUFFIX = "\n</think>\n\n"
THOUGHT_PATTERN = re.compile(f"{re.escape(THOUGHT_PREFIX)}(.*?){re.escape(THOUGHT_SUFFIX)}", re.DOTALL)
THOUGHT_MARKERS = (THOUGHT_PREFIX.strip(), THOUGHT_SUFFIX.strip())


class QwenChatSchemaHandler(LLMSchemaHandler):
    """Normalize Qwen text-chat rows into loss-marked source and assistant target segments."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        thinking_mode: bool | None | Literal["non-empty"] = True,
    ) -> None:
        """Store Qwen chat-template options."""
        if not (
            thinking_mode is True or thinking_mode is False or thinking_mode is None or thinking_mode == "non-empty"
        ):
            raise ValueError("thinking_mode must be True, False, None, or 'non-empty'.")
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Qwen chat schema requires a tokenizer with apply_chat_template().")
        self.tokenizer = tokenizer
        self.thinking_mode = thinking_mode

    def check_row(self, row: Mapping[str, Any]) -> str | None:
        """Validate the configured conversation field before template rendering."""
        messages = row.get("messages") or row.get("conversations")
        if not isinstance(messages, list) or not messages:
            return "raw.invalid_messages"
        for message in messages:
            if not isinstance(message, dict):
                return "raw.invalid_message"
            content = message.get("content") if "content" in message else message.get("value")
            if not isinstance(content, (str, list)):
                return "raw.invalid_message_content"
        return None

    def normalize(self, row: Mapping[str, Any]) -> tuple[list[LLMSegment], dict[str, Any]]:
        """Render one Qwen conversation into source/target text segments."""
        reason = self.check_row(row)
        if reason is not None:
            raise ValueError(reason)

        messages = row.get("messages") or row.get("conversations")
        qwen_messages: list[tuple[dict[str, Any], bool]] = []
        for message in messages:
            normalized_message = self._normalize_message(message)
            content = normalized_message["content"]
            skip_think_prefix = False
            if normalized_message["role"] == "assistant":
                content, skip_think_prefix = self._apply_thinking_mode(content)
            qwen_messages.append(({"role": normalized_message["role"], "content": content}, skip_think_prefix))

        segments = [
            LLMSegment(type="text", loss=loss, value=text)
            for text, loss in self._render_chat_template_segments(qwen_messages)
            if text
        ]
        return segments, {"schema": "qwen_chat"}

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
            source_text = self.tokenizer.apply_chat_template(
                source_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if not full_text.startswith(source_text):
                raise ValueError("tokenizer chat template does not preserve source prefix for target split.")

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

    @staticmethod
    def _normalize_message(message: dict[str, Any]) -> dict[str, str]:
        """Normalize one raw conversation message to Qwen role and text content."""
        role = message.get("role")
        content = message.get("content")
        if isinstance(role, str) and role:
            normalized_role = ROLE_MAP.get(role)
            if normalized_role is None:
                raise ValueError(f"contains an invalid role: {role!r}")
            return {"role": normalized_role, "content": _message_text(content)}

        source_role = message.get("from")
        normalized_role = ROLE_MAP.get(source_role)
        if normalized_role is None:
            raise ValueError(f"contains an invalid role: {source_role!r}")
        return {"role": normalized_role, "content": _message_text(message.get("value"))}


def _message_text(content: Any) -> str:
    """Return text from either a plain string or OpenAI-style text content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type", "text") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    raise ValueError("contains non-string content.")
