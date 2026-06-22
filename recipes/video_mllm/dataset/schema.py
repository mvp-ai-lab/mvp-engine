"""Schema normalization for video MLLM DataKit samples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mvp_engine.kit import MLLMMediaSlot, MLLMSchemaHandler, MLLMSegment
from mvp_engine.kit.mllm.data.schema import ROLE_MAP

VIDEO_PLACEHOLDER = "<video>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"


IMAGE_PLACEHOLDER = "<image>"


class VideoChatSchemaHandler(MLLMSchemaHandler):
    """Normalize single-visual chat rows into loss-marked DataKit segments.

    Handles a single video clip or a single still image. An image row (``image`` /
    ``images`` field with an ``<image>`` placeholder, e.g. OpenBee alignment data) is
    treated as one visual slot rendered with the video token, so it shares the same
    OneVision visual path as video; the recipe image media handler encodes it as one
    frame.
    """

    def __init__(
        self,
        processor: Any,
        *,
        role_map: dict[str, str] | None = None,
        video_placeholder: str = VIDEO_PLACEHOLDER,
        image_placeholder: str = IMAGE_PLACEHOLDER,
    ) -> None:
        """Store the processor chat template and raw role/media conventions."""
        self.processor = processor
        self.role_map = dict(role_map or ROLE_MAP)
        self.video_placeholder = video_placeholder
        self.image_placeholder = image_placeholder

    def normalize(self, row: Mapping[str, Any]) -> tuple[list[MLLMSegment], list[MLLMMediaSlot], dict[str, Any]]:
        """Return source-only prompt segments, one video segment, and assistant labels."""
        raw = dict(row)
        media_slot = self._normalize_video_slot(raw)
        prompt_messages, target_messages = self._render_prompt_and_target(raw)

        source_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.processor.apply_chat_template(
            target_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not full_text.startswith(source_text):
            raise ValueError("processor chat template does not preserve prompt prefix for label masking.")

        video_cursor = [0]
        segments: list[MLLMSegment] = []
        self._append_rendered_segments(segments, source_text, loss=False, media_slot=media_slot, cursor=video_cursor)
        self._append_rendered_segments(
            segments,
            full_text[len(source_text) :],
            loss=True,
            media_slot=media_slot,
            cursor=video_cursor,
        )
        if video_cursor[0] != 1:
            raise ValueError("rendered chat template must contain exactly one video token.")
        return segments, [media_slot], {}

    def build_prompt_messages(self, row: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Return prompt messages for eval-time generation."""
        prompt_messages, _ = self._render_prompt_and_target(dict(row))
        return prompt_messages

    def _render_prompt_and_target(
        self,
        row: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Render raw messages into Qwen chat blocks around the last assistant turn."""
        messages = row.get("messages") or row.get("conversations")
        if not isinstance(messages, list) or not messages:
            raise ValueError("sample has no `messages`/`conversations`.")

        rendered_messages: list[dict[str, Any]] = []
        total_video_slots = 0
        video_slot_index: int | None = None
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError("contains invalid message.")
            normalized = self._normalize_message(message)
            blocks, video_count = self._to_chat_blocks(normalized["content"])
            if video_count:
                video_slot_index = index
            total_video_slots += video_count
            rendered_messages.append({"role": normalized["role"], "content": blocks})

        if total_video_slots == 0:
            first_user = next((idx for idx, message in enumerate(rendered_messages) if message["role"] == "user"), None)
            if first_user is None:
                raise ValueError("sample has no user message to host the video.")
            rendered_messages[first_user]["content"].insert(0, {"type": "video"})
            total_video_slots = 1
            video_slot_index = first_user

        if total_video_slots != 1:
            raise ValueError(f"video MLLM supports exactly one video per sample, got {total_video_slots}.")

        last_assistant = max(
            (index for index, message in enumerate(rendered_messages) if message["role"] == "assistant"),
            default=None,
        )
        if last_assistant is None:
            raise ValueError("sample has no assistant turn to supervise.")
        if video_slot_index is None:
            raise ValueError("sample has no video slot.")
        if video_slot_index >= last_assistant or rendered_messages[video_slot_index]["role"] == "assistant":
            raise ValueError("the video placeholder must be in a user/system turn before the assistant target.")

        return rendered_messages[:last_assistant], rendered_messages[: last_assistant + 1]

    def _append_rendered_segments(
        self,
        segments: list[MLLMSegment],
        text: str,
        *,
        loss: bool,
        media_slot: MLLMMediaSlot,
        cursor: list[int],
    ) -> None:
        """Split rendered text around the model video token and append DataKit segments."""
        video_token = self._video_token()
        parts = text.split(video_token)
        for index, part in enumerate(parts):
            if part:
                segments.append(MLLMSegment(type="text", loss=loss, value=part))
            if index >= len(parts) - 1:
                continue
            if cursor[0] >= 1:
                raise ValueError("rendered chat template contains more than one video token.")
            segments.append(MLLMSegment(type=media_slot.media_type, loss=False, value=media_slot.media_id))
            cursor[0] += 1

    def _video_token(self) -> str:
        """Return the processor video token used by its chat template."""
        video_token = getattr(self.processor, "video_token", DEFAULT_VIDEO_TOKEN)
        if not isinstance(video_token, str) or not video_token:
            raise ValueError("processor must expose a valid video token.")
        return video_token

    def _normalize_video_slot(self, row: Mapping[str, Any]) -> MLLMMediaSlot:
        """Bind the single visual reference to the raw source field that stores it.

        Image fields (``image`` / ``images``) bind as a ``video`` slot too so a still
        image shares the OneVision visual path; the image media handler encodes it as
        one frame.
        """
        if isinstance(row.get("video"), str):
            return MLLMMediaSlot(media_id="video:0", media_type="video", field="video")

        for key in ("videos", "images_source", "image", "images"):
            value = row.get(key)
            if isinstance(value, (str, dict)):
                return MLLMMediaSlot(media_id="video:0", media_type="video", field=key)
            if isinstance(value, (list, tuple)) and value:
                return MLLMMediaSlot(media_id="video:0", media_type="video", field=key, index=0)

        raise ValueError("video MLLM sample requires `video`, `videos`, `images_source`, `image`, or `images`.")

    def _normalize_message(self, message: Mapping[str, Any]) -> dict[str, str]:
        """Normalize source role/content aliases to Qwen chat role/content strings."""
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

    def _to_chat_blocks(self, content: str) -> tuple[list[dict[str, Any]], int]:
        """Split raw text on the source visual placeholder into HF chat blocks.

        An ``<image>`` placeholder is treated as the single visual slot (rendered as a
        video block) so image and video rows share one normalization path.
        """
        blocks: list[dict[str, Any]] = []
        video_count = 0
        content = content.replace(self.image_placeholder, self.video_placeholder)
        parts = content.split(self.video_placeholder)
        for index, part in enumerate(parts):
            if part:
                blocks.append({"type": "text", "text": part})
            if index < len(parts) - 1:
                blocks.append({"type": "video"})
                video_count += 1
        return blocks, video_count


def build_video_generation_input_ids(
    sample: Mapping[str, Any],
    *,
    processor: Any,
    video_token_count: int,
) -> torch.Tensor:
    """Render one eval prompt with the same video-token expansion as training."""
    if video_token_count < 1:
        raise ValueError("video_token_count must be positive.")

    schema_handler = VideoChatSchemaHandler(processor)
    prompt_messages = schema_handler.build_prompt_messages(sample)
    video_token = schema_handler._video_token()
    text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    if text.count(video_token) != 1:
        raise ValueError("generation prompt must contain exactly one video token.")
    text = text.replace(video_token, video_token * int(video_token_count))
    input_ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
    return torch.tensor(input_ids, dtype=torch.long)
