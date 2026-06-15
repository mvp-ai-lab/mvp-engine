"""Model-family-generic video media processing for MLLM data pipelines.

Covers the parts of video SFT that are identical across Qwen-VL-family recipes:

- rendering a chat row that carries a single ``<video>`` placeholder into prompt/
  target message lists split for last-assistant supervision;
- expanding that placeholder into the model's video pad tokens and building
  assistant-only supervised labels (prompt prefix and all video tokens masked).

Frame decoding and model-specific frame->tensor encoding (e.g. OneVision patch
layout, ``patch_positions``, synthetic ``video_grid_thw``) are intentionally NOT
owned here: a recipe produces the visual tensors with its own encoder and passes
only the resulting ``video_token_count`` into :meth:`VideoMediaKit.build_inputs_and_labels`.
Recipe-specific visual-layout metadata stays in the recipe collator.
"""

from __future__ import annotations

from typing import Any

import torch

from .sample import ROLE_MAP

VIDEO_PLACEHOLDER = "<video>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"


class VideoMediaKit:
    """Generic Qwen-VL video chat rendering and video-token-expanded SFT tokenization."""

    def __init__(
        self,
        *,
        role_map: dict[str, str] | None = None,
        video_placeholder: str = VIDEO_PLACEHOLDER,
    ) -> None:
        """Configure raw role aliases and the source video placeholder."""
        self.role_map = dict(role_map or ROLE_MAP)
        self.video_placeholder = video_placeholder

    def render_chat(self, sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Render one raw row into prompt/target chat messages around a single video.

        Both the source and target message lists use HF chat content blocks. The
        target ends at the last assistant turn; the prompt is its prefix, so the
        caller can mask everything before the supervised assistant turn. Exactly one
        ``<video>`` is required and must sit in a user/system turn before that
        assistant turn; caption-style rows that omit it get a video block prepended
        to the first user turn. Raises ``ValueError`` for malformed rows so callers
        can drop them.

        Returns:
            ``(prompt_messages, target_messages)``.
        """
        messages = sample.get("messages") or sample.get("conversations")
        if not messages:
            raise ValueError("sample has no `messages`/`conversations`.")

        rendered_messages: list[dict[str, Any]] = []
        total_video_slots = 0
        video_slot_index: int | None = None
        for index, message in enumerate(messages):
            normalized = self._normalize_message(message)
            blocks, video_count = self._to_chat_blocks(normalized["content"])
            if video_count:
                video_slot_index = index
            total_video_slots += video_count
            rendered_messages.append({"role": normalized["role"], "content": blocks})

        if total_video_slots == 0:
            first_user = next((i for i, m in enumerate(rendered_messages) if m["role"] == "user"), None)
            if first_user is None:
                raise ValueError("sample has no user message to host the video.")
            rendered_messages[first_user]["content"].insert(0, {"type": "video"})
            total_video_slots = 1
            video_slot_index = first_user

        if total_video_slots != 1:
            raise ValueError(f"video MLLM v1 supports exactly one video per sample, got {total_video_slots}.")

        last_assistant = max(
            (index for index, message in enumerate(rendered_messages) if message["role"] == "assistant"),
            default=None,
        )
        if last_assistant is None:
            raise ValueError("sample has no assistant turn to supervise.")
        if video_slot_index >= last_assistant or rendered_messages[video_slot_index]["role"] == "assistant":
            raise ValueError(
                "the <video> placeholder must be in a user/system turn before the supervised assistant turn."
            )

        return rendered_messages[:last_assistant], rendered_messages[: last_assistant + 1]

    def build_inputs_and_labels(
        self,
        *,
        prompt_messages: list[dict[str, Any]],
        target_messages: list[dict[str, Any]],
        processor: Any,
        video_token_count: int,
        max_length: int,
        overlength_hint: str = "",
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand the single video placeholder into pad tokens and build supervised labels.

        The recipe encodes the video itself and passes ``video_token_count`` (the
        number of visual tokens its encoder produced); this method renders the chat,
        replaces the single rendered video pad token with that many pad tokens,
        tokenizes, and masks the prompt prefix plus every video token from the loss.

        Returns:
            ``(input_ids, attention_mask, labels)``.
        """
        if video_token_count < 1:
            raise ValueError("video_token_count must be positive.")

        video_token = getattr(processor, "video_token", DEFAULT_VIDEO_TOKEN)
        if not isinstance(video_token, str) or not video_token:
            raise ValueError("processor must expose a valid video token for preprocessing.")
        expanded_video = video_token * int(video_token_count)

        def _render_and_expand(messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
            if text.count(video_token) != 1:
                raise ValueError("video preprocessing expects exactly one video pad token in the rendered chat.")
            return text.replace(video_token, expanded_video)

        full_text = _render_and_expand(target_messages, add_generation_prompt=False)
        prompt_text = _render_and_expand(prompt_messages, add_generation_prompt=True)
        if not full_text.startswith(prompt_text):
            raise ValueError("processor chat template does not preserve the prompt prefix for label masking.")

        tokenizer = processor.tokenizer
        input_ids = torch.tensor(tokenizer(full_text, add_special_tokens=False)["input_ids"], dtype=torch.long)
        prompt_ids = torch.tensor(tokenizer(prompt_text, add_special_tokens=False)["input_ids"], dtype=torch.long)
        if int(input_ids.shape[0]) > int(max_length):
            raise ValueError(
                f"sequence length {int(input_ids.shape[0])} exceeds max_seq_len {int(max_length)}; {overlength_hint}"
            )

        video_token_id = int(processor.video_token_id)
        video_token_total = int((input_ids == video_token_id).sum().item())
        if video_token_total != int(video_token_count):
            raise ValueError(f"expanded video tokens ({video_token_total}) do not match expected {video_token_count}.")

        max_prefix = min(int(input_ids.shape[0]), int(prompt_ids.shape[0]))
        prefix_length = 0
        while prefix_length < max_prefix and int(input_ids[prefix_length]) == int(prompt_ids[prefix_length]):
            prefix_length += 1

        labels = input_ids.clone()
        labels[:prefix_length] = ignore_index
        labels[input_ids == video_token_id] = ignore_index
        if not torch.any(labels != ignore_index):
            raise ValueError("sample has no supervised assistant tokens after tokenization.")

        return input_ids, torch.ones_like(input_ids), labels

    def build_generation_inputs(
        self,
        *,
        prompt_messages: list[dict[str, Any]],
        processor: Any,
        video_token_count: int,
    ) -> torch.Tensor:
        """Render a prompt-only sequence with the single video placeholder expanded, for generation.

        The eval-time counterpart of :meth:`build_inputs_and_labels`: it shares the exact
        chat rendering + ``<video>``→``video_token_count`` pad-token expansion, so eval prompts
        match training prompts token-for-token (one expansion implementation for train and eval).

        Returns:
            ``input_ids`` for the prompt (with ``add_generation_prompt=True``).
        """
        if video_token_count < 1:
            raise ValueError("video_token_count must be positive.")

        video_token = getattr(processor, "video_token", DEFAULT_VIDEO_TOKEN)
        if not isinstance(video_token, str) or not video_token:
            raise ValueError("processor must expose a valid video token for preprocessing.")

        text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        if text.count(video_token) != 1:
            raise ValueError("video preprocessing expects exactly one video pad token in the rendered chat.")
        text = text.replace(video_token, video_token * int(video_token_count))

        input_ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        return torch.tensor(input_ids, dtype=torch.long)

    def _normalize_message(self, message: dict[str, Any]) -> dict[str, str]:
        """Map a raw conversation message to ``{"role", "content"}`` with canonical roles."""
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
        """Split text on the video placeholder into HF chat content blocks, counting slots."""
        blocks: list[dict[str, Any]] = []
        video_count = 0
        parts = content.split(self.video_placeholder)
        for index, part in enumerate(parts):
            if part:
                blocks.append({"type": "text", "text": part})
            if index < len(parts) - 1:
                blocks.append({"type": "video"})
                video_count += 1
        return blocks, video_count
