"""Convert raw-video chat samples into OneVision-backed Qwen3-VL training inputs.

The recipe owns video encoding locally, then expands the single video placeholder
to exactly the number of OneVision visual tokens. Labels supervise the final
assistant turn; the prompt and all video tokens are masked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mvp_engine.kit.mllm.data.media import build_empty_sample
from mvp_engine.utils.log import simple_info

from .codec import CodecPatchConfig, process_video_with_codec
from .video_encoding import (
    DenseVideoConfig,
    KeyframeLowresVideoConfig,
    process_video_with_dense_frames,
    process_video_with_keyframe_lowres,
)

ROLE_MAP = {
    "assistant": "assistant",
    "gpt": "assistant",
    "human": "user",
    "system": "system",
    "tool": "tool",
    "user": "user",
}

VIDEO_PLACEHOLDER = "<video>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"


def _normalize_message(message: dict[str, Any]) -> dict[str, str]:
    """Map a raw conversation message to ``{"role", "content"}`` with canonical roles."""
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


def _to_chat_blocks(content: str) -> tuple[list[dict[str, Any]], int]:
    """Split text on ``<video>`` into HF chat content blocks, counting video slots."""
    blocks: list[dict[str, Any]] = []
    video_count = 0
    parts = content.split(VIDEO_PLACEHOLDER)
    for index, part in enumerate(parts):
        if part:
            blocks.append({"type": "text", "text": part})
        if index < len(parts) - 1:
            blocks.append({"type": "video"})
            video_count += 1
    return blocks, video_count


def _resolve_video_path(sample: dict[str, Any], *, video_root: str | None) -> str:
    """Resolve the single video path from a raw row's ``video``/``videos``/``images_source``."""
    video_path = sample.get("video")
    if video_path is None:
        for key in ("videos", "images_source"):
            value = sample.get(key)
            if isinstance(value, str):
                video_path = value
                break
            if isinstance(value, (list, tuple)) and value:
                video_path = value[0]
                break
    if not isinstance(video_path, str) or not video_path:
        raise ValueError("video MLLM sample requires a video path in 'video', 'videos', or 'images_source'.")
    if video_root is not None and not Path(video_path).is_absolute():
        video_path = str(Path(video_root) / video_path)
    return video_path


def _render_chat_with_single_video(sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render messages into chat blocks and split them for last-assistant supervision.

    Both the uniform and codec paths require exactly one video, placed in a
    user/system turn before the supervised assistant turn. Captioning-style rows
    that omit the explicit ``<video>`` placeholder get a video block prepended to
    the first user turn.

    Returns:
        ``(prompt_messages, target_messages)`` where ``target_messages`` ends at
        the last assistant turn and ``prompt_messages`` is its prefix.
    """
    messages = sample.get("messages") or sample.get("conversations")
    if not messages:
        raise ValueError("sample has no `messages`/`conversations`.")

    rendered_messages: list[dict[str, Any]] = []
    total_video_slots = 0
    video_slot_index: int | None = None
    for index, message in enumerate(messages):
        normalized = _normalize_message(message)
        blocks, video_count = _to_chat_blocks(normalized["content"])
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
        raise ValueError("the <video> placeholder must be in a user/system turn before the supervised assistant turn.")

    return rendered_messages[:last_assistant], rendered_messages[: last_assistant + 1]


def _build_text_tensors_with_expanded_video(
    *,
    prompt_messages: list[dict[str, Any]],
    target_messages: list[dict[str, Any]],
    processor: Any,
    video_token_count: int,
    max_length: int,
    overlength_hint: str,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render chat text, expand one video placeholder, and build supervised labels."""
    if video_token_count < 1:
        raise ValueError("video_token_count must be positive.")

    video_token = getattr(processor, "video_token", DEFAULT_VIDEO_TOKEN)
    if not isinstance(video_token, str) or not video_token:
        raise ValueError("processor must expose a valid video token for preprocessing.")
    expanded_video = video_token * int(video_token_count)

    def _render_and_expand(messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
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
            f"sequence length {int(input_ids.shape[0])} exceeds max_seq_len {int(max_length)}; "
            f"{overlength_hint}"
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


def _build_uniform_sample(
    sample: dict[str, Any],
    *,
    processor: Any,
    max_length: int,
    dense_config: DenseVideoConfig,
    video_root: str | None = None,
    ignore_index: int = -100,
):
    """Convert one raw row into dense OneVision video training inputs."""
    video_path = _resolve_video_path(sample, video_root=video_root)
    prompt_messages, target_messages = _render_chat_with_single_video(sample)

    video_outputs = process_video_with_dense_frames(video_path, processor=processor, config=dense_config)
    input_ids, attention_mask, labels = _build_text_tensors_with_expanded_video(
        prompt_messages=prompt_messages,
        target_messages=target_messages,
        processor=processor,
        video_token_count=video_outputs.visual_token_count,
        max_length=max_length,
        overlength_hint="reduce data.num_frames, data.video_frame_size, or max_seq_len.",
        ignore_index=ignore_index,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        **video_outputs.to_model_inputs(),
    }


def _build_codec_sample(
    sample: dict[str, Any],
    *,
    processor: Any,
    max_length: int,
    codec_config: CodecPatchConfig,
    video_root: str | None = None,
    ignore_index: int = -100,
):
    """Convert one raw row into OneVision codec training inputs.

    Codec patchification emits the same visual-token layout as dense strategies:
    ``patch_values`` carry selected patch pixels, and ``token_positions`` preserve
    the original ``[t, h, w]`` coordinates for OneVision RoPE.
    """
    video_path = _resolve_video_path(sample, video_root=video_root)

    # 1. Render messages into chat blocks and split for last-assistant supervision.
    prompt_messages, target_messages = _render_chat_with_single_video(sample)

    # 2. Codec-patchify the video (residual-selected patches packed into dense frames).
    codec_outputs = process_video_with_codec(video_path, processor=processor, config=codec_config)
    if codec_outputs.visual_token_count != int(codec_config.k_keep):
        raise ValueError(
            f"codec output contains {codec_outputs.visual_token_count} tokens but k_keep={codec_config.k_keep}."
        )

    input_ids, attention_mask, labels = _build_text_tensors_with_expanded_video(
        prompt_messages=prompt_messages,
        target_messages=target_messages,
        processor=processor,
        video_token_count=codec_outputs.visual_token_count,
        max_length=max_length,
        overlength_hint="reduce codec geometry or max_seq_len.",
        ignore_index=ignore_index,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        **codec_outputs.to_model_inputs(),
    }


def _build_keyframe_lowres_sample(
    sample: dict[str, Any],
    *,
    processor: Any,
    max_length: int,
    keyframe_config: KeyframeLowresVideoConfig,
    video_root: str | None = None,
    ignore_index: int = -100,
):
    """Convert one raw row into dense variable-resolution OneVision video inputs."""
    video_path = _resolve_video_path(sample, video_root=video_root)
    prompt_messages, target_messages = _render_chat_with_single_video(sample)

    video_outputs = process_video_with_keyframe_lowres(video_path, processor=processor, config=keyframe_config)
    input_ids, attention_mask, labels = _build_text_tensors_with_expanded_video(
        prompt_messages=prompt_messages,
        target_messages=target_messages,
        processor=processor,
        video_token_count=video_outputs.visual_token_count,
        max_length=max_length,
        overlength_hint=(
            "reduce data.num_frames, data.video_frame_size, data.keyframe_lowres_frame_size, "
            "or max_seq_len."
        ),
        ignore_index=ignore_index,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        **video_outputs.to_model_inputs(),
    }


def process_sample(
    sample: dict[str, Any],
    *,
    processor: Any,
    max_length: int,
    video_root: str | None = None,
    ignore_index: int = -100,
    video_encoding_strategy: str = "uniform",
    dense_config: DenseVideoConfig | None = None,
    codec_config: CodecPatchConfig | None = None,
    keyframe_config: KeyframeLowresVideoConfig | None = None,
):
    """Process one row into training inputs, dropping bad rows instead of crashing.

    Selects the preprocessing path by ``video_encoding_strategy``. Returns an
    empty sentinel (filtered out downstream by ``build_dataset``) when a row is
    malformed or its tokenized length exceeds ``max_length``, so a single bad or
    over-length sample never kills the data worker.
    """
    if video_encoding_strategy == "uniform" and dense_config is None:
        raise ValueError("`dense_config` is required for `video_encoding_strategy=uniform`.")
    if video_encoding_strategy == "codec_patch" and codec_config is None:
        raise ValueError("`codec_config` is required for `video_encoding_strategy=codec_patch`.")
    if video_encoding_strategy == "keyframe_lowres" and keyframe_config is None:
        raise ValueError("`keyframe_config` is required for `video_encoding_strategy=keyframe_lowres`.")
    if video_encoding_strategy not in {"uniform", "codec_patch", "keyframe_lowres"}:
        raise ValueError(f"unsupported video encoding strategy: {video_encoding_strategy!r}")

    try:
        if video_encoding_strategy == "keyframe_lowres":
            return _build_keyframe_lowres_sample(
                sample,
                processor=processor,
                max_length=max_length,
                keyframe_config=keyframe_config,
                video_root=video_root,
                ignore_index=ignore_index,
            )
        if video_encoding_strategy == "codec_patch":
            return _build_codec_sample(
                sample,
                processor=processor,
                max_length=max_length,
                codec_config=codec_config,
                video_root=video_root,
                ignore_index=ignore_index,
            )
        return _build_uniform_sample(
            sample,
            processor=processor,
            max_length=max_length,
            dense_config=dense_config,
            video_root=video_root,
            ignore_index=ignore_index,
        )
    except Exception as exc:
        simple_info(f"video_mllm: dropping sample ({exc})", level="debug")
        return build_empty_sample()
