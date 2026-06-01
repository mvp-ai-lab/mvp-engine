"""Convert raw-video chat samples into Qwen3-VL training inputs.

The recipe selects frames (``sampling.py``) and decodes them (``decoder.py``),
then hands the frames to the Qwen3-VL processor. The processor owns the
video-token expansion (it interleaves per-frame ``<t seconds>`` timestamps and
the right number of ``<|video_pad|>`` tokens from ``video_grid_thw``), so this
module never computes vision-token counts by hand. Labels supervise the final
assistant turn; the prompt and all video tokens are masked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers.video_utils import VideoMetadata

from mvp_engine.kit.mllm.data.media import build_empty_sample
from mvp_engine.utils.log import simple_info

from .codec import CodecPatchConfig, process_video_with_codec
from .decoder import decode_frames, probe_video
from .sampling import sample_frame_indices

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


def _build_sample(
    sample: dict[str, Any],
    *,
    processor: Any,
    num_frames: int,
    max_length: int,
    video_root: str | None = None,
    ignore_index: int = -100,
):
    """Convert one raw row into video training inputs.

    The row must provide ``messages`` (or ``conversations``) containing exactly
    one ``<video>`` placeholder, and a ``video`` path (or single-element
    ``videos`` list) resolved relative to ``video_root``.
    """
    video_path = _resolve_video_path(sample, video_root=video_root)

    # 1. Render messages into chat blocks and split for last-assistant supervision.
    prompt_messages, target_messages = _render_chat_with_single_video(sample)

    # 2. Select + decode frames once, and build the metadata the processor needs for timestamps.
    meta = probe_video(video_path)
    indices = sample_frame_indices(meta, num_frames)
    frames = decode_frames(video_path, indices)
    video_metadata = VideoMetadata(
        total_num_frames=meta.total_num_frames,
        fps=meta.fps,
        width=meta.width,
        height=meta.height,
        duration=meta.duration,
        frames_indices=indices,
    )

    # 3. Let the processor expand video tokens for the full conversation and the prompt prefix.
    full_text = processor.apply_chat_template(target_messages, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    if not full_text.startswith(prompt_text):
        raise ValueError("processor chat template does not preserve the prompt prefix for label masking.")

    processor_kwargs = dict(
        videos=[frames],
        do_sample_frames=False,
        video_metadata=[video_metadata],
        return_tensors="pt",
    )
    full = processor(text=[full_text], **processor_kwargs)
    prompt = processor(text=[prompt_text], **processor_kwargs)

    input_ids = full["input_ids"][0]
    if int(input_ids.shape[0]) > int(max_length):
        raise ValueError(
            f"sequence length {int(input_ids.shape[0])} exceeds max_seq_len {int(max_length)}; "
            "reduce data.num_frames or the video resolution."
        )

    # Mask the prompt by the token-level common prefix, not the standalone prompt
    # length: a string prefix is not a guaranteed token prefix because BPE can
    # merge the assistant-boundary token with the response's first token (e.g. an
    # assistant turn that begins with a newline), which would silently misalign
    # supervision.
    prompt_ids = prompt["input_ids"][0]
    max_prefix = min(int(input_ids.shape[0]), int(prompt_ids.shape[0]))
    prefix_length = 0
    while prefix_length < max_prefix and int(input_ids[prefix_length]) == int(prompt_ids[prefix_length]):
        prefix_length += 1

    labels = input_ids.clone()
    labels[:prefix_length] = ignore_index
    labels[input_ids == processor.video_token_id] = ignore_index
    if not torch.any(labels != ignore_index):
        raise ValueError("sample has no supervised assistant tokens after tokenization.")

    # NOTE: we deliberately do NOT return ``mm_token_type_ids`` here. The Qwen3-VL
    # processor expands one video into ``grid_t`` separate ``<|vision_start|>...<|vision_end|>``
    # spans (with timestamps between), so ``mm_token_type_ids`` reports ``grid_t``
    # contiguous video segments while ``video_grid_thw`` carries only one row
    # ``(grid_t, h, w)``. Qwen3-VL's ``get_rope_index`` pulls one grid row per
    # contiguous video span, so passing ``mm_token_type_ids`` makes the second span
    # raise ``StopIteration`` mid-forward. Omitting it makes
    # ``compute_3d_position_ids`` skip M-RoPE and fall back to default positions —
    # training runs, with the cost of degraded vision positional encoding. Fixing
    # this properly (expanding ``video_grid_thw`` to ``grid_t`` rows of ``(1, h, w)``)
    # is a follow-up.
    return {
        "input_ids": input_ids,
        "attention_mask": full["attention_mask"][0],
        "labels": labels,
        "pixel_values_videos": full["pixel_values_videos"],
        "video_grid_thw": full["video_grid_thw"],
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

    Unlike the uniform path, the Qwen3-VL video processor is bypassed: codec
    patchification owns ``pixel_values_videos``/``video_grid_thw`` and the
    language side is expanded by hand to exactly ``codec_k_keep`` video-pad
    tokens. The OneVision tower uses ``spatial_merge_size == 1``, so the
    language-side video-token count equals ``codec_k_keep == prod(video_grid_thw)``.
    The single ``patch_positions`` tensor is returned for the engine to route
    through the encoder.
    """
    video_path = _resolve_video_path(sample, video_root=video_root)

    # 1. Render messages into chat blocks and split for last-assistant supervision.
    prompt_messages, target_messages = _render_chat_with_single_video(sample)

    # 2. Codec-patchify the video (residual-selected patches packed into dense frames).
    codec_outputs = process_video_with_codec(video_path, processor=processor, config=codec_config)
    patch_positions = codec_outputs["patch_positions"]
    video_grid_thw = codec_outputs["video_grid_thw"]
    expected_tokens = int(video_grid_thw.prod(dim=-1).sum().item())
    if expected_tokens != int(codec_config.k_keep):
        raise ValueError(
            f"codec video_grid_thw implies {expected_tokens} tokens but k_keep={codec_config.k_keep}."
        )

    # 3. Render text with the single <|video_pad|> from the video block, then expand it by hand
    # to k_keep pads. We deliberately do NOT pass videos to the processor: codec owns the pixels.
    video_token = getattr(processor, "video_token", DEFAULT_VIDEO_TOKEN)
    if not isinstance(video_token, str) or not video_token:
        raise ValueError("processor must expose a valid video token for codec preprocessing.")
    expanded_video = video_token * int(codec_config.k_keep)

    def _render_and_expand(messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        if text.count(video_token) != 1:
            raise ValueError("codec preprocessing expects exactly one video pad token in the rendered chat.")
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
            "reduce codec geometry or max_seq_len."
        )

    video_token_id = int(processor.video_token_id)
    video_token_total = int((input_ids == video_token_id).sum().item())
    if video_token_total != int(codec_config.k_keep):
        raise ValueError(
            f"expanded video tokens ({video_token_total}) do not match k_keep ({codec_config.k_keep})."
        )

    # Mask the prompt by the token-level common prefix (BPE can merge the assistant boundary
    # token with the response's first token), then mask all video pads out of the loss.
    max_prefix = min(int(input_ids.shape[0]), int(prompt_ids.shape[0]))
    prefix_length = 0
    while prefix_length < max_prefix and int(input_ids[prefix_length]) == int(prompt_ids[prefix_length]):
        prefix_length += 1

    labels = input_ids.clone()
    labels[:prefix_length] = ignore_index
    labels[input_ids == video_token_id] = ignore_index
    if not torch.any(labels != ignore_index):
        raise ValueError("sample has no supervised assistant tokens after tokenization.")

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "pixel_values_videos": codec_outputs["pixel_values_videos"],
        "video_grid_thw": video_grid_thw,
        "patch_positions": patch_positions,
    }


def process_sample(
    sample: dict[str, Any],
    *,
    processor: Any,
    num_frames: int,
    max_length: int,
    video_root: str | None = None,
    ignore_index: int = -100,
    video_encoding_strategy: str = "uniform",
    codec_config: CodecPatchConfig | None = None,
):
    """Process one row into training inputs, dropping bad rows instead of crashing.

    Selects the preprocessing path by ``video_encoding_strategy``. Returns an
    empty sentinel (filtered out downstream by ``build_dataset``) when a row is
    malformed or its tokenized length exceeds ``max_length``, so a single bad or
    over-length sample never kills the data worker.
    """
    if video_encoding_strategy == "codec_patch" and codec_config is None:
        raise ValueError("`codec_config` is required for `video_encoding_strategy=codec_patch`.")
    if video_encoding_strategy == "keyframe_lowres":
        raise NotImplementedError("`video_encoding_strategy=keyframe_lowres` is not implemented yet.")
    if video_encoding_strategy not in {"uniform", "codec_patch"}:
        raise ValueError(f"unsupported video encoding strategy: {video_encoding_strategy!r}")

    try:
        if video_encoding_strategy == "codec_patch":
            return _build_codec_sample(
                sample,
                processor=processor,
                max_length=max_length,
                codec_config=codec_config,
                video_root=video_root,
                ignore_index=ignore_index,
            )
        return _build_sample(
            sample,
            processor=processor,
            num_frames=num_frames,
            max_length=max_length,
            video_root=video_root,
            ignore_index=ignore_index,
        )
    except Exception as exc:
        simple_info(f"video_mllm: dropping sample ({exc})", level="debug")
        return build_empty_sample()
