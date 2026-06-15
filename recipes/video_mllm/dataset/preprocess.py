"""Convert raw-video chat samples into OneVision-backed Qwen3-VL training inputs.

The recipe owns video encoding locally, then expands the single video placeholder
to exactly the number of OneVision visual tokens. Generic Qwen-VL chat rendering
and video-token/label construction come from
``mvp_engine.kit.mllm.data.video.VideoMediaKit``; this module keeps only the
recipe-specific path resolution, strategy dispatch, and OneVision encoding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mvp_engine.kit.mllm.data.media import build_empty_sample
from mvp_engine.kit.mllm.data.video import VideoMediaKit
from mvp_engine.utils.log import simple_info

from .codec import CodecPatchConfig, process_video_with_codec
from .video_encoding import (
    DenseVideoConfig,
    KeyframeLowresVideoConfig,
    process_video_with_dense_frames,
    process_video_with_keyframe_lowres,
)

_VIDEO_MEDIA = VideoMediaKit()


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
    prompt_messages, target_messages = _VIDEO_MEDIA.render_chat(sample)

    video_outputs = process_video_with_dense_frames(video_path, processor=processor, config=dense_config)
    input_ids, attention_mask, labels = _VIDEO_MEDIA.build_inputs_and_labels(
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
    prompt_messages, target_messages = _VIDEO_MEDIA.render_chat(sample)

    # 2. Codec-patchify the video (residual-selected patches packed into dense frames).
    codec_outputs = process_video_with_codec(video_path, processor=processor, config=codec_config)
    if codec_outputs.visual_token_count != int(codec_config.k_keep):
        raise ValueError(
            f"codec output contains {codec_outputs.visual_token_count} tokens but k_keep={codec_config.k_keep}."
        )

    input_ids, attention_mask, labels = _VIDEO_MEDIA.build_inputs_and_labels(
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
    prompt_messages, target_messages = _VIDEO_MEDIA.render_chat(sample)

    video_outputs = process_video_with_keyframe_lowres(video_path, processor=processor, config=keyframe_config)
    input_ids, attention_mask, labels = _VIDEO_MEDIA.build_inputs_and_labels(
        prompt_messages=prompt_messages,
        target_messages=target_messages,
        processor=processor,
        video_token_count=video_outputs.visual_token_count,
        max_length=max_length,
        overlength_hint=(
            "reduce data.num_frames, data.video_frame_size, data.keyframe_lowres_frame_size, or max_seq_len."
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
