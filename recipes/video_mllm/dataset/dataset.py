"""Dataset pipeline for the video MLLM recipe."""

from __future__ import annotations

from functools import partial
from typing import Any

from mvp_dataset import Dataset
from mvp_dataset.core import RuntimeContext

from mvp_engine.kit.mllm.data.guard import DataGuard

from .codec import CodecPatchConfig
from .preprocess import process_sample
from .video_encoding import DenseVideoConfig, KeyframeLowresVideoConfig


def _drop_empty_samples(assemble_context: Any = None) -> DataGuard:
    """Assembler factory that drops empty sentinels left by failed preprocessing."""
    return DataGuard(check_basic_formats=False, check_input_ids=True, check_image_sizes=False)


def build_dataset(config: Any, *, processor: Any):
    """Build the training dataset pipeline for the recipe.

    Loads chat rows from jsonl/parquet (each row references a video path and a
    conversation with one ``<video>`` placeholder) and maps them to model inputs.
    The source kind is taken from ``config.data.source``.

    Args:
        config: Recipe config with dataset, runtime, and seed settings.
        processor: Hugging Face processor used during sample processing.

    Returns:
        An ``mvp_dataset.Dataset`` pipeline of processed video samples.
    """
    dataset_path = config.data.train_path
    if dataset_path is None:
        raise ValueError("Missing `data.train_path` for the video MLLM recipe.")

    context = RuntimeContext.from_runtime(seed=int(config.seed))
    dataset = Dataset.from_source(str(config.data.source), dataset_path, context=context, resample=True)

    # Build strategy-local geometry once and thread it into process_sample.
    dense_config = None
    codec_config = None
    keyframe_config = None
    if config.data.video_encoding_strategy == "uniform":
        dense_config = DenseVideoConfig(
            num_frames=int(config.data.num_frames),
            frame_size=int(config.data.video_frame_size),
            patch_size=int(getattr(processor, "onevision_patch_size", 14)),
        )
    elif config.data.uses_keyframe_lowres:
        keyframe_config = KeyframeLowresVideoConfig(
            num_frames=int(config.data.num_frames),
            full_frame_size=int(config.data.video_frame_size),
            lowres_frame_size=int(config.data.keyframe_lowres_frame_size),
            patch_size=int(getattr(processor, "onevision_patch_size", 14)),
            keyframe_interval=int(config.data.keyframe_interval),
        )
    elif config.data.uses_codec_patches:
        codec_config = CodecPatchConfig(
            num_frames=int(config.data.codec_num_frames),
            packed_frames=int(config.data.codec_packed_frames),
            frame_size=int(config.data.codec_frame_size),
            patch_size=int(config.data.codec_patch_size),
            k_keep=int(config.data.codec_k_keep),
            cv_reader_required=bool(config.data.cv_reader_required),
        )

    dataset = dataset.map(
        partial(
            process_sample,
            processor=processor,
            max_length=int(config.data.max_seq_len),
            video_root=config.data.video_root,
            video_encoding_strategy=config.data.video_encoding_strategy,
            dense_config=dense_config,
            codec_config=codec_config,
            keyframe_config=keyframe_config,
        )
    )
    # Drop rows that failed preprocessing (over-length or malformed): process_sample
    # returns an empty sentinel on error, and this guard filters it out instead of
    # letting the exception crash the data worker.
    dataset = dataset.assemble(_drop_empty_samples)
    return dataset
