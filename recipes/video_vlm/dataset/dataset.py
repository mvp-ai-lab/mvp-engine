"""Dataset processing utilities for the Video VLM recipe."""

from __future__ import annotations

import glob
from functools import partial
from pathlib import Path
from typing import Any

from mvp_dataset import Dataset
from mvp_dataset.core import RuntimeContext
from mvp_dataset.utils.url import normalize_paths

from ..configs.schema import VideoVLMConfig
from ..guards.data import build_dataguard
from .codec import CodecPatchConfig
from .packing import build_packed_sample_assembler, finalize_packed_sample_group
from .preprocess import convert_images_to_pixel_values, process_sample


def resolve_dataset_shards(train_path: str) -> list[str]:
    """Resolve raw parquet shard specs into absolute file paths."""
    shard_specs = normalize_paths(train_path)
    shard_paths: list[str] = []
    for shard_spec in shard_specs:
        if any(char in shard_spec for char in "*?["):
            matches = sorted(glob.glob(shard_spec, recursive=True))
            if matches:
                shard_paths.extend(str(Path(match).expanduser().resolve()) for match in matches)
                continue
        shard_paths.append(str(Path(shard_spec).expanduser().resolve()))
    return shard_paths


def build_dataset(
    config: VideoVLMConfig,
    *,
    processor: Any,
    process_fn: Any = process_sample,
    resample: bool = True,
    resolve_refs: bool = True,
) -> Dataset:
    """Build the training dataset pipeline for the recipe.

    Args:
        config: Recipe config with dataset, runtime, and shuffle settings.
        processor: Hugging Face processor used during sample processing.
        process_fn: Function to process individual samples.
        resample: Whether to loop dataset shards indefinitely across rounds.
        resolve_refs: Whether to resolve references in the dataset.
    Returns:
        An ``mvp_dataset.Dataset`` parquet pipeline with processing and sample-level shuffling.
    """
    dataset_path_value = config.data.train_path
    if dataset_path_value is None:
        raise ValueError("Missing `data.train_path` for the Video VLM recipe.")
    dataset_paths = resolve_dataset_shards(dataset_path_value)

    context = RuntimeContext.from_runtime(seed=int(config.seed))

    # 1. Create the data source.
    dataset = Dataset.from_source(
        "parquet",
        dataset_paths,
        context=context,
        resample=resample,
    )

    # 2. Data guard for handling invalid/bad samples.
    dataset = dataset.assemble(
        partial(
            build_dataguard,
            check_basic_formats=True,
            check_input_ids=False,
            check_image_sizes=True,
        )
    )

    # 3. Pre-process the data samples.
    process_kwargs: dict[str, Any] = {
        "processor": processor,
        "max_length": int(config.data.max_seq_len),
        "thinking_mode": config.data.thinking_mode,
        "video_placeholder": config.data.video_placeholder,
        "codec_config": CodecPatchConfig(
            num_frames=int(config.data.codec_num_frames),
            packed_frames=int(config.data.codec_packed_frames),
            frame_size=int(config.data.codec_frame_size),
            patch_size=int(config.data.codec_patch_size),
            k_keep=int(config.data.codec_k_keep),
            hevc_decoder_bin=config.data.hevc_decoder_bin,
            cv_reader_required=bool(config.data.cv_reader_required),
        )
        if config.data.codec_enabled
        else None,
    }
    dataset = dataset.map(partial(process_fn, **process_kwargs))

    # 4. Guard to filter out any bad samples
    dataset = dataset.assemble(
        partial(
            build_dataguard,
            check_basic_formats=False,
            check_input_ids=True,
            check_image_sizes=False,
            record=False,
        )
    )

    # 5. Optionally pack samples into longer sequences before resolving image refs.
    if config.data.packing:
        dataset = dataset.assemble(
            partial(
                build_packed_sample_assembler,
                max_length=config.data.max_seq_len,
                selection_strategy=config.data.packing_selection_strategy,
                open_pack_limit=config.data.packing_open_pack_limit,
                pack_buffer_size=config.data.packing_buffer_size,
                defer_finalize=True,
            )
        )

    # 6. Resolve references after packing so invalid/short samples avoid image IO.
    if resolve_refs:
        codec_config = process_kwargs["codec_config"]
        if hasattr(dataset, "resolve_ref"):
            dataset = dataset.resolve_ref(ref_names=config.data.ref_columns)
        dataset = dataset.map(partial(convert_images_to_pixel_values, processor=processor, codec_config=codec_config))

    # 7. Drop sentinels created by late image decode/materialization failures.
    # In packed mode this also drops empty packed groups before finalization.
    dataset = dataset.assemble(
        partial(
            build_dataguard,
            check_basic_formats=False,
            check_input_ids=True,
            check_image_sizes=False,
            record=False,
        )
    )

    # 8. Materialize deferred packs after references have been resolved.
    if config.data.packing:
        dataset = dataset.map(finalize_packed_sample_group)

    return dataset
