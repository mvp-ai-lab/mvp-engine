"""Dataset processing utilities for the Basic VLM recipe."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

from mvp_dataset import Dataset
from mvp_dataset.core import RuntimeContext

from ..configs.schema import BasicVLMConfig
from ..guards.data import build_dataguard
from .packing import build_packed_sample_assembler, finalize_packed_sample_group
from .preprocess import convert_images_to_pixel_values, process_sample
from .skip import SkipMode, build_skip_by_worker, build_skip_recorder


def build_dataset(
    config: BasicVLMConfig,
    *,
    processor: Any,
    process_fn: Any = process_sample,
    resample: bool = True,
    resolve_refs: bool = True,
    skip_mode: SkipMode = "off",
    skip_counts: Mapping[int, int] | None = None,
) -> Dataset:
    """Build the training dataset pipeline for the recipe.

    Args:
        config: Recipe config with dataset, runtime, and shuffle settings.
        processor: Hugging Face processor used during sample processing.
        process_fn: Function to process individual samples.
        resample: Whether to loop dataset shards indefinitely across rounds.
        resolve_refs: Whether to resolve references in the dataset.
        skip_mode: Optional post-pack fast-resume mode.
        skip_counts: Worker-slot skip counts used by ``skip_mode="perform"``.
    Returns:
        An ``mvp_dataset.Dataset`` pipeline with parquet loading, processing, and
        sample-level shuffling.
    """
    dataset_path = config.data.train_path
    if dataset_path is None:
        raise ValueError("Missing `data.train_path` for the Basic VLM recipe.")

    context = RuntimeContext.from_runtime(seed=int(config.seed))

    # 1. Create the data source.
    dataset = Dataset.from_source(
        "lance",
        dataset_path,
        context=context,
        resample=resample,
        shuffle_mode="fragment_aware",
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

    # 6. Optional post-pack resume skip boundary. The recorder path returns here
    # so the lightweight resume pass never resolves image references.
    if skip_mode == "pre_calculate":
        return dataset.assemble(build_skip_recorder)
    elif skip_mode == "perform":
        dataset = dataset.assemble(
            partial(
                build_skip_by_worker, skip_counts={str(slot): int(count) for slot, count in (skip_counts or {}).items()}
            )
        )
    elif skip_mode != "off":
        raise ValueError(f"Invalid skip_mode: {skip_mode}. Must be one of 'off', 'pre_calculate', or 'perform'.")

    # 7. Resolve references after packing so invalid/short samples avoid image IO.
    if resolve_refs:
        # TODO: add error handeling inside the mvp-dataset
        # TODO: what about pure text data?
        dataset = dataset.resolve_ref(ref_names=config.data.ref_columns).map(
            partial(convert_images_to_pixel_values, processor=processor)
        )

    # 8. Drop sentinels created by late image decode/materialization failures.
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

    # 9. Materialize deferred packs after references have been resolved.
    if config.data.packing:
        dataset = dataset.map(finalize_packed_sample_group)

    return dataset
