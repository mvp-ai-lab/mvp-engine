"""JSONL dataset processing utilities for the minimal VLM recipe."""

from __future__ import annotations

from functools import partial
from typing import Any

from mvp_dataset import Dataset
from mvp_dataset.core import RuntimeContext

from .preprocess import convert_images_to_pixel_values, process_sample


def build_dataset(config: Any, *, processor: Any):
    """Build the training dataset pipeline for the recipe.

    Args:
        config: Recipe config with dataset, runtime, and shuffle settings.
        processor: Hugging Face processor used during sample processing.

    Returns:
        An ``mvp_dataset.Dataset`` pipeline with JSONL loading, processing, and
        sample-level shuffling.
    """
    dataset_path = config.data.train_path
    if dataset_path is None:
        raise ValueError("Missing `data.train_path` for the minimal VLM recipe.")

    context = RuntimeContext.from_runtime(seed=int(config.seed))

    # 1. Create the data source.
    dataset = Dataset.from_source(
        "parquet",
        dataset_path,
        context=context,
        resample=True,
    )

    # 3. Pre-process the data samples.
    process_kwargs: dict[str, Any] = {
        "processor": processor,
        "max_length": int(config.data.max_seq_len),
    }
    dataset = dataset.map(partial(process_sample, **process_kwargs))
    dataset = dataset.map(partial(convert_images_to_pixel_values, processor=processor))

    return dataset
