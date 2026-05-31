"""Dataset pipeline for the video MLLM recipe."""

from __future__ import annotations

from functools import partial
from typing import Any

from mvp_dataset import Dataset
from mvp_dataset.core import RuntimeContext

from .preprocess import process_sample


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

    dataset = dataset.map(
        partial(
            process_sample,
            processor=processor,
            num_frames=int(config.data.num_frames),
            max_length=int(config.data.max_seq_len),
            video_root=config.data.video_root,
        )
    )
    return dataset
