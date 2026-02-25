from pathlib import Path
from typing import Callable

from mvp_dataset import Dataset, TorchLoader

from .preprocess import sharegpt_mapping_fn


def build_jsonl_dataloader(
    jsonl_path: str | list[str],
    base_dir: str | None = None,
    batch_size: int = 32,
    num_workers: int = 8,
    shuffle_buffer: int = 1000,
    collate_fn: Callable | None = None,
) -> TorchLoader:
    if isinstance(jsonl_path, str):
        jsonl_path = [jsonl_path]

    base_dir = base_dir or Path(jsonl_path[0]).parent

    # TODO: support more mapping functions for different jsonl formats
    mapping_fn = sharegpt_mapping_fn

    dataset = (
        Dataset.from_source(jsonl_path, resample=True)
        .group_by("image")
        .resolve_refs([("image", base_dir)])  # tar://... URI base dir
        .map(mapping_fn)
        .batch(batch_size)
    )

    dataloader = (
        TorchLoader(dataset, num_workers=num_workers, collate_fn=collate_fn)
        .unbatch()
        .shuffle(buffer_size=shuffle_buffer)
        .batch(batch_size)
    )

    return dataloader
