"""Dataset processing utilities for the Qwen3 LM recipe."""

from __future__ import annotations

import json
import random
from functools import partial
from pathlib import Path
from typing import Any

import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from ..configs.schema import Qwen3LMConfig
from ..guards.data import DataGuard, build_dataguard
from .packing import PackedSampleAssembler, build_packed_sample_assembler
from .preprocess import process_sample


def _get_distributed_worker() -> tuple[int, int]:
    """Return the distributed rank and world size for JSONL sharding."""
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())
    return 0, 1


def build_jsonl_worker_order(
    row_count: int,
    *,
    seed: int,
    round_index: int,
    worker_id: int,
    num_workers: int,
    rank: int,
    world_size: int,
) -> list[int]:
    """Build the shared shuffled row order for one rank/worker shard."""
    if row_count <= 0:
        return []
    global_worker_count = max(int(num_workers), 1) * max(int(world_size), 1)
    global_worker_id = int(rank) * max(int(num_workers), 1) + int(worker_id)

    order = list(range(row_count))
    random.Random(int(seed) + int(round_index)).shuffle(order)
    if row_count < global_worker_count:
        return [order[global_worker_id % row_count]]
    return order[global_worker_id::global_worker_count]


class JsonlTextDataset(IterableDataset):
    """Local JSONL text dataset with optional infinite resampling."""

    def __init__(
        self,
        config: Qwen3LMConfig,
        *,
        tokenizer: Any,
        resample: bool,
    ) -> None:
        """Load JSONL rows and configure the local iterator pipeline."""
        super().__init__()
        dataset_path = Path(config.data.train_path).expanduser()
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Qwen3 LM JSONL dataset does not exist: {dataset_path}")

        self.rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"Qwen3 LM JSONL dataset is empty: {dataset_path}")

        self.config = config
        self.tokenizer = tokenizer
        self.resample = resample

    def __iter__(self):
        """Yield processed samples from a worker-sharded JSONL stream."""
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        rank, world_size = _get_distributed_worker()
        seed = int(self.config.seed)
        round_index = 0

        while True:
            order = build_jsonl_worker_order(
                len(self.rows),
                seed=seed,
                round_index=round_index,
                worker_id=worker_id,
                num_workers=num_workers,
                rank=rank,
                world_size=world_size,
            )

            guard = DataGuard(check_basic_formats=True, check_input_ids=False, verbose=True)
            input_guard = DataGuard(check_basic_formats=False, check_input_ids=True, verbose=False)
            packer = None
            if self.config.data.packing:
                packer = PackedSampleAssembler(
                    max_length=int(self.config.data.max_seq_len),
                    selection_strategy=self.config.data.packing_selection_strategy,
                    open_pack_limit=int(self.config.data.packing_open_pack_limit),
                    pack_buffer_size=int(self.config.data.packing_buffer_size),
                    seed=seed + round_index * max(num_workers * world_size, 1) + rank * num_workers + worker_id,
                )

            for row_index in order:
                for guarded_sample in guard.push(dict(self.rows[row_index])):
                    sample = process_sample(
                        guarded_sample,
                        tokenizer=self.tokenizer,
                        max_length=int(self.config.data.max_seq_len),
                        thinking_mode=self.config.data.thinking_mode,
                    )
                    for valid_sample in input_guard.push(sample):
                        if packer is None:
                            yield valid_sample
                        else:
                            yield from packer.push(valid_sample)

            if packer is not None:
                yield from packer.finish(drop_last=False)

            if not self.resample:
                break
            round_index += 1


def build_dataset(
    config: Qwen3LMConfig,
    *,
    tokenizer: Any,
    process_fn: Any = process_sample,
    resample: bool = True,
) -> Any:
    """Build the training dataset pipeline for the recipe."""
    dataset_path = config.data.train_path
    if dataset_path is None:
        raise ValueError("Missing `data.train_path` for the Qwen3 LM recipe.")

    if config.data.source_type == "jsonl":
        return JsonlTextDataset(config, tokenizer=tokenizer, resample=resample)

    try:
        from mvp_dataset import Dataset
        from mvp_dataset.core import RuntimeContext
    except ImportError as exc:
        raise ImportError(
            "`data.source_type=mvp_dataset` requires mvp_dataset. "
            "Install it from https://github.com/mvp-ai-lab/mvp-dataset."
        ) from exc

    context = RuntimeContext.from_runtime(seed=int(config.seed))
    source_kwargs: dict[str, Any] = {
        "context": context,
        "resample": resample,
    }
    if config.data.source_format == "lance":
        source_kwargs["shuffle_mode"] = "fragment_aware"
    dataset = Dataset.from_source(
        config.data.source_format,
        dataset_path,
        **source_kwargs,
    )

    dataset = dataset.assemble(
        partial(
            build_dataguard,
            check_basic_formats=True,
            check_input_ids=False,
        )
    )
    dataset = dataset.map(
        partial(
            process_fn,
            tokenizer=tokenizer,
            max_length=int(config.data.max_seq_len),
            thinking_mode=config.data.thinking_mode,
        )
    )
    dataset = dataset.assemble(
        partial(
            build_dataguard,
            check_basic_formats=False,
            check_input_ids=True,
            record=False,
        )
    )

    if config.data.packing:
        dataset = dataset.assemble(
            partial(
                build_packed_sample_assembler,
                max_length=config.data.max_seq_len,
                selection_strategy=config.data.packing_selection_strategy,
                open_pack_limit=config.data.packing_open_pack_limit,
                pack_buffer_size=config.data.packing_buffer_size,
            )
        )

    return dataset
