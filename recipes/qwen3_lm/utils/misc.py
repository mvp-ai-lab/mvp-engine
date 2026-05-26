"""Miscellaneous helpers for the Qwen3 LM recipe."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from mvp_engine.utils.log import logger

from ..configs.schema import Qwen3LMConfig
from ..dataset.collator import Qwen3LMCollator
from ..dataset.dataset import build_dataset


def resolve_batching_config(config: Qwen3LMConfig, *, data_parallel_world_size: int) -> None:
    """Resolve Qwen3 LM global batch size into micro batch size or accumulation."""
    target_global_batch_size = config.optim.global_batch_size
    accumulation_steps = int(config.optim.gradient_accumulation_steps)
    micro_batch_size = int(config.data.batch_size)

    if target_global_batch_size is None:
        if accumulation_steps == -1:
            raise ValueError("`optim.gradient_accumulation_steps=-1` requires `optim.global_batch_size`.")
        if micro_batch_size == -1:
            raise ValueError("`data.batch_size=-1` requires `optim.global_batch_size`.")
        return

    if micro_batch_size == -1 and accumulation_steps == -1:
        raise ValueError(
            "`optim.global_batch_size` cannot infer both `data.batch_size` and "
            "`optim.gradient_accumulation_steps` at the same time."
        )

    if micro_batch_size == -1:
        batch_size_divisor = int(data_parallel_world_size) * accumulation_steps
        if batch_size_divisor <= 0 or target_global_batch_size % batch_size_divisor != 0:
            raise ValueError(
                "`data.batch_size` cannot be inferred exactly: "
                "`optim.global_batch_size` must be divisible by "
                "`data_parallel_world_size * optim.gradient_accumulation_steps`."
            )
        config.data.batch_size = target_global_batch_size // batch_size_divisor
        micro_batch_size = int(config.data.batch_size)
    elif accumulation_steps == -1:
        accumulation_divisor = int(data_parallel_world_size) * micro_batch_size
        if accumulation_divisor <= 0 or target_global_batch_size % accumulation_divisor != 0:
            raise ValueError(
                "`optim.gradient_accumulation_steps` cannot be inferred exactly: "
                "`optim.global_batch_size` must be divisible by "
                "`data_parallel_world_size * data.batch_size`."
            )
        config.optim.gradient_accumulation_steps = target_global_batch_size // accumulation_divisor
        accumulation_steps = int(config.optim.gradient_accumulation_steps)

    effective_global_batch_size = int(data_parallel_world_size) * micro_batch_size * accumulation_steps
    if effective_global_batch_size != target_global_batch_size:
        raise ValueError(
            "`optim.global_batch_size` does not match the configured batching: "
            f"expected {effective_global_batch_size} from "
            "`data_parallel_world_size * data.batch_size * optim.gradient_accumulation_steps`."
        )


def build_torch_loader(config: Qwen3LMConfig, dataset, *, collate_fn, device: torch.device, finite: bool = False):
    """Build a TorchLoader or PyTorch DataLoader for the configured dataset backend."""
    if config.data.source_type == "mvp_dataset":
        try:
            from mvp_dataset import TorchLoader
        except ImportError as exc:
            raise ImportError(
                "`data.source_type=mvp_dataset` requires mvp_dataset. "
                "Install it from https://github.com/mvp-ai-lab/mvp-dataset."
            ) from exc

        loader = TorchLoader(
            dataset,
            num_workers=int(config.data.num_workers),
            pin_memory=device.type in ["cuda", "npu"],
            persistent_workers=False,
            drop_last=finite,
            multiprocessing_context="spawn",
        )
        return loader.batch(
            batch_size=int(config.data.batch_size),
            drop_last=True,
            collate_fn=collate_fn,
        )

    return DataLoader(
        dataset,
        batch_size=int(config.data.batch_size),
        drop_last=True,
        num_workers=int(config.data.num_workers),
        pin_memory=device.type in ["cuda", "npu"],
        persistent_workers=False,
        collate_fn=collate_fn,
    )


def infer_total_steps(
    config: Qwen3LMConfig,
    *,
    tokenizer,
    device: torch.device,
    data_parallel_world_size: int,
) -> int:
    """Infer total optimization steps from one finite Qwen3 LM data pass."""
    inference_config = config.model_copy(deep=True)
    collate_fn = Qwen3LMCollator(pad_token_id=int(tokenizer.pad_token_id))
    inference_dataset = build_dataset(
        inference_config,
        tokenizer=tokenizer,
        resample=False,
    )
    inference_loader = build_torch_loader(
        inference_config,
        inference_dataset,
        collate_fn=collate_fn,
        device=device,
        finite=True,
    )

    local_packed_samples = 0
    local_batches = 0
    for batch in inference_loader:
        local_packed_samples += int(batch["input_ids"].shape[0])
        local_batches += 1
        if local_batches % 100 == 0:
            logger.info(f"Step inference progress: {local_batches} local batches...")

    global_packed_samples_tensor = torch.tensor(local_packed_samples, device=device, dtype=torch.long)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_packed_samples_tensor, op=dist.ReduceOp.SUM)

    global_packed_samples = int(global_packed_samples_tensor.item())
    if global_packed_samples <= 0:
        raise RuntimeError("Qwen3 LM step inference found no packed training samples.")

    samples_per_optimization_step = (
        int(data_parallel_world_size) * int(config.data.batch_size) * int(config.optim.gradient_accumulation_steps)
    )
    if samples_per_optimization_step <= 0:
        raise RuntimeError(
            "Qwen3 LM step inference cannot infer steps with non-positive samples per optimization step."
        )

    total_steps = (global_packed_samples + samples_per_optimization_step - 1) // samples_per_optimization_step
    if total_steps <= 0:
        raise RuntimeError("Step inference found fewer packed samples than one optimization step requires.")

    logger.info(
        f"Step inference: packed_samples={global_packed_samples}, "
        f"dp_world_size={data_parallel_world_size}, "
        f"samples_per_optimization_step={samples_per_optimization_step}, "
        f"inferred_total_steps={total_steps}"
    )
    return total_steps
