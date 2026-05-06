"""Miscellaneous helpers for the OpenBee recipe."""

from __future__ import annotations

import torch
import torch.distributed as dist
from mvp_dataset import TorchLoader

from mvp_engine.utils.log import logger

from ..configs.schema import OpenbeeConfig
from ..dataset.collator import OpenbeeCollator
from ..dataset.dataset import build_dataset


def resolve_batching_config(config: OpenbeeConfig, *, data_parallel_world_size: int) -> None:
    """Resolve OpenBee global batch size into micro batch size or accumulation."""
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


def infer_total_steps(
    config: OpenbeeConfig,
    *,
    processor,
    device: torch.device,
    data_parallel_world_size: int,
) -> int:
    """Infer total optimization steps from one finite OpenBee data pass."""
    inference_config = config.model_copy(deep=True)

    collate_fn = OpenbeeCollator(
        pad_token_id=int(processor.tokenizer.pad_token_id),
        processor=processor,
    )
    inference_dataset = build_dataset(
        inference_config,
        processor=processor,
        resample=False,
        resolve_refs=False,
    )
    inference_loader = TorchLoader(
        inference_dataset,
        num_workers=int(inference_config.data.num_workers),
        pin_memory=device.type in ["cuda", "npu"],
        persistent_workers=False,
        drop_last=False,
        multiprocessing_context="spawn",
    ).batch(
        batch_size=int(inference_config.data.batch_size),
        drop_last=True,
        collate_fn=collate_fn,
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
        raise RuntimeError("OpenBee step inference found no packed training samples.")

    samples_per_optimization_step = (
        int(data_parallel_world_size) * int(config.data.batch_size) * int(config.optim.gradient_accumulation_steps)
    )
    if samples_per_optimization_step <= 0:
        raise RuntimeError("OpenBee step inference cannot infer steps with non-positive samples per optimization step.")

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
