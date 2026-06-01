"""Small helpers for the qwen2_5_vl recipe."""

from __future__ import annotations

import torch
import torch.distributed as dist

from mvp_engine.kit.mllm import MLLMDataKit, PackingOptions
from mvp_engine.utils.log import logger

from ..configs.schema import Qwen2_5VLConfig


def infer_total_steps(
    config: Qwen2_5VLConfig,
    *,
    processor,
    device: torch.device,
    data_parallel_world_size: int,
    data_parallel_group: dist.ProcessGroup | None = None,
) -> int:
    """Infer total optimization steps from one finite packed data pass."""
    inference_config = config.model_copy(deep=True)
    data_kit = MLLMDataKit()
    packing = PackingOptions(
        selection_strategy=inference_config.data.packing_selection_strategy,
        open_pack_limit=int(inference_config.data.packing_open_pack_limit),
        buffer_size=int(inference_config.data.packing_buffer_size),
    )
    dataset = data_kit.build_dataset(
        dataset_path=inference_config.data.train_path,
        processor=processor,
        max_seq_len=int(inference_config.data.max_seq_len),
        resample=False,
        resolve_refs=False,
        ref_columns=inference_config.data.ref_columns,
        seed=int(inference_config.seed),
        packing=packing,
        thinking_mode=inference_config.data.thinking_mode,
    )
    loader = data_kit.build_dataloader(
        dataset,
        batch_size=int(inference_config.data.batch_size),
        num_workers=int(inference_config.data.num_workers),
        pin_memory=device.type in ["cuda", "npu"],
        collate_fn=data_kit.build_collator(
            pad_token_id=int(processor.tokenizer.pad_token_id),
            processor=processor,
        ),
        drop_last=False,
    )

    local_packed_samples = 0
    for local_batches, batch in enumerate(loader, start=1):
        local_packed_samples += int(batch["input_ids"].shape[0])
        if local_batches % 100 == 0:
            logger.info(f"Step inference progress: {local_batches} local batches...")

    packed_samples = torch.tensor(local_packed_samples, device=device, dtype=torch.long)
    if dist.is_available() and dist.is_initialized() and data_parallel_world_size > 1:
        dist.all_reduce(packed_samples, op=dist.ReduceOp.SUM, group=data_parallel_group)

    global_packed_samples = int(packed_samples.item())
    if global_packed_samples <= 0:
        raise RuntimeError("Step inference found no packed training samples.")

    samples_per_step = (
        int(data_parallel_world_size) * int(config.data.batch_size) * int(config.optim.gradient_accumulation_steps)
    )
    if samples_per_step <= 0:
        raise RuntimeError("Step inference cannot use non-positive samples per optimization step.")

    total_steps = (global_packed_samples + samples_per_step - 1) // samples_per_step
    logger.info(
        f"Step inference: packed_samples={global_packed_samples}, dp_world_size={data_parallel_world_size}, "
        f"samples_per_optimization_step={samples_per_step}, inferred_total_steps={total_steps}"
    )
    return int(total_steps)

