"""Training engine for the minimal VLM recipe."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import get_rank, get_world_size, is_main_process
from mvp_engine.engine import ENGINE_REGISTRY, Engine
from mvp_engine.utils.log import logger
from mvp_engine.utils.misc import calculate_model_size

from ..dataset import InfiniteDistributedSampler, MinimalVlmCollator, build_dataset
from ..model import build_qwen3_vl_model, build_qwen3_vl_processor
from ..types import TrainBatch


@ENGINE_REGISTRY.register()
class MinimalVlmEngine(Engine):
    """Recipe-local engine for supervised Qwen3-VL fine-tuning."""

    processor: Any | None = None

    def prepare_dataloader(self, workflow: str = "train") -> DataLoader:
        """Build the dataloader that yields multimodal chat batches."""
        dataset = build_dataset(self.config, workflow)
        if self.processor is None:
            self.processor = build_qwen3_vl_processor(self.config.model)

        collate_fn = MinimalVlmCollator(
            processor=self.processor,
            max_length=int(self.config.data.max_seq_len),
        )

        is_train = workflow == "train"
        if is_train:
            sampler = InfiniteDistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=True,
                seed=int(OmegaConf.select(self.config, "project.seed", default=42)),
            )
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=False,
            )

        return DataLoader(
            dataset,
            batch_size=int(self.config.data.batch_size),
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=int(self.config.data.num_workers),
            pin_memory=self.device.type == "cuda",
            drop_last=is_train,
            persistent_workers=int(self.config.data.num_workers) > 0,
        )

    def prepare_model(self) -> torch.nn.Module:
        """Build, freeze, and parallelize the Qwen3-VL model."""
        model = build_qwen3_vl_model(self.config.model).to(self.device)
        logger.info(f" - Model name: {model.__class__.__name__}")

        parallelized_model = parallelize_model(
            model,
            device_mesh=self.device_mesh,
            backend_kwargs=self.config.parallel.get("backend_kwargs", {}),
        )

        if is_main_process():
            model_size, trainable_size = calculate_model_size(parallelized_model)
            logger.info(f" - Model size: {model_size / 1e9:.4f} B")
            logger.info(f" - Trainable model size: {trainable_size / 1e9:.4f} B")

        return parallelized_model

    def prepare_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW over trainable parameters only."""
        trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise ValueError("No trainable parameters found for the minimal VLM recipe.")

        return torch.optim.AdamW(
            trainable_parameters,
            lr=float(self.config.optim.lr),
            weight_decay=float(self.config.optim.weight_decay),
        )

    def prepare_scheduler(self) -> SequentialLR | CosineAnnealingLR:
        """Build the warmup plus cosine decay learning-rate schedule."""
        warmup_steps = int(self.total_steps * float(self.config.optim.warmup_ratio))
        if warmup_steps <= 0:
            return CosineAnnealingLR(self.optimizer, T_max=self.total_steps)

        scheduler_warmup = LinearLR(
            self.optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        scheduler_main = CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.total_steps - warmup_steps, 1),
        )
        return SequentialLR(
            self.optimizer,
            [scheduler_warmup, scheduler_main],
            milestones=[warmup_steps],
        )

    def train_pre_step(self, data: dict[str, Any]) -> TrainBatch:
        """Move the collated batch onto the current device."""
        batch: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=True)
            else:
                batch[key] = value

        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
            "pixel_values": batch.get("pixel_values"),
            "image_grid_thw": batch.get("image_grid_thw"),
        }

    def train_one_step(self, data: TrainBatch) -> dict[str, Any]:
        """Run one Qwen3-VL forward pass and collect training metrics."""
        with torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        ):
            outputs = self.model(**data)

        supervised_tokens = (data["labels"] != -100).sum()

        return {
            "loss": outputs.loss,
            "logs": {
                "train/loss": outputs.loss.item(),
                "train/supervised_tokens": float(supervised_tokens.item()),
            },
        }
