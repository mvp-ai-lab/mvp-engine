"""Training engine for the Magic Transformer fake-data recipe."""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import get_rank, get_world_size
from mvp_engine.engine import ENGINE_REGISTRY, Engine, TrainStepContext
from mvp_engine.utils.log import logger

from ..configs.schema import MagicTransformerConfig
from ..dataset import InfiniteDistributedSampler, build_dataset
from ..model import build_magic_transformer_model


class TrainBatch(TypedDict):
    """Normalized batch structure consumed by the Magic Transformer."""

    input_ids: torch.Tensor
    labels: torch.Tensor


@ENGINE_REGISTRY.register()
class MagicTransformerEngine(Engine):
    """Minimal next-token training engine backed by fake token data."""

    ConfigClass = MagicTransformerConfig
    config: MagicTransformerConfig

    def prepare_dataloader(self, workflow: str = "train") -> DataLoader:
        """Build the requested split with distributed-aware sampling."""
        dataset = build_dataset(self.config, workflow)
        is_train = workflow == "train"

        if is_train:
            sampler = InfiniteDistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=True,
                seed=self.config.seed,
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
            num_workers=int(self.config.data.num_workers),
            pin_memory=self.device.type == "cuda",
            drop_last=is_train,
            persistent_workers=int(self.config.data.num_workers) > 0,
        )

    def prepare_model(self) -> torch.nn.Module:
        """Build the recipe model and adapt it to the configured mesh."""
        model = build_magic_transformer_model(self.config.model).to(self.device)
        logger.info(f" - Model name: {model.__class__.__name__}")

        parallelized_model = parallelize_model(
            model,
            device_mesh=self.device_mesh,
            backend_kwargs=self.config.parallel.backend_kwargs.model_dump(),
        )

        return parallelized_model

    def prepare_optimizer(self) -> torch.optim.Optimizer:
        """Build the AdamW optimizer for all trainable parameters."""
        model_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(
            model_parameters,
            lr=float(self.config.optim.lr),
            weight_decay=float(self.config.optim.weight_decay),
        )

    def prepare_scheduler(self) -> SequentialLR | CosineAnnealingLR:
        """Build the warmup-plus-cosine learning-rate schedule."""
        warmup_steps = int(self.total_steps * float(self.config.optim.warmup_ratio))
        if warmup_steps <= 0:
            return CosineAnnealingLR(self.optimizer, T_max=max(self.total_steps, 1))

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
        return SequentialLR(self.optimizer, [scheduler_warmup, scheduler_main], milestones=[warmup_steps])

    def train_pre_step(self, ctx: TrainStepContext) -> TrainStepContext:
        """Move the fake token batch to the active device."""
        data: dict[str, torch.Tensor] = ctx.data
        ctx.data = {
            "input_ids": data["input_ids"].to(self.device, non_blocking=True),
            "labels": data["labels"].to(self.device, non_blocking=True),
        }
        return ctx

    def forward_step(self, ctx: TrainStepContext) -> None:
        """Run one next-token prediction step and collect metrics."""
        data: TrainBatch = ctx.data
        self._last_batch_size = int(data["input_ids"].shape[0])
        self._last_seq_len = int(data["input_ids"].shape[1])

        with torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        ):
            logits = self.model(input_ids=data["input_ids"])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                data["labels"].reshape(-1),
            )

        predictions = logits.argmax(dim=-1)
        token_accuracy = (predictions == data["labels"]).float().mean()

        ctx.outputs = {
            "loss": loss,
            "logs": {
                "train/loss": float(loss.item()),
                "train/token_acc": float(token_accuracy.item()),
            },
        }
