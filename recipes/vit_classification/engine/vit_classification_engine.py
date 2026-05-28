"""Training engine for the ViT image classification recipe."""

from typing import TypedDict

import torch
from torch.utils.data import DataLoader, DistributedSampler

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import get_rank, get_world_size, is_main_process
from mvp_engine.engine import ENGINE_REGISTRY, Engine, TrainStepContext
from mvp_engine.kit import OptimKit
from mvp_engine.utils.log import logger
from mvp_engine.utils.misc import calculate_model_size

from ..configs.schema import ViTClassificationConfig
from ..dataset import build_dataset
from ..dataset.sampler import InfiniteDistributedSampler
from ..model import build_vit_model


class TrainBatch(TypedDict):
    """Normalized batch structure consumed by the ViT classifier."""

    pixel_values: torch.Tensor
    labels: torch.Tensor


@ENGINE_REGISTRY.register()
class ViTClassificationEngine(Engine):
    """Minimal ImageNet classification engine for the ViT recipe template."""

    ConfigClass = ViTClassificationConfig
    config: ViTClassificationConfig

    def __init__(self, config: ViTClassificationConfig):
        """Initialize recipe-local reusable kits."""
        super().__init__(config)
        self.optim_kit = OptimKit()

    def prepare_dataloader(self, workflow: str = "train") -> DataLoader:
        """Build the dataloader for the requested workflow."""
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
        """Build and parallelize the ViT classifier."""
        model = build_vit_model(self.config.model).to(self.device)
        logger.info(f" - Model name: {model.__class__.__name__}")

        parallelized_model = parallelize_model(
            model,
            device_mesh=self.device_mesh,
            backend_kwargs=self.config.parallel.backend_kwargs.model_dump(),
        )

        if is_main_process():
            model_size, trainable_size = calculate_model_size(parallelized_model)
            logger.info(f" - Model size: {model_size / 1e9:.4f} B")
            logger.info(f" - Trainable model size: {trainable_size / 1e9:.4f} B")

        return parallelized_model

    def prepare_optimizer(self) -> torch.optim.Optimizer:
        """Build the AdamW optimizer used by the recipe."""
        return self.optim_kit.build_optimizer(
            self.model,
            optimizer="AdamW",
            lr=float(self.config.optim.lr),
            weight_decay=float(self.config.optim.weight_decay),
        )

    def prepare_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Build the warmup plus cosine learning-rate schedule."""
        warmup_steps = int(self.total_steps * float(self.config.optim.warmup_ratio))
        return self.optim_kit.build_lr_scheduler(
            optimizer=self.optimizer,
            lr_scheduler="cosine",
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

    def train_pre_step(self, ctx: TrainStepContext) -> TrainStepContext:
        """Move a batch from the dataloader onto the current device."""
        data: tuple[torch.Tensor, torch.Tensor] = ctx.data
        pixel_values, labels = data
        ctx.data = {
            "pixel_values": pixel_values.to(self.device, non_blocking=True),
            "labels": labels.to(self.device, non_blocking=True),
        }
        return ctx

    def forward_step(self, ctx: TrainStepContext) -> None:
        """Run the forward pass and collect training metrics."""
        data: TrainBatch = ctx.data
        with torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        ):
            outputs = self.model(pixel_values=data["pixel_values"], labels=data["labels"])

        predictions = outputs.logits.argmax(dim=-1)
        accuracy = (predictions == data["labels"]).float().mean()

        ctx.outputs = {
            "loss": outputs.loss,
            "logs": {
                "train/loss": outputs.loss.item(),
                "train/acc1": accuracy.item(),
            },
        }
