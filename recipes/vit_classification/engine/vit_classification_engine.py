"""Training engine for the ViT image classification recipe."""

from typing import TypedDict

import torch
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import (
    get_rank,
    get_world_size,
    has_dtensor_parameters,
    is_main_process,
)
from mvp_engine.engine import ENGINE_REGISTRY, Engine
from mvp_engine.utils.log import get_logger, logger
from mvp_engine.utils.misc import calculate_model_size

from ..dataset import build_dataset
from ..dataset.sampler import InfiniteDistributedSampler
from ..model import build_vit_model


class TrainBatch(TypedDict):
    """Normalized batch structure consumed by the ViT classifier."""

    pixel_values: torch.Tensor
    labels: torch.Tensor


class TrainStepOutput(TypedDict):
    """Outputs returned by one training step."""

    loss: torch.Tensor
    logs: dict[str, float]


@ENGINE_REGISTRY.register()
class ViTClassificationEngine(Engine):
    """Minimal ImageNet classification engine for the ViT recipe template."""

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
            num_workers=int(self.config.data.num_workers),
            pin_memory=self.device.type == "cuda",
            drop_last=is_train,
            persistent_workers=int(self.config.data.num_workers) > 0,
        )

    def prepare_model(self) -> torch.nn.Module:
        """Build and parallelize the ViT classifier."""
        model = build_vit_model(self.config.model).to(self.device)
        logger.info(f" - Model name: {model.__class__.__name__}")

        parallel_backend = OmegaConf.select(self.config, "parallel.type", default=None)
        if parallel_backend not in ["ddp", "fsdp2"]:
            raise NotImplementedError(f"Parallel type {parallel_backend} not implemented.")

        parallelized_model = parallelize_model(
            model,
            device_mesh=self.device_mesh,
            backend=parallel_backend,
            backend_kwargs=self.config.parallel.get("backend_kwargs", {}),
        )

        if is_main_process():
            model_size, trainable_size = calculate_model_size(parallelized_model)
            logger.info(f" - Model size: {model_size / 1e9:.4f} B")
            logger.info(f" - Trainable model size: {trainable_size / 1e9:.4f} B")

        if self.config.optim.compile:
            parallelized_model = torch.compile(
                parallelized_model,
                backend=OmegaConf.select(self.config, "optim.compile_backend", default="inductor"),
                mode=OmegaConf.select(self.config, "optim.compile_mode", default="default"),
            )

        return parallelized_model

    def prepare_optimizer(self) -> torch.optim.Optimizer:
        """Build the AdamW optimizer used by the recipe."""
        model_parameters = list(self.model.parameters())
        optimizer_kwargs = {
            "lr": float(self.config.optim.lr),
            "weight_decay": float(self.config.optim.weight_decay),
        }
        foreach_cfg = OmegaConf.select(self.config, "optim.foreach", default=None)

        if has_dtensor_parameters(model_parameters):
            if foreach_cfg is not False:
                log = get_logger()
                if log is not None:
                    log.info(
                        " - Detected DTensor parameters. Falling back to AdamW foreach=False "
                        "to avoid mixed Tensor/DTensor foreach kernel errors."
                    )
            optimizer_kwargs["foreach"] = False
        elif foreach_cfg is not None:
            optimizer_kwargs["foreach"] = bool(foreach_cfg)

        return torch.optim.AdamW(model_parameters, **optimizer_kwargs)

    def prepare_scheduler(self) -> SequentialLR | CosineAnnealingLR:
        """Build the warmup plus cosine learning-rate schedule."""
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
        return SequentialLR(self.optimizer, [scheduler_warmup, scheduler_main], milestones=[warmup_steps])

    def train_pre_step(self, data: tuple[torch.Tensor, torch.Tensor]) -> TrainBatch:
        """Move a batch from the dataloader onto the current device."""
        pixel_values, labels = data
        return {
            "pixel_values": pixel_values.to(self.device, non_blocking=True),
            "labels": labels.to(self.device, non_blocking=True),
        }

    def train_one_step(self, data: TrainBatch) -> TrainStepOutput:
        """Run the forward pass and collect training metrics."""
        with torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        ):
            outputs = self.model(pixel_values=data["pixel_values"], labels=data["labels"])

        predictions = outputs.logits.argmax(dim=-1)
        accuracy = (predictions == data["labels"]).float().mean()

        return {
            "loss": outputs.loss,
            "logs": {
                "train/loss": outputs.loss.item(),
                "train/acc1": accuracy.item(),
            },
        }
