"""Training engine for the minimal VLM recipe."""

from __future__ import annotations

from typing import Any

import torch
from mvp_dataset import TorchLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.engine import ENGINE_REGISTRY, Engine, TrainStepContext
from mvp_engine.utils.log import logger

from ..configs.schema import MinimalVLMConfig
from ..dataset import (
    MinimalVLMCollator,
    ModelInputs,
    build_dataset,
    build_qwen3_vl_processor,
)
from ..model import build_qwen3_vl_model


@ENGINE_REGISTRY.register()
class MinimalVLMEngine(Engine):
    """Recipe-local engine for supervised Qwen3-VL fine-tuning."""

    ConfigClass = MinimalVLMConfig
    config: MinimalVLMConfig

    processor: Any | None = None

    def prepare_dataloader(self, workflow: str = "train"):
        """Build the train-only dataloader over preprocessed multimodal samples.

        Args:
            workflow: Workflow name passed by the shared engine. Only ``train``
                is supported by this recipe.

        Returns:
            A ``TorchLoader`` pipeline that yields padded multimodal batches.
        """
        if workflow != "train":
            logger.warning(f"Minimal VLM engine does not support workflow '{workflow}'.")
            return

        if self.processor is None:
            self.processor = build_qwen3_vl_processor(self.config.model)
        dataset = build_dataset(self.config, processor=self.processor)

        collate_fn = MinimalVLMCollator(
            pad_token_id=int(self.processor.tokenizer.pad_token_id),
        )

        loader = TorchLoader(
            dataset,
            num_workers=int(self.config.data.num_workers),
            pin_memory=self.device.type in ["cuda", "npu"],
            persistent_workers=False,
        )
        return loader.batch(
            batch_size=int(self.config.data.batch_size),
            drop_last=True,
            collate_fn=collate_fn,
        )

    def prepare_model(self) -> torch.nn.Module:
        """Build the recipe model and wrap it for distributed training.

        Args:
            None.

        Returns:
            The distributed-ready Qwen3-VL model.
        """
        model = build_qwen3_vl_model(self.config.model).to(self.device)
        logger.info(f" - Model name: {model.__class__.__name__}")

        parallelized_model = parallelize_model(
            model,
            device_mesh=self.device_mesh,
            backend_kwargs=self.config.parallel.backend_kwargs.model_dump(),
        )

        return parallelized_model

    def prepare_optimizer(self) -> torch.optim.Optimizer:
        """Construct AdamW over the subset of trainable model parameters.

        Args:
            None.

        Returns:
            The optimizer used by this recipe.
        """
        trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise ValueError("No trainable parameters found for the minimal VLM recipe.")

        return torch.optim.AdamW(
            trainable_parameters,
            lr=float(self.config.optim.lr),
            weight_decay=float(self.config.optim.weight_decay),
        )

    def prepare_scheduler(self) -> SequentialLR | CosineAnnealingLR:
        """Construct the learning-rate schedule used by this recipe.

        Args:
            None.

        Returns:
            The warmup-plus-cosine scheduler for the current optimizer.
        """
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

    def train_pre_step(self, ctx: TrainStepContext) -> TrainStepContext:
        """Move the collated batch to the local device and normalize keys.

        Args:
            ctx: Current training step context.

        Returns:
            The context with a normalized batch dictionary ready for the model
            forward pass.
        """
        data: ModelInputs = ctx.data
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(self.device, non_blocking=True)
        ctx.data = data
        return ctx

    def forward_step(self, ctx: TrainStepContext) -> None:
        """Run one forward pass and collect training metrics.

        Args:
            ctx: Current training step context.
        """
        data: ModelInputs = ctx.data
        with torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        ):
            outputs = self.model(**data)

        ctx.outputs = {
            "loss": outputs.loss,
            "logs": {
                "train/loss": outputs.loss.item(),
            },
        }
