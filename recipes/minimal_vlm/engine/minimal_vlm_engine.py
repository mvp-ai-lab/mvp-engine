"""Training engine for the minimal VLM recipe."""

from __future__ import annotations

import math
from typing import Any

import torch
from mvp_dataset import TorchLoader
from transformers import get_scheduler

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.engine import ENGINE_REGISTRY, Engine, TrainStepContext
from mvp_engine.utils.log import logger

from ..configs.schema import MinimalVLMConfig
from ..dataset.collator import MinimalVLMCollator
from ..dataset.dataset import build_dataset
from ..dataset.processor import build_qwen3_vl_processor
from ..dataset.types import ModelInputs
from ..model.qwen3_vl import build_qwen3_vl_model


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

        # Step 1: build the processor
        if self.processor is None:
            self.processor = build_qwen3_vl_processor(self.config.model)

        # Step 2: build the real training dataset with the full preprocess.
        dataset = build_dataset(self.config, processor=self.processor)
        collate_fn = MinimalVLMCollator(
            pad_token_id=int(self.processor.tokenizer.pad_token_id),
        )

        # Step 3: wrap the dataset in the normal TorchLoader and return the
        # batched dataloader used by the engine.
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
        """Build, patch, and parallelize the recipe model.

        Args:
            None.

        Returns:
            The distributed-ready Qwen3-VL model.
        """
        model = build_qwen3_vl_model(self.config.model).to(self.device)
        logger.info(f"Model name: {model.__class__.__name__}")

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

    def prepare_scheduler(self):
        """Construct the learning-rate schedule used by this recipe.

        Args:
            None.

        Returns:
            The warmup-plus-cosine scheduler for the current optimizer.
        """
        warmup_steps = math.ceil(self.total_steps * float(self.config.optim.warmup_ratio))
        return get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

    def train_pre_step(self, ctx: TrainStepContext) -> TrainStepContext:
        """Prepare one micro-batch for forward."""
        data: ModelInputs = ctx.data
        batch: ModelInputs = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=True)
            else:
                batch[key] = value
        ctx.data = batch
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
