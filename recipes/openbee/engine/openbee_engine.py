"""Training engine for the OpenBee recipe."""

from __future__ import annotations

from typing import Any

import torch
from mvp_dataset import TorchLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import is_main_process
from mvp_engine.engine import ENGINE_REGISTRY, Engine
from mvp_engine.utils.log import logger
from mvp_engine.utils.misc import calculate_model_size

from ..configs.schema import OpenbeeConfig
from ..dataset import (
    ModelInputs,
    OpenbeeCollator,
    build_dataset,
    build_qwen3_vl_processor,
)
from ..model import build_qwen3_vl_model


@ENGINE_REGISTRY.register()
class OpenbeeEngine(Engine):
    """Recipe-local engine for the OpenBee alignment stage."""

    ConfigClass = OpenbeeConfig
    config: OpenbeeConfig

    processor: Any | None = None

    def prepare_dataloader(self, workflow: str = "train"):
        """Build the train-only dataloader over preprocessed multimodal samples.

        Args:
            workflow: Workflow name passed by the shared engine. Only ``train``
                is supported by this recipe.

        Returns:
            A ``TorchLoader`` pipeline that yields padded multimodal batches.
        """
        del workflow  # The shared engine still passes this, but the recipe only supports training.

        if self.processor is None:
            self.processor = build_qwen3_vl_processor(self.config.model)
        dataset = build_dataset(self.config, processor=self.processor)

        collate_fn = OpenbeeCollator(
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

        if is_main_process():
            model_size, trainable_size = calculate_model_size(parallelized_model)
            logger.info(f" - Model size: {model_size / 1e9:.4f} B")
            logger.info(f" - Trainable model size: {trainable_size / 1e9:.4f} B")

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
            raise ValueError("No trainable parameters found for the OpenBee recipe.")

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

    def train_pre_step(self, data: ModelInputs) -> ModelInputs:
        """Move the collated batch to the local device and normalize keys.

        Args:
            data: Raw batch emitted by the dataloader.

        Returns:
            A normalized batch dictionary ready for the model forward pass.
        """
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(self.device, non_blocking=True)
        return data

    def train_one_step(self, data: ModelInputs) -> dict[str, Any]:
        """Run one forward pass and collect training metrics.

        Args:
            data: Normalized multimodal batch on the local device.

        Returns:
            A dictionary containing the loss tensor and logging scalars.
        """
        with torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        ):
            outputs = self.model(**data)

        return {
            "loss": outputs.loss,
            "logs": {
                "train/loss": outputs.loss.item(),
            },
        }
