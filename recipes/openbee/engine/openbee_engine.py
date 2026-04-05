"""Training engine for the OpenBee recipe."""

from __future__ import annotations

from typing import Any

import torch
from mvp_dataset import TorchLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers.utils.logging import disable_progress_bar

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import is_main_process
from mvp_engine.engine import ENGINE_REGISTRY, Engine
from mvp_engine.utils.log import logger
from mvp_engine.utils.misc import calculate_model_size
from mvp_engine.utils.training import accumulate_gradients, clip_grad_norm_

from ..configs.schema import OpenbeeConfig
from ..dataset import (
    ModelInputs,
    OpenbeeCollator,
    build_dataset,
    build_qwen3_vl_processor,
)
from ..model import build_qwen3_vl_model
from ..model.packing import apply_packed_fa2_patch, prepare_packed_model_inputs
from ..utils.log.mfu import build_mfu_log


@ENGINE_REGISTRY.register()
class OpenbeeEngine(Engine):
    """Recipe-local engine for the OpenBee alignment stage."""

    ConfigClass = OpenbeeConfig
    config: OpenbeeConfig

    processor: Any | None = None

    def __init__(self, config):
        super().__init__(config)
        disable_progress_bar()

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
        if self.config.data.packing and getattr(model.config, "_attn_implementation", None) == "flash_attention_2":
            apply_packed_fa2_patch()
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
        batch: ModelInputs = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=True)
            else:
                batch[key] = value

        return prepare_packed_model_inputs(
            batch,
            model_config=self.unwrapped_model.config,
            attn_implementation=getattr(self.unwrapped_model.config, "_attn_implementation", None),
            mask_dtype=self.dtype if self.dtype.is_floating_point else torch.float32,
        )

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
            "mfu_inputs": {
                "batch_size": int(data["input_ids"].shape[0]),
                "seq_len": int(data["input_ids"].shape[1]),
                "image_grid_thw": data.get("image_grid_thw"),
            },
        }

    def train_after_step(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Run the optimizer step and include recipe-local MFU metrics in logs."""
        assert "loss" in outputs, "The model output must contain 'loss' key."
        assert "logs" in outputs, "The model output must contain 'logs' key."

        is_sync = self.accumulate_step()

        gradient_accumulation_steps = self.config.optim.gradient_accumulation_steps
        loss = outputs["loss"] / gradient_accumulation_steps

        with accumulate_gradients(self.model, sync=is_sync):
            self.scaler.scale(loss).backward()

        if is_sync:
            self.scaler.unscale_(self.optimizer)

            max_grad_norm = self.config.optim.clip_grad_norm
            if max_grad_norm is not None:
                clip_grad_norm_(self.model, max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.step += 1
            self.timer.tick()

            other_logs = {
                "eta": self.timer.eta_string,
                "time/batch": self.timer.batch_time,
                "time/throughput": self.timer.throughput,
            }
            other_logs.update(
                build_mfu_log(
                    model=self.unwrapped_model,
                    device_type=self.device.type,
                    precision=str(self.config.optim.mixed_precision),
                    batch_size=int(outputs["mfu_inputs"]["batch_size"]),
                    seq_len=int(outputs["mfu_inputs"]["seq_len"]),
                    image_grid_thw=outputs["mfu_inputs"].get("image_grid_thw"),
                    step_time_seconds=float(self.timer.batch_time_latest),
                    gradient_accumulation_steps=int(self.config.optim.gradient_accumulation_steps),
                    freeze_vit=bool(self.config.model.freeze_vit),
                    freeze_merger=bool(self.config.model.freeze_merger),
                    freeze_llm=bool(self.config.model.freeze_llm),
                )
            )

            for i, lr in enumerate(self.scheduler.get_last_lr()):
                other_logs[f"lr/group_{i}"] = lr

            logger.log_metrics(
                {**outputs["logs"], **other_logs},
                step=self.step,
            )

            self.save()

        return outputs

    def run_iter_train(self) -> None:
        """Run iteration-based training loop until total_steps is reached."""
        while self.step < self.total_steps:
            if hasattr(self, "data"):
                if self.step >= self.total_steps:
                    # In case it's a infinity loader
                    break
                self.train_after_step(self.train_one_step(self.train_pre_step(self.data)))
            else:
                for data in self.train_loader:
                    self.data = data
                    if self.step >= self.total_steps:
                        # In case it's a infinity loader
                        break
                    self.train_after_step(self.train_one_step(self.train_pre_step(data)))
                    break
