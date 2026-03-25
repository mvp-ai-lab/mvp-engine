"""Training engine for the ViT image classification recipe."""

from typing import TypedDict

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import get_rank, get_world_size, is_main_process
from mvp_engine.engine import ENGINE_REGISTRY, Engine
from mvp_engine.utils.log import logger
from mvp_engine.utils.misc import calculate_model_size
from mvp_engine.utils.training import accumulate_gradients, clip_grad_norm_

from ..configs.schema import ViTClassificationConfig
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


PEAK_TFLOPS_BY_DEVICE_AND_PRECISION = {
    "NVIDIA H200": {
        "bf16": 989.0,
        "fp16": 989.0,
        "fp32": 67.0,
    }
}


def normalize_precision_name(precision: str) -> str:
    """Normalize runtime precision names to the config vocabulary."""
    return precision.lower().replace("float", "fp")


def resolve_device_name(config: ViTClassificationConfig, device: torch.device) -> str:
    """Resolve the device name, preferring the explicit recipe config."""
    configured_name = config.model.mfu.device_name.strip()
    if configured_name:
        return configured_name
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    raise ValueError("Unable to resolve GPU device name for MFU calculation.")


def resolve_peak_tflops(config: ViTClassificationConfig, device_name: str) -> float:
    """Resolve single-device peak throughput for the current precision."""
    if config.model.mfu.peak_tflops > 0:
        return float(config.model.mfu.peak_tflops)

    precision = normalize_precision_name(config.optim.mixed_precision)
    device_flops = PEAK_TFLOPS_BY_DEVICE_AND_PRECISION.get(device_name)
    if device_flops is None or precision not in device_flops:
        raise ValueError(f"Unsupported MFU hardware mapping: device={device_name!r}, precision={precision!r}")
    return float(device_flops[precision])


def calculate_mfu(
    *,
    model_flops_per_step: float,
    step_time_seconds: float,
    device_peak_tflops: float,
    world_size: int,
) -> float:
    """Compute achieved model FLOPs utilization for the current optimization step."""
    if step_time_seconds <= 0:
        raise ValueError("step_time_seconds must be > 0")
    if device_peak_tflops <= 0:
        raise ValueError("device_peak_tflops must be > 0")
    if world_size <= 0:
        raise ValueError("world_size must be > 0")

    total_peak_flops = device_peak_tflops * 1e12 * world_size
    achieved_flops_per_second = model_flops_per_step / step_time_seconds
    return float(achieved_flops_per_second / total_peak_flops)


@ENGINE_REGISTRY.register()
class ViTClassificationEngine(Engine):
    """Minimal ImageNet classification engine for the ViT recipe template."""

    ConfigClass = ViTClassificationConfig
    config: ViTClassificationConfig

    @property
    def peak_tflops_per_device(self) -> float:
        device_name = resolve_device_name(self.config, self.device)
        return resolve_peak_tflops(self.config, device_name)

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
        model_parameters = list(self.model.parameters())
        optimizer_kwargs = {
            "lr": float(self.config.optim.lr),
            "weight_decay": float(self.config.optim.weight_decay),
        }

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

    def log_mfu_metrics(self) -> dict[str, float]:
        """Collect MFU-related metrics for the current optimization step."""
        model_flops = self.unwrapped_model.calculate_model_flops(
            batch_size=int(self.config.data.batch_size),
            image_size=int(self.config.model.image_size),
            patch_size=int(self.unwrapped_model.config.patch_size),
            is_training=True,
        )
        step_time = float(self.timer.batch_time_latest)
        peak_tflops = self.peak_tflops_per_device
        mfu = calculate_mfu(
            model_flops_per_step=model_flops,
            step_time_seconds=step_time,
            device_peak_tflops=peak_tflops,
            world_size=get_world_size(),
        )

        if mfu < 0:
            raise ValueError(f"MFU must be non-negative, got {mfu}")
        if mfu > 1.0:
            logger.warning(
                "MFU %.4f is above 1.0 for device=%s precision=%s world_size=%s",
                mfu,
                resolve_device_name(self.config, self.device),
                self.config.optim.mixed_precision,
                get_world_size(),
            )

        return {
            "mfu": mfu,
            "time/step": step_time,
            "hardware/peak_tflops": peak_tflops,
        }

    def train_after_step(self, outputs: TrainStepOutput) -> TrainStepOutput:
        """Run optimizer update and append MFU to the step logs."""
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
            other_logs.update(self.log_mfu_metrics())

            for i, lr in enumerate(self.scheduler.get_last_lr()):
                other_logs[f"lr/group_{i}"] = lr

            logger.log_metrics({**outputs["logs"], **other_logs}, step=self.step)
            self.save()

        return outputs
