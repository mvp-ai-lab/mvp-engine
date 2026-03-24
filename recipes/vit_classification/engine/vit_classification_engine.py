"""Training engine for the ViT image classification recipe."""

from typing import TypedDict

import torch
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import get_rank, get_world_size, is_main_process
from mvp_engine.engine import ENGINE_REGISTRY, Engine
from mvp_engine.utils.log import logger
from mvp_engine.utils.misc import calculate_model_size
from mvp_engine.utils.training import accumulate_gradients, clip_grad_norm_

from ..dataset import build_dataset
from ..dataset.sampler import InfiniteDistributedSampler
from ..model import build_vit_model

PEAK_TFLOPS_BY_DEVICE_AND_PRECISION: dict[tuple[str, str], float] = {
    ("NVIDIA H200", "bf16"): 989.0,
    ("NVIDIA H200", "fp16"): 989.0,
}


class TrainBatch(TypedDict):
    """Normalized batch structure consumed by the ViT classifier."""

    pixel_values: torch.Tensor
    labels: torch.Tensor


class TrainStepOutput(TypedDict):
    """Outputs returned by one training step."""

    loss: torch.Tensor
    logs: dict[str, float]


def normalize_precision_name(precision: str) -> str:
    """Normalize precision aliases for peak FLOPs lookup."""
    precision = precision.lower()
    if precision == "float16":
        return "fp16"
    if precision == "bfloat16":
        return "bf16"
    if precision == "float32":
        return "fp32"
    return precision


def resolve_device_name(config) -> str:
    """Resolve the accelerator name used for MFU peak FLOPs lookup."""
    configured_name = OmegaConf.select(config, "model.mfu.device_name", default=None)
    if configured_name:
        return str(configured_name)

    if torch.cuda.is_available():
        return str(torch.cuda.get_device_name(torch.cuda.current_device()))

    return "NVIDIA H200"


def resolve_peak_tflops(config, device: torch.device | None = None, dtype: torch.dtype | None = None) -> float:
    """Resolve peak TFLOPs for the current device and precision."""
    configured_peak_tflops = OmegaConf.select(config, "model.mfu.peak_tflops", default=None)
    if configured_peak_tflops is not None:
        return float(configured_peak_tflops)

    device_name = resolve_device_name(config)
    precision = normalize_precision_name(OmegaConf.select(config, "optim.mixed_precision", default=str(dtype)))
    peak_tflops = PEAK_TFLOPS_BY_DEVICE_AND_PRECISION.get((device_name, precision))

    if peak_tflops is None and device is not None and device.type != "cuda":
        peak_tflops = PEAK_TFLOPS_BY_DEVICE_AND_PRECISION.get(("NVIDIA H200", precision))

    if peak_tflops is None:
        raise ValueError(
            f"Missing MFU peak FLOPs mapping for device={device_name!r}, precision={precision!r}. "
            "Set `model.mfu.device_name` and/or `model.mfu.peak_tflops` in the recipe config."
        )

    return float(peak_tflops)


def calculate_mfu(
    model_flops_per_step_per_rank: float,
    step_time_s: float,
    peak_flops_per_device: float,
    world_size: int,
) -> float:
    """Compute MFU from per-rank model FLOPs, step time, and hardware peak."""
    if step_time_s <= 0:
        raise ValueError(f"`step_time_s` must be positive, got {step_time_s}.")
    if peak_flops_per_device <= 0:
        raise ValueError(f"`peak_flops_per_device` must be positive, got {peak_flops_per_device}.")
    if world_size <= 0:
        raise ValueError(f"`world_size` must be positive, got {world_size}.")

    global_model_flops = model_flops_per_step_per_rank * world_size
    global_peak_flops = peak_flops_per_device * world_size
    return float(global_model_flops / (step_time_s * global_peak_flops))


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

    @property
    def peak_tflops_per_device(self) -> float:
        """Peak accelerator throughput used by MFU logging."""
        return resolve_peak_tflops(self.config, device=self.device, dtype=self.dtype)

    def prepare_model(self) -> torch.nn.Module:
        """Build and parallelize the ViT classifier."""
        model = build_vit_model(self.config.model).to(self.device)
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
        """Compute MFU-related metrics for the current training step."""
        model = self.unwrapped_model
        batch_size = int(self.config.data.batch_size)
        image_size = int(self.config.model.image_size)
        patch_size = int(model.config.patch_size)
        step_time_s = float(self.timer.batch_time_latest)
        peak_tflops = self.peak_tflops_per_device
        mfu = calculate_mfu(
            model_flops_per_step_per_rank=model.calculate_model_flops(
                batch_size=batch_size,
                image_size=image_size,
                patch_size=patch_size,
                is_training=True,
            ),
            step_time_s=step_time_s,
            peak_flops_per_device=peak_tflops * 1e12,
            world_size=get_world_size(),
        )

        if mfu < 0:
            raise ValueError(f"MFU must be non-negative, got {mfu}.")
        if mfu > 1.0:
            logger.warning(
                "MFU sanity check failed: "
                f"mfu={mfu:.4f}, device={resolve_device_name(self.config)}, "
                f"precision={normalize_precision_name(self.config.optim.mixed_precision)}, "
                f"world_size={get_world_size()}, step_time={step_time_s:.4f}s"
            )

        return {
            "mfu": mfu,
            "time/step": step_time_s,
            "hardware/peak_tflops": peak_tflops,
        }

    def train_after_step(self, outputs: dict) -> dict:
        """Execute backward pass, optimizer step, and recipe-local MFU logging."""
        assert "loss" in outputs, "The model output must contain 'loss' key."
        assert "logs" in outputs, "The model output must contain 'logs' key."

        is_sync = self.accumulate_step()
        gradient_accumulation_steps = OmegaConf.select(self.config, "optim.gradient_accumulation_steps", default=1)
        loss = outputs["loss"] / gradient_accumulation_steps

        with accumulate_gradients(self.model, sync=is_sync):
            self.scaler.scale(loss).backward()

        if is_sync:
            self.scaler.unscale_(self.optimizer)

            max_grad_norm = OmegaConf.select(self.config, "optim.clip_grad_norm", default=None)
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

            current_lrs = self.scheduler.get_last_lr()
            for i, lr in enumerate(current_lrs):
                other_logs[f"lr/group_{i}"] = lr

            logger.log_metrics(
                {**outputs["logs"], **other_logs},
                step=self.step,
            )

            self.save()

        return outputs
