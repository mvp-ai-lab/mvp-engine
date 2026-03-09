from typing import Dict

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
from ..model import build_vit_model, prepare_vit_model_for_tp


@ENGINE_REGISTRY.register()
class ViTClassificationEngine(Engine):
    """Minimal ImageNet classification engine for the ViT recipe template."""

    def prepare_dataloader(self, workflow: str = "train") -> DataLoader:
        dataset = build_dataset(self.config, workflow)
        is_train = workflow == "train"
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=is_train,
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
        model = build_vit_model(self.config.model).to(self.device)
        logger.info(f" - Model name: {model.__class__.__name__}")

        tp_mesh = self.device_mesh["tp"] if self.parallel_backend == "fsdp2" else None
        if tp_mesh is not None and tp_mesh.size() > 1:
            prepare_vit_model_for_tp(model, tp_mesh.size())

        if self.parallel_backend not in ["ddp", "fsdp2"]:
            raise NotImplementedError(f"Parallel type {self.parallel_backend} not implemented.")

        parallelized_model = parallelize_model(
            model,
            device_mesh=self.device_mesh,
            backend=self.parallel_backend,
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

    def prepare_scheduler(self):
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

    def verify_checkpoint(self) -> None:
        load_from_cfg = OmegaConf.select(self.config, "model.load_from", default=None)
        if load_from_cfg and load_from_cfg.path:
            self.load(ckpt_path=load_from_cfg.path)

    def run_train(self) -> None:
        self.model.train()
        self.timer.start()

        epoch = 0
        while self.step < self.total_steps:
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            for data in self.train_loader:
                if self.step >= self.total_steps:
                    break
                self.train_after_step(self.train_one_step(self.train_pre_step(data)))

            epoch += 1

    def train_pre_step(self, data) -> Dict[str, torch.Tensor]:
        pixel_values, labels = data
        return {
            "pixel_values": pixel_values.to(self.device, non_blocking=True),
            "labels": labels.to(self.device, non_blocking=True),
        }

    def train_one_step(self, data: Dict[str, torch.Tensor]) -> Dict:
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

    @torch.no_grad()
    def evaluate(self):
        logger.log_config(self.config)
        self.evaluate_loader = self.prepare_dataloader("evaluate")
        self.model = self.prepare_model()
        self.model.eval()

        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        total_samples = 0

        for data in self.evaluate_loader:
            batch = self.train_pre_step(data)
            with torch.autocast(
                device_type=self.device_type,
                dtype=self.dtype,
                enabled=self.dtype != torch.float32,
            ):
                outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])

            logits = outputs.logits
            labels = batch["labels"]
            batch_size = labels.size(0)
            topk = min(5, logits.size(-1))

            total_loss += outputs.loss.item() * batch_size
            total_top1 += (logits.argmax(dim=-1) == labels).float().sum().item()
            total_top5 += logits.topk(topk, dim=-1).indices.eq(labels.unsqueeze(-1)).any(dim=-1).float().sum().item()
            total_samples += batch_size

        metrics = {
            "evaluate/loss": total_loss / max(total_samples, 1),
            "evaluate/acc1": total_top1 / max(total_samples, 1),
            "evaluate/acc5": total_top5 / max(total_samples, 1),
        }
        logger.log_metrics(metrics, step=self.step)
        logger.info("Evaluation finished!")
        logger.destroy()
