import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from webdataset import WebLoader

from mvp_engine.dataset.webdataset import WebDatasetBuilder
from mvp_engine.engine import ENGINE_REGISTRY, Engine
from mvp_engine.utils.distributed.utils import get_rank, is_main_process
from mvp_engine.utils.log import logger

from ..dataset.preprocess import make_sample
from ..model.ibot import iBOTLoss
from ..model.partial_fc import PartialFC
from ..model.tomato_vit import TomatoViTModel


@ENGINE_REGISTRY.register()
class TomatoViTEngine(Engine):
    teacher_model: torch.nn.Module
    rgb_head: torch.nn.Module
    depth_head: torch.nn.Module
    ibot_loss: iBOTLoss

    def __init__(self, config):
        super().__init__(config)

    def prepare_dataloader(self, workflow="train"):
        if workflow == "train":
            dataset = WebDatasetBuilder(self.config.data.dataset_url).build(
                batch_size=self.config.data.batch_size,
                make_sample_func=make_sample,
                shuffle_buffer=self.config.data.shuffle_buffer,
            )
            dataloader = (
                WebLoader(
                    dataset, batch_size=None, num_workers=self.config.data.num_workers
                )
                .unbatched()
                .shuffle(self.config.data.shuffle_buffer)
                .batched(dataset.batch_size)
            )
            return dataloader
        else:
            raise NotImplementedError(f"Workflow {workflow} not implemented.")

    def prepare_model(self):
        # 0. Main model
        model = TomatoViTModel.from_pretrained(self.config.model.pretrained)

        # 1. Teacher model
        self.teacher_model = deepcopy(model).to(self.device)
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        def cosine_scheduler(base_value, final_value, nsteps, warmup_steps=0, start_warmup_value=0):
            """Cosine scheduler for momentum or learning rate."""
            warmup_schedule = np.array([])
            if warmup_steps > 0:
                warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_steps)

            iters = np.arange(nsteps - warmup_steps)
            schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

            schedule = np.concatenate((warmup_schedule, schedule))
            assert len(schedule) == nsteps
            return schedule

        self.momentum_schedule = cosine_scheduler(
            base_value=0.996,
            final_value=1.0,
            nsteps=self.total_steps,
            warmup_steps=0,
        )

        # 2. Partial FC heads
        embedding_size = model.config.hidden_size
        self.depth_head = PartialFC(
            embedding_size=embedding_size,
            num_classes=self.config.model.partial_fc.num_classes,
            sample_rate=self.config.model.partial_fc.sample_rate,
            margin=self.config.model.partial_fc.margin,
        ).to(self.device)
        self.rgb_head = PartialFC(
            embedding_size=embedding_size,
            num_classes=self.config.model.partial_fc.num_classes,
            sample_rate=self.config.model.partial_fc.sample_rate,
            margin=self.config.model.partial_fc.margin,
        ).to(self.device)

        # 3. Freeze modules
        if self.config.model.freeze_rgb_backbone:
            for param in self.model.embeddings.parameters():
                param.requires_grad = False
            for param in self.model.layernorm_pre.parameters():
                param.requires_grad = False
            for param in self.model.video_rope.parameters():
                param.requires_grad = False
            for param in self.model.encoder.layers.parameters():
                param.requires_grad = False
            for name, param in self.model.encoder.mixture_layers.named_parameters():
                if "_a." in name:
                    param.requires_grad = False

        if self.config.model.freeze_depth_backbone:
            for param in self.model.embeddings_depth.parameters():
                param.requires_grad = False
            for param in self.model.layernorm_pre_depth.parameters():
                param.requires_grad = False
            for param in self.model.video_rope.parameters():
                param.requires_grad = False
            for param in self.model.encoder.layers_depth.parameters():
                param.requires_grad = False
            for name, param in self.model.encoder.mixture_layers.named_parameters():
                if "_b." in name:
                    param.requires_grad = False

        if self.config.model.freeze_rgb_pooling:
            for param in self.model.layernorm_post.parameters():
                param.requires_grad = False
            for param in self.model.head.parameters():
                param.requires_grad = False

        if self.config.model.freeze_depth_pooling:
            for param in self.model.layernorm_post_depth.parameters():
                param.requires_grad = False
            for param in self.model.head_depth.parameters():
                param.requires_grad = False

        if self.config.model.freeze_rgb_head:
            for param in self.rgb_head.parameters():
                param.requires_grad = False
        
        if self.config.model.freeze_depth_head:
            for param in self.depth_head.parameters():
                param.requires_grad = False

        # 4. iBOT masked modeling loss
        self.ibot_loss = iBOTLoss(
            embedding_size,
            warmup_teacher_temp=self.config.model.ibot.warmup_teacher_temp,
            teacher_temp=self.config.model.ibot.teacher_temp,
            warmup_teacher_temp_steps=self.config.model.ibot.warmup_teacher_temp_steps,
            nsteps=self.total_steps,
            student_temp=self.config.model.ibot.student_temp,
            center_momentum=self.config.model.ibot.center_momentum,
            lam=self.config.model.ibot.lam,
            mim_start_step=self.config.model.ibot.mim_start_step,
        ).to(self.device)

        # 5. DDP wrap
        if self.config.parallel.type == "ddp":
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                model.to(self.device),
                device_ids=[self.device.index]
                if self.device.type == "cuda"
                else None,
                output_device=self.device.index
                if self.device.type == "cuda"
                else None,
            )
        else:
            raise NotImplementedError(
                f"Parallel type {self.config.parallel.type} not implemented."
            )

        # 5. Compile model
        ddp_model = torch.compile(ddp_model, backend=self.config.optim.compile_backend, mode=self.config.optim.compile_mode)
        self.teacher_model = torch.compile(
            self.teacher_model, backend=self.config.optim.compile_backend, mode=self.config.optim.compile_mode
        )

        return ddp_model

    def prepare_optimizer(self):
        return torch.optim.AdamW(
            [
                {"params": self.model.parameters(), "lr": self.config.optim.lr},
                {"params": self.depth_head.parameters(), "lr": self.config.optim.lr},
                {"params": self.rgb_head.parameters(), "lr": self.config.optim.lr},
            ],
            weight_decay=self.config.optim.weight_decay,
        )

    def prepare_scheduler(self):
        warmup_steps = int(self.config.loop.total_steps * self.config.optim.warmup_ratio)
        scheduler_warmup = LinearLR(
            self.optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps
        )
        scheduler_main = PolynomialLR(
            self.optimizer, total_iters=self.cfg.loop.total_steps - warmup_steps, power=2
        )
        return SequentialLR(
            self.optimizer, [scheduler_warmup, scheduler_main], milestones=[warmup_steps]
        )

    def save(self, force: bool = False) -> None:
        """Save training checkpoint to disk.

        Args:
            force: If True, save regardless of save_interval.
        """
        save_interval = OmegaConf.select(
            self.config, "loop.checkpoint.save_interval", default=1000
        )
        if not force and (self.step % save_interval != 0):
            return

        super().save(force=force)

        checkpoints_dir: Path = self.project_dir / "checkpoints"
        cur_checkpoint_dir = checkpoints_dir / (
            f"iter_{self.step}" if self.loop_policy == "iter" else f"epoch_{self.epoch}"
        )

        parallel_backend = OmegaConf.select(self.config, "parallel.type", default=None)
        if parallel_backend == "ddp":
            if is_main_process():
                torch.save(
                    self.teacher_model.state_dict(),
                    cur_checkpoint_dir / "teacher_model.pt",
                )
                torch.save(
                    self.ibot_loss.state_dict(),
                    cur_checkpoint_dir / "ibot_loss.pt",
                )
            rank = get_rank()
            torch.save(
                self.depth_head.state_dict(),
                cur_checkpoint_dir / f"depth_head_rank{rank}.pt",
            )
            torch.save(
                self.rgb_head.state_dict(),
                cur_checkpoint_dir / f"rgb_head_rank{rank}.pt",
            )
        else:
            raise NotImplementedError(
                f"Unsupported parallel backend: {parallel_backend}"
            )

        torch.distributed.barrier()

    def load(self, ckpt_path: Union[str, os.PathLike]) -> None:
        """Load training checkpoint from disk.

        Args:
            ckpt_path: Path to checkpoint directory.
        """
        super().load(ckpt_path)

        ckpt_path = Path(ckpt_path)
        parallel_backend = OmegaConf.select(self.config, "parallel.type", default=None)
        if parallel_backend == "ddp":
            teacher_model_path = ckpt_path / "teacher_model.pt"
            ibot_loss_path = ckpt_path / "ibot_loss.pt"
            rank = get_rank()
            depth_head_path = ckpt_path / f"depth_head_rank{rank}.pt"
            rgb_head_path = ckpt_path / f"rgb_head_rank{rank}.pt"

            for model_path, model, model_name in [
                (teacher_model_path, self.teacher_model, "Teacher Model"),
                (ibot_loss_path, self.ibot_loss, "iBOT Loss"),
                (depth_head_path, self.depth_head, "Depth Head"),
                (rgb_head_path, self.rgb_head, "RGB Head"),
            ]:
                if model_path.exists():
                    state_dict = torch.load(
                        model_path, map_location="cpu"
                    )
                    model.load_state_dict(state_dict)
                else:
                    logger.warning(f"{model_name} checkpoint not found at {model_path}.")
        else:
            raise NotImplementedError(
                f"Unsupported parallel backend: {parallel_backend}"
            )

    def run_train(self):
        self.teacher_model.eval()
        self.rgb_head.train()
        self.depth_head.train()
        self.ibot_loss.train()

        return super().run_train()

    def train_pre_step(self, data: Dict) -> Dict:
        """Preprocess the input data before training step."""

        # Prepare head labels for Partial FC
        head_label = data["labels"].long().to(self.device)
        label_select = self.config.model.partial_fc.label_select
        random_diff = self.config.model.partial_fc.random_diff
        head_label = head_label[:, label_select : label_select + random_diff]
        data["labels"] = head_label

        return data

    def train_one_step(self, data: Dict) -> Dict:
        """Execute the model forward to get outputs.

        Args:
            data: Preprocessed input batch.

        Returns:
            Dict containing 'loss' and 'logs' keys from model forward.
        """

        # Forward pass with mixed precision autocast
        with torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        ):
            outputs = self.model(data)

        embeddings_rgb = outputs["pooler_output"].float()
        embeddings_depth = outputs["pooler_output_depth"].float()

        random_diff = self.config.model.partial_fc.random_diff
        loss_mlcd_rgb = self.head(embeddings_rgb, data["labels"], random_diff)
        loss_mlcd_depth = self.depth_head(embeddings_depth, data["labels"], random_diff)

        loss_ibot = self.ibot_loss(
            outputs["last_hidden_state"],
            outputs["last_hidden_state_depth"],
            student_mask=data["mask"],
            step=self.step,
        )

        total_loss = (
            loss_mlcd_rgb * self.config.model.partial_fc.rgb_lam
            + loss_mlcd_depth * self.config.model.partial_fc.depth_lam
            + loss_ibot
        )

        return {
            "loss": total_loss,
            "logs": {
                "train/loss": total_loss.item(),
                "train/loss_mlcd_rgb": loss_mlcd_rgb.item(),
                "train/loss_mlcd_depth": loss_mlcd_depth.item(),
                "train/loss_ibot": loss_ibot.item(),
            },
        }

    def train_after_step(self, outputs: Dict) -> Dict:
        outputs = super().train_after_step(outputs)

        if self.accumulate_step(skip_increase=True):
            # EMA update for the teacher
            with torch.no_grad():
                m = self.momentum_schedule[self.step]  # momentum parameter
                for param_q, param_k in zip(self.model.module.parameters(), self.teacher_model.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)