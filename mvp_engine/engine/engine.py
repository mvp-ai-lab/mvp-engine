import logging
import os
import platform
import secrets
import shutil
import socket
import time
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, Type, Union

import torch
from accelerate.utils import set_seed
from addict import Dict
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from mvp_engine.config.schema import BaseEngineConfig
from mvp_engine.distributed.device_mesh import initialize_device_mesh
from mvp_engine.distributed.init import initialize_process_group
from mvp_engine.distributed.utils import (
    broadcast_from_main,
    get_local_rank,
    is_main_process,
)
from mvp_engine.utils.checkpointing.parallel_sl_util import (
    load_checkpoint,
    save_checkpoint,
)
from mvp_engine.utils.log import init_logger, logger
from mvp_engine.utils.log.backend import FileBackend, TerminalBackend, WandbBackend
from mvp_engine.utils.misc import Timer, get_device, get_git_info
from mvp_engine.utils.training import (
    GradientScaler,
    accumulate_gradients,
    clip_grad_norm_,
)

logging.captureWarnings(True)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "ERROR").upper())


class Engine(ABC):
    """Abstract base class for training and evaluation workflows.

    Provides a structured pipeline for deep learning experiments with support for:
    - Distributed training (DDP)
    - Mixed precision (fp16, bf16, fp32)
    - Gradient accumulation
    - Gradient clipping
    - Checkpointing
    - Flexible logging backends

    Subclasses must implement:
        - prepare_model()
        - prepare_optimizer()
        - prepare_scheduler()
        - prepare_dataloader()
        - evaluate()

    Attributes:
        config: Validated Pydantic configuration for the experiment.
        train_loader: DataLoader for training data.
        evaluate_loader: DataLoader for evaluation data.
        model: Neural network model (possibly wrapped in DDP).
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        scaler: Gradient scaler for mixed precision.
        epoch: Current epoch number.
        step: Global optimization step counter.
    """

    ConfigClass: ClassVar[Type[BaseEngineConfig]] = BaseEngineConfig
    config: BaseEngineConfig

    device_mesh: DeviceMesh

    train_loader: DataLoader
    evaluate_loader: DataLoader

    model: Union[torch.nn.Module, DistributedDataParallel, FSDPModule]

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: GradientScaler

    epoch: int = 0  # current epoch
    step: int = 0  # current optimization step = (epoch * len(train_loader) + iter) / gradient_accumulation_steps
    _accumulate_step: int = 0  # internal counter for gradient accumulation

    timer: Timer  # timer for tracking per-batch time and ETA

    def __init__(self, config: DictConfig):
        self.config = self.prepare_config(config)
        self.prepare_parallel()
        self.prepare_runtime_info()
        self.set_seed(self.config.seed, self.config.deterministic)
        self.prepare_logger()

    def prepare_config(self, config: DictConfig) -> BaseEngineConfig:
        """Convert an OmegaConf config into the validated Pydantic config model."""
        d = OmegaConf.to_container(config, resolve=True)
        return self.ConfigClass.model_validate(d)

    @property
    def device(self) -> torch.device:
        return get_device(index=get_local_rank())

    @property
    def device_type(self) -> str:
        """Return device type string for torch.autocast."""
        device = self.device
        if device.type == "cuda":
            return "cuda"
        elif device.type == "npu":
            return "npu"
        return "cpu"

    @property
    def dtype(self) -> torch.dtype:
        """Compute dtype for mixed precision training."""
        dtype_str = self.config.optim.mixed_precision
        if dtype_str == "fp32":
            return torch.float32
        elif dtype_str == "fp16":
            return torch.float16
        elif dtype_str == "bf16":
            return torch.bfloat16
        else:
            return torch.float32

    @property
    def total_steps(self) -> int:
        """Total number of optimization steps for the training run."""
        if self.loop_policy == "iter":
            return self.config.loop.total_steps
        else:
            raise NotImplementedError(f"Unsupported loop policy: {self.loop_policy}")

    @property
    def max_grad_norm(self) -> float | None:
        """Maximum gradient norm for clipping, or None to disable."""
        return self.config.optim.clip_grad_norm

    @property
    def project_dir(self) -> Path:
        """Root directory for outputs and checkpoints."""
        return Path(self.config.runtime.output_dir)

    @property
    def run_id(self) -> str:
        """Unique identifier for this training run."""
        return self.config.runtime.run_id

    @property
    def loop_policy(self) -> str:
        """Training loop policy: 'iter' or 'epoch'."""
        return self.config.loop.policy

    @property
    def progress(self) -> float:
        """Training progress as a float between 0 and 1."""
        if self.loop_policy == "iter":
            return self.step / self.total_steps
        else:
            raise ValueError(f"Unsupported loop policy: {self.loop_policy}")

    @property
    def unwrapped_model(self) -> torch.nn.Module:
        """Return the underlying model, unwrapping DistributedDataParallel if needed."""
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model

    def set_seed(self, seed: int, deterministic: bool = False) -> None:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed value.
            deterministic: Whether to enable deterministic mode (slower but reproducible).
        """
        set_seed(seed, deterministic)

    def prepare_parallel(self) -> None:
        """Initialize distributed training backend."""
        initialize_process_group()
        mesh_cfg = self.config.parallel.mesh.model_dump()
        self.device_mesh = initialize_device_mesh(self.device.type, mesh_cfg)

    def prepare_runtime_info(self) -> None:
        """Inject runtime metadata that depends on the initialized distributed state."""
        git_info = get_git_info()
        runtime = self.config.runtime
        runtime.git_info = f"<{git_info['branch']}> {git_info['commit_hash']}"
        runtime.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        runtime.hostname = socket.gethostname()
        runtime.python_version = platform.python_version()
        runtime.torch_version = torch.__version__

        if not runtime.run_id:
            local_run_id = (
                f"{self.config.project.name}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{secrets.token_hex(2)}"
            )
            runtime.run_id = broadcast_from_main(local_run_id)

        runtime.output_dir = str(Path(self.config.project.dir) / runtime.run_id)

    def prepare_logger(self) -> None:
        """Initialize logging backends based on configuration."""
        logger_backends = []
        if self.config.dev_mode:
            logger_backends = [TerminalBackend(id=self.config.runtime.run_id)]
        else:
            logger_backends = []
            config_backends = self.config.log.backends

            for backend in config_backends:
                if backend == "terminal":
                    logger_backends.append(TerminalBackend(id=self.config.runtime.run_id))
                elif backend == "file":
                    logger_backends.append(
                        FileBackend(
                            id=self.run_id,
                            path=Path(self.config.runtime.output_dir),
                        )
                    )
                elif backend == "wandb":
                    logger_backends.append(
                        WandbBackend(
                            id=self.run_id,
                            project=self.config.project.name,
                            path=Path(self.config.runtime.output_dir),
                        )
                    )
                else:
                    raise ValueError(f"Invalid log backend: {backend}")

        global logger
        logger = init_logger(
            logger_backends,
            interval=self.config.log.interval,
            accumulation_size=(
                self.config.log.accumulation_size
                if self.config.log.accumulation_size is not None
                else self.config.log.interval
            ),
        )

    @abstractmethod
    def prepare_model(self) -> torch.nn.Module:
        """Build and return the model instance."""

    @abstractmethod
    def prepare_optimizer(self) -> torch.optim.Optimizer:
        """Build and return the optimizer instance."""

    @abstractmethod
    def prepare_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Build and return the scheduler instance."""

    @abstractmethod
    def prepare_dataloader(self, workflow: str = "train") -> DataLoader:
        """Build and return the dataloader instance for the given stage."""

    def save(self, force: bool = False) -> None:
        """Save training checkpoint to disk.

        Args:
            force: If True, save regardless of save_interval.
        """
        save_interval = self.config.checkpoint.interval
        if (not force and (self.step % save_interval != 0)) or self.config.dev_mode:
            return
        logger.info(f"Saving checkpoint for step {self.step}...")

        checkpoints_dir: Path = self.project_dir / "checkpoints"

        # Check if checkpoints directory exists
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Keep only last N checkpoints
        if is_main_process():
            all_checkpoints = os.listdir(str(checkpoints_dir))
            if len(all_checkpoints) >= self.config.checkpoint.keep_n:
                checkpoint_paths = sorted(
                    all_checkpoints,
                    key=lambda dir: int(dir.split("_")[-1]),
                )
                delete_n = len(checkpoint_paths) - self.config.checkpoint.keep_n + 1
                for delete_path in checkpoint_paths[:delete_n]:
                    shutil.rmtree(checkpoints_dir / delete_path)

        cur_checkpoint_dir = checkpoints_dir / (
            f"{self.loop_policy}_{self.epoch if self.loop_policy == 'epoch' else self.step}"
        )
        cur_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()

        save_checkpoint(
            self.device_mesh,
            cur_checkpoint_dir,
            self.model,
            self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            step=self.step,
            epoch=self.epoch,
            _accumulate_step=self._accumulate_step,
            hf_enable=self.config.checkpoint.hf_enable,
        )

        torch.distributed.barrier()

    def load(
        self,
        ckpt_path: Union[str, PathLike],
        restore_training_state: bool = True,
        restore_rng_state: bool = True,
    ) -> None:
        """Load training checkpoint from disk.

        Args:
            ckpt_path: Path to checkpoint directory.
            restore_training_state: Whether to restore optimizer, scheduler,
                scaler, and engine state in addition to model weights.
            restore_rng_state: Whether to restore RNG state when loading a
                training checkpoint.
        """
        action = "Loading checkpoint" if restore_training_state else "Initializing model from checkpoint"
        logger.info(f"{action} {ckpt_path}...")

        engine_state = load_checkpoint(
            self.device_mesh,
            ckpt_path,
            self.model,
            self.optimizer if restore_training_state else None,
            self.scheduler if restore_training_state else None,
            self.scaler if restore_training_state else None,
            restore_engine_state=restore_training_state,
            restore_rng_state=restore_rng_state,
            hf_enable=self.config.checkpoint.hf_enable,
        )
        if engine_state is None:
            return

        self.step = engine_state["step"]
        self.epoch = engine_state["epoch"]
        self._accumulate_step = engine_state["_accumulate_step"]

        if hasattr(self, "timer"):
            self.timer.set_progress(self.step, self.total_steps)

    def accumulate_step(self, skip_increase: bool = False) -> bool:
        """Check if the gradients should be synchronized this step."""
        if not skip_increase:
            self._accumulate_step += 1

        gradient_accumulation_steps = self.config.optim.gradient_accumulation_steps

        if self._accumulate_step % gradient_accumulation_steps == 0:
            self._accumulate_step = 0
            return True
        else:
            return False

    """
    Train workflow:
     |   1. before_train: Initialize components (model, optimizer, dataloaders)
     |   2. run_train: Execute training loop
     |       a. train_pre_step: Preprocess batch data
     |       b. train_one_step: Forward pass and compute loss
     |       c. train_after_step: Backward pass, optimizer step, logging
     v   3. after_train: Final checkpoint and evaluation
    """

    def train(self) -> None:
        """Execute complete training pipeline."""
        self.before_train()
        self.run_train()
        self.after_train()

    def before_train(self) -> None:
        """Initialize all components before training starts."""
        logger.log_config(self.config.model_dump())

        logger.info("Building DataLoader...")
        self.train_loader = self.prepare_dataloader("train")
        self.evaluate_loader = self.prepare_dataloader("evaluate")

        logger.info("Building Model...")
        self.model = self.prepare_model()

        logger.info("Building Optimizer...")
        self.optimizer = self.prepare_optimizer()

        logger.info("Building Scheduler...")
        self.scheduler = self.prepare_scheduler()

        logger.info("Building GradientScaler...")
        mixed_precision_enabled = self.dtype != torch.float32
        self.scaler = GradientScaler(
            enabled=mixed_precision_enabled,
            dtype=self.dtype,
            device=self.device_type,
        )

        logger.info("Initializing Timer...")
        # FIXME: use progress rather than total_steps.
        self.timer = Timer(
            total_batches=self.total_steps,
            window_size=self.config.log.timer_window_size,
        )

        # FIXME: remove this.
        if self.config.init_from_checkpoint is not None:
            self.load(
                self.config.init_from_checkpoint,
                restore_training_state=False,
                restore_rng_state=False,
            )

    def run_train(self) -> None:
        """Execute the main training loop based on loop_policy."""
        logger.info(
            "Start training: "
            + (
                f"{self.model.module.__class__.__name__} / "
                if hasattr(self.model, "module")
                else f"{self.model.__class__.__name__} / "
            )
            + (
                f"{self.optimizer.__class__.__name__} / "
                if hasattr(self, "optimizer") and self.optimizer is not None
                else ""
            )
            + (
                f"{self.scheduler.__class__.__name__} / "
                if hasattr(self, "scheduler") and self.scheduler is not None
                else ""
            )
            + "..."
        )

        self.model.train()
        self.timer.start()

        if self.loop_policy == "iter":
            self.run_iter_train()
        elif self.loop_policy == "epoch":
            self.run_epoch_train()

    def run_iter_train(self) -> None:
        """Run iteration-based training loop until total_steps is reached."""
        while self.progress < 1.0:
            for data in self.train_loader:
                if self.progress >= 1.0:
                    # In case it's a infinity loader
                    break
                self.train_after_step(self.train_one_step(self.train_pre_step(data)))

    def run_epoch_train(self) -> None:
        """Run epoch-based training loop (not yet implemented)."""
        raise NotImplementedError("Epoch-based training is not implemented yet.")

    def train_pre_step(self, data: Any) -> Any:
        """Preprocess the input data before training step."""
        return data

    def train_one_step(self, data: Any) -> Dict:
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

        return outputs

    def train_after_step(self, outputs: Dict) -> Dict:
        """Execute backward pass, optimizer step, and logging.

        Handles:
        - Gradient accumulation
        - Mixed precision scaling
        - Gradient clipping
        - Optimizer and scheduler stepping
        - Metric logging
        - Checkpoint saving

        Args:
            outputs: Model outputs containing 'loss' and 'logs'.

        Returns:
            The same outputs dict.
        """
        assert "loss" in outputs, "The model output must contain 'loss' key."
        assert "logs" in outputs, "The model output must contain 'logs' key."

        # Determine if we should sync gradients this step
        is_sync = self.accumulate_step()

        # Scale loss for gradient accumulation
        gradient_accumulation_steps = self.config.optim.gradient_accumulation_steps
        loss = outputs["loss"] / gradient_accumulation_steps

        # Backward pass with optional DDP no_sync for accumulation
        with accumulate_gradients(self.model, sync=is_sync):
            self.scaler.scale(loss).backward()

        # Only step optimizer when gradients are synchronized
        if is_sync:
            # Unscale gradients before clipping (required for GradScaler)
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            max_grad_norm = self.config.optim.clip_grad_norm
            if max_grad_norm is not None:
                clip_grad_norm_(self.model, max_grad_norm)

            # Optimizer step (skipped if inf/nan gradients detected by scaler)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Scheduler step (after optimizer step)
            self.scheduler.step()

            # Zero gradients for next accumulation cycle
            self.optimizer.zero_grad(set_to_none=True)

            # Increment global step counter
            self.step += 1

            # Record batch time and update timer
            self.timer.tick()

            # Log training metrics with timing info
            other_logs = {
                "eta": self.timer.eta_string,
                "time/batch": self.timer.batch_time,
                "time/throughput": self.timer.throughput,
            }

            # Log LR
            current_lrs = self.scheduler.get_last_lr()
            for i, lr in enumerate(current_lrs):
                other_logs[f"lr/group_{i}"] = lr

            logger.log_metrics(
                {**outputs["logs"], **other_logs},
                step=self.step,
                total_steps=self.total_steps,
            )

            # Save checkpoint if needed
            self.save()

        return outputs

    def after_train(self) -> None:
        """Finalize training with checkpoint save, evaluation, and cleanup."""
        self.save(force=True and not self.config.dev_mode)
        logger.info("Training finished!")
        logger.destroy()

    """
    evaluate:
     |   1. before_evaluate:
     |   2. run_evaluate:
     |       a. evaluate_pre_step:
     |       b. evaluate_run_step:
     |       c. evaluate_after_step: Optinal
     v   3. after_evaluate: Optional
    """

    @torch.no_grad()
    def evaluate(self):
        """Evaluate the model on the evaluation dataset."""
        raise NotImplementedError("The evaluate method must be implemented in subclasses.")
