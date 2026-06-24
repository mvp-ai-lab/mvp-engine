"""Base training engine pipeline and hook definitions."""

import logging
import os
import platform
import secrets
import shutil
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, Type, Union

import torch
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from mvp_engine.config.schema import BaseEngineConfig
from mvp_engine.distributed.init import initialize_process_group
from mvp_engine.distributed.parallel_mesh import ParallelMesh
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
from mvp_engine.utils.log.timer import Timer
from mvp_engine.utils.misc import calculate_model_size, get_device, get_git_info
from mvp_engine.utils.training import (
    GradientAccumulationState,
    GradientScaler,
    accumulate_gradients,
    clip_grad_norm_,
)

logging.captureWarnings(True)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "ERROR").upper())


@dataclass
class TrainStepContext:
    """Mutable state for one training micro-batch."""

    data: Any
    step: int
    epoch: int
    micro_step: int
    outputs: dict[str, Any] | None = None
    loss: torch.Tensor | None = None
    should_sync: bool = False
    optimizer_step_completed: bool = False


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

    parallel_mesh: ParallelMesh

    train_loader: DataLoader
    evaluate_loader: DataLoader

    model: Union[torch.nn.Module, DistributedDataParallel, FSDPModule]

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: GradientScaler

    epoch: int = 0  # current epoch
    step: int = 0  # current optimization step = (epoch * len(train_loader) + micro_step) / gradient_accumulation_steps
    ga_state: GradientAccumulationState

    timer: Timer  # timer for tracking per-batch time and ETA

    def __init__(self, config: DictConfig):
        """Validate config and initialize distributed, runtime, seed, and logging state."""
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
        """Return the torch device assigned to the current local rank."""
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
    def _accumulate_step(self) -> int:
        """Backward-compatible view of the current accumulation micro-step."""
        return self.ga_state.micro_step

    @_accumulate_step.setter
    def _accumulate_step(self, value: int) -> None:
        """Set the current accumulation micro-step for backward compatibility."""
        self.ga_state.micro_step = int(value)

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
        self.parallel_mesh = ParallelMesh.initialize(self.device.type, mesh_cfg)

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

        cur_checkpoint_dir = checkpoints_dir / (
            f"{self.loop_policy}_{self.epoch if self.loop_policy == 'epoch' else self.step}"
        )

        # Keep only last N checkpoints
        if is_main_process():
            all_checkpoints = os.listdir(str(checkpoints_dir))
            checkpoint_paths = sorted(
                all_checkpoints,
                key=lambda dir: int(dir.split("_")[-1]),
            )
            checkpoint_paths = [path for path in checkpoint_paths if path != cur_checkpoint_dir.name]
            keep_existing_n = self.config.checkpoint.keep_n - 1
            delete_n = max(len(checkpoint_paths) - keep_existing_n, 0)
            for delete_path in checkpoint_paths[:delete_n]:
                shutil.rmtree(checkpoints_dir / delete_path)

        cur_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()

        save_checkpoint(
            self.parallel_mesh.device_mesh,
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
            self.parallel_mesh.device_mesh,
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

    """
    Train workflow:
     |   1. before_train: Initialize components (model, optimizer, dataloaders)
     |   2. do_train: Execute training loop
     |       a. train_pre_step: Preprocess batch data
     |       b. train_exec_step: Forward pass, backward pass, and optimizer step
     |       c. train_post_step: Step-level logging and checkpointing
     v   3. after_train: Final checkpoint and evaluation
    """

    def train(self) -> None:
        """Execute complete training pipeline."""
        self.before_train()
        self.do_train()
        self.after_train()

    def before_train(self) -> None:
        """Initialize all components before training starts."""
        logger.log_config(self.config.model_dump())

        logger.info("Building DataLoader...")
        self.train_loader = self.prepare_dataloader("train")
        self.evaluate_loader = self.prepare_dataloader("evaluate")

        logger.info("Building Model...")
        self.model = self.prepare_model()
        model_size, trainable_size = calculate_model_size(self.model)
        logger.info(f" - Model size: {model_size / 1e9:.4f} B")
        logger.info(f" - Trainable model size: {trainable_size / 1e9:.4f} B")

        logger.info("Building Optimizer...")
        self.optimizer = self.prepare_optimizer()
        self.ga_state = GradientAccumulationState(self.config.optim.gradient_accumulation_steps)

        logger.info("Building Scheduler...")
        self.scheduler = self.prepare_scheduler()

        logger.info("Building GradientScaler...")
        mixed_precision_enabled = self.dtype != torch.float32
        self.scaler = GradientScaler(
            enabled=mixed_precision_enabled,
            dtype=self.dtype,
            device=self.device_type,
        )

        resume_path = getattr(self.config, "resume", None)
        if resume_path is not None:
            self.load(
                resume_path,
                restore_training_state=True,
                restore_rng_state=True,
            )

        logger.info("Initializing Timer...")
        self.timer = Timer(
            total_progress=self.total_steps,
            window_size=self.config.log.timer_window_size,
        )

    def do_train(self) -> None:
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
                f"{self.scheduler.__class__.__name__}"
                if hasattr(self, "scheduler") and self.scheduler is not None
                else ""
            )
            + "..."
        )

        self.model.train()
        self.timer.start()
        self.timer.set_progress(self.step)

        while self.progress < 1.0:
            train_loader_iter = iter(self.train_loader)
            try:
                while True:
                    if self.progress >= 1.0:
                        break  # In case it's a infinity loader

                    with self.timer.scope("data_time"):
                        data = next(train_loader_iter)
                        ctx = TrainStepContext(
                            data=data,
                            step=self.step,
                            epoch=self.epoch,
                            micro_step=self.ga_state.micro_step,
                        )
                        prepared = self.train_pre_step(ctx)
                        if prepared is not None and prepared is not ctx:
                            ctx.data = prepared

                    with self.timer.scope("exec_time"):
                        self.train_exec_step(ctx)

                    if not ctx.optimizer_step_completed:
                        continue

                    assert ctx.outputs is not None, "The forward step must populate ctx.outputs."

                    self.step += 1
                    self.timer.tick()

                    self.train_post_step(ctx)

                    self.save()
            except StopIteration:
                self.epoch += 1
                logger.info(f"Starting epoch {self.epoch}...")
                continue

    def train_pre_step(self, ctx: TrainStepContext) -> TrainStepContext:
        """Preprocess the input data before training step."""
        pass

    def train_exec_step(self, ctx: TrainStepContext) -> None:
        """Execute forward, backward, and optimizer phases for one micro-batch."""
        self.forward_step(ctx)
        self.backward_step(ctx)
        self.optimizer_step(ctx)

    def forward_step(self, ctx: TrainStepContext) -> None:
        """Run the model forward pass and store outputs in the step context."""
        # Forward pass with mixed precision autocast
        with torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        ):
            outputs = self.model(ctx.data)
        ctx.outputs = outputs

    def backward_step(self, ctx: TrainStepContext) -> None:
        """Scale loss, advance accumulation state, and run backward."""
        assert ctx.outputs is not None, "The forward step must populate ctx.outputs."
        assert "loss" in ctx.outputs, "The model output must contain 'loss' key."
        assert "logs" in ctx.outputs, "The model output must contain 'logs' key."

        ctx.should_sync = self.ga_state.advance()

        ctx.loss = ctx.outputs["loss"] / self.config.optim.gradient_accumulation_steps

        with accumulate_gradients(self.model, sync=ctx.should_sync):
            self.scaler.scale(ctx.loss).backward()

    def optimizer_step(self, ctx: TrainStepContext) -> None:
        """Apply optimizer, scaler, scheduler, and timer updates at sync steps."""
        if not ctx.should_sync:
            return

        self.scaler.unscale_(self.optimizer)
        max_grad_norm = self.config.optim.clip_grad_norm
        if max_grad_norm is not None:
            clip_grad_norm_(self.model, max_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        ctx.optimizer_step_completed = True

    def train_post_step(self, ctx: TrainStepContext) -> None:
        """Log optimizer-step metrics and save checkpoints."""
        other_logs = {
            "eta": self.timer.eta_string,
            "perf/data_time": self.timer.get_scope_time("data_time"),
            "perf/exec_time": self.timer.get_scope_time("exec_time"),
        }

        current_lrs = self.scheduler.get_last_lr()
        for i, lr in enumerate(current_lrs):
            other_logs[f"lr/group_{i}"] = lr

        logger.log_metrics(
            {**ctx.outputs["logs"], **other_logs},
            step=self.step,
            total_steps=self.total_steps,
        )

    def after_train(self) -> None:
        """Finalize training with checkpoint save, evaluation, and cleanup."""
        self.save(force=True and not self.config.dev_mode)
        logger.info("Training finished!")
        logger.destroy()

    """
    Evaluate workflow:
     |   1. before_evaluate
     |   2. do_evaluate
     v   3. after_evaluate
    """

    @torch.no_grad()
    def evaluate(self):
        """Evaluate the model on the evaluation dataset."""
        raise NotImplementedError("The evaluate method must be implemented in subclasses.")
