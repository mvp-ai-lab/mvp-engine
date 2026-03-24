import os.path
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from mvp_engine.distributed.utils import get_rank, is_main_process
from mvp_engine.utils.log import simple_info
from mvp_engine.utils.training import GradientScaler

OPTIM_STATE_KEY = "state"


def _get_checkpoint_process_group():
    if dist.is_available() and dist.is_initialized():
        return dist.group.WORLD
    return None


def _get_accelerator_rng_state() -> tuple[str | None, torch.Tensor | None]:
    """Capture RNG state for the active accelerator when supported."""
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_rng_state()

    npu_module = getattr(torch, "npu", None)
    if npu_module is not None and hasattr(npu_module, "get_rng_state"):
        return "npu", npu_module.get_rng_state()

    return None, None


def _set_accelerator_rng_state(device_type: str | None, rng_state: torch.Tensor | None) -> None:
    """Restore RNG state for the accelerator encoded in the checkpoint."""
    if device_type is None or rng_state is None:
        return

    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state(rng_state)
        return

    npu_module = getattr(torch, "npu", None)
    if device_type == "npu" and npu_module is not None and hasattr(npu_module, "set_rng_state"):
        npu_module.set_rng_state(rng_state)


def _infer_checkpoint_backend(mesh: DeviceMesh) -> str:
    """Infer whether checkpoint IO should use the DDP or FSDP2 path from the mesh."""
    mesh_dim_names = mesh.mesh_dim_names or ()

    if "shard" in mesh_dim_names and mesh["shard"].size() > 1:
        return "fsdp2"
    if "tensor" in mesh_dim_names and mesh["tensor"].size() > 1:
        return "fsdp2"

    # Fallback for unnamed meshes: any extra non-replicate dimension implies
    # sharded checkpointing rather than rank-0-only DDP checkpointing.
    mesh_shape = tuple(mesh.shape)
    if len(mesh_shape) <= 1:
        return "ddp"
    if any(dim_size > 1 for dim_size in mesh_shape[1:]):
        return "fsdp2"
    return "ddp"


def _optimizer_state_contains_fqn(optim_state: dict[str, Any], fqn: str) -> bool:
    state = optim_state.get(OPTIM_STATE_KEY)
    if not isinstance(state, dict):
        return False
    if fqn in state:
        return True
    if fqn.startswith("_orig_mod.") and fqn[len("_orig_mod.") :] in state:
        return True
    if f"_orig_mod.{fqn}" in state:
        return True
    return False


def _fill_missing_optimizer_states(model: nn.Module, optim_state: dict[str, Any]) -> int:
    state = optim_state.get(OPTIM_STATE_KEY)
    if not isinstance(state, dict):
        return 0

    missing_count = 0
    for fqn, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _optimizer_state_contains_fqn(optim_state, fqn):
            continue

        state[fqn] = {}
        missing_count += 1

    return missing_count


def save_checkpoint(
    mesh: DeviceMesh,
    cur_checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    scaler: GradientScaler = None,
    step: int = 0,
    epoch: int = 0,
    _accumulate_step: int = 0,
    prefix: str = "",
) -> None:
    """Save model, optimizer, and engine state for the current distributed mesh."""
    backend = _infer_checkpoint_backend(mesh)

    if prefix == "":
        accelerator_type, accelerator_rng_state = _get_accelerator_rng_state()
        rng_state = {
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
        }
        if accelerator_type is not None and accelerator_rng_state is not None:
            rng_state["accelerator_type"] = accelerator_type
            rng_state["accelerator_rng_state"] = accelerator_rng_state
        rank = get_rank()
        engine_path = cur_checkpoint_dir / "engine"
        engine_path.mkdir(parents=True, exist_ok=True)

        dist.barrier()
        torch.save(rng_state, engine_path / f"rank_{rank}.pt")

    if backend == "ddp" and not is_main_process():
        return

    if backend == "ddp":
        if hasattr(model, "module"):
            torch.save(
                model.module.state_dict(),
                cur_checkpoint_dir / f"{prefix}model.pt",
            )
        else:
            simple_info("DDP backend detected but model is not wrapped!")
            torch.save(
                model.state_dict(),
                cur_checkpoint_dir / f"{prefix}model.pt",
            )
        if optimizer is not None:
            torch.save(
                optimizer.state_dict(),
                cur_checkpoint_dir / f"{prefix}optimizer.pt",
            )
    else:
        options = StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
        )
        model_sd = get_model_state_dict(model, options=options)
        state_dict = {
            "model": model_sd,
        }
        if optimizer is not None:
            optim_sd = get_optimizer_state_dict(model, optimizer, options=options)
            missing_count = _fill_missing_optimizer_states(model, optim_sd)
            if missing_count > 0 and is_main_process():
                simple_info(f"Filled {missing_count} missing optimizer state entries before checkpoint save.")
            state_dict["optimizer"] = optim_sd

        if prefix == "":
            dcp_save_path = cur_checkpoint_dir
        else:
            dcp_save_path = cur_checkpoint_dir / f"{prefix}model"

        writer = dcp.FileSystemWriter(dcp_save_path)
        dcp.save(
            state_dict,
            checkpoint_id=str(dcp_save_path),
            process_group=_get_checkpoint_process_group(),
            storage_writer=writer,
        )

    if prefix != "":
        return

    if is_main_process():
        # Save engine state with scheduler, scaler, and rng states
        engine_state = {
            "step": step,
            "epoch": epoch,
            "_accumulate_step": _accumulate_step,
        }
        if scheduler is not None:
            engine_state["scheduler"] = scheduler.state_dict()
        if scaler is not None:
            engine_state["scaler"] = scaler.state_dict()
        torch.save(engine_state, cur_checkpoint_dir / "engine.pt")


def load_checkpoint(
    mesh: DeviceMesh,
    ckpt_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    scaler: GradientScaler = None,
    prefix: str = "",
) -> dict[str, int] | None:
    """Load checkpoint from disk.

    Args:
        mesh: Device mesh for distributed training.
        ckpt_path: Path to checkpoint directory.
        model: Model to load state into.
        optimizer: (Optional) Optimizer to load state into.
        scheduler: (Optional) Learning rate scheduler to load state into.
        scaler: (Optional) Gradient scaler to load state into.
        prefix: (Optional) Prefix to load state into.

    Returns:
        dict: Engine state containing 'step', 'epoch', '_accumulate_step'.
    """
    backend = _infer_checkpoint_backend(mesh)

    if backend == "ddp":
        if hasattr(model, "module"):
            model.module.load_state_dict(torch.load(Path(ckpt_path) / f"{prefix}model.pt", map_location="cpu"))
        else:
            simple_info("DDP backend detected but model is not wrapped!")
            model.load_state_dict(torch.load(Path(ckpt_path) / f"{prefix}model.pt", map_location="cpu"))
        if optimizer is not None:
            optimizer.load_state_dict(torch.load(Path(ckpt_path) / f"{prefix}optimizer.pt", map_location="cpu"))
    else:
        options = StateDictOptions(full_state_dict=False, cpu_offload=True, strict=False)
        model_sd = get_model_state_dict(model, options=options)
        state_dict = {
            "model": model_sd,
        }

        if optimizer is not None:
            optim_sd = get_optimizer_state_dict(model, optimizer, options=options)
            state_dict["optimizer"] = optim_sd

        if prefix == "":
            dcp_save_path = Path(ckpt_path)
        else:
            dcp_save_path = Path(ckpt_path) / f"{prefix}model"

        dcp.load(
            state_dict,
            checkpoint_id=str(dcp_save_path),
            process_group=_get_checkpoint_process_group(),
        )
        set_model_state_dict(model, state_dict["model"])
        if optimizer is not None:
            missing_count = _fill_missing_optimizer_states(model, state_dict["optimizer"])
            if missing_count > 0 and is_main_process():
                simple_info(f"Filled {missing_count} missing optimizer state entries before checkpoint load.")
            set_optimizer_state_dict(model, optimizer, state_dict["optimizer"])

    if prefix != "":
        return None

    # Load scheduler and scaler if available in checkpoint
    engine_state = torch.load(Path(ckpt_path) / "engine.pt", map_location="cpu")

    if scheduler is not None and "scheduler" in engine_state:
        scheduler.load_state_dict(engine_state["scheduler"])
    if scaler is not None and "scaler" in engine_state:
        scaler.load_state_dict(engine_state["scaler"])

    # Restore RNG states if available in checkpoint
    rank = get_rank()
    rng_path = Path(ckpt_path) / "engine" / f"rank_{rank}.pt"
    if os.path.exists(rng_path):
        rng_state = torch.load(
            rng_path,
            map_location="cpu",
            weights_only=False,
        )
        if "torch_rng_state" in rng_state:
            torch.set_rng_state(rng_state["torch_rng_state"])
        accelerator_type = rng_state.get("accelerator_type")
        accelerator_rng_state = rng_state.get("accelerator_rng_state")
        if accelerator_type is None and "cuda_rng_state" in rng_state:
            # Backward compatibility for older CUDA checkpoints.
            accelerator_type = "cuda"
            accelerator_rng_state = rng_state["cuda_rng_state"]
        _set_accelerator_rng_state(accelerator_type, accelerator_rng_state)
        if "python_rng_state" in rng_state:
            random.setstate(rng_state["python_rng_state"])
        if "numpy_rng_state" in rng_state:
            np.random.set_state(rng_state["numpy_rng_state"])

    return {
        "step": engine_state["step"],
        "epoch": engine_state["epoch"],
        "_accumulate_step": engine_state["_accumulate_step"],
    }
