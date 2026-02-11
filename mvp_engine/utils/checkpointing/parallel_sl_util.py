from pathlib import Path

import torch
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

from mvp_engine.distributed.utils import is_main_process
from mvp_engine.engine.engine import Engine
from mvp_engine.utils.training import GradientScaler


def parallel_save(
    backend: str,
    mesh: DeviceMesh,
    cur_checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    scaler: GradientScaler = None,
    step: int = 0,
    epoch: int = 0,
    _accumulate_step: int = 0,
):
    if backend == "ddp" and not is_main_process():
        return

    if backend == "ddp":
        torch.save(
            model.module.state_dict(),
            cur_checkpoint_dir / "model.pt",
        )
        torch.save(
            optimizer.state_dict(),
            cur_checkpoint_dir / "optimizer.pt",
        )
        if scheduler is not None:
            torch.save(
                scheduler.state_dict(),
                cur_checkpoint_dir / "scheduler.pt",
            )
        if scaler is not None:
            torch.save(
                scaler.state_dict(),
                cur_checkpoint_dir / "scaler.pt",
            )
        torch.save(
            {
                "step": step,
                "epoch": epoch,
                "_accumulate_step": _accumulate_step,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            },
            cur_checkpoint_dir / "engine.pt",
        )
    else:
        options = StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
        )
        model_sd = get_model_state_dict(model, options=options)
        optim_sd = get_optimizer_state_dict(model, optimizer, options=options)

        state_dict = {
            "model": model_sd,
            "optimizer": optim_sd,
        }

        writer = dcp.FileSystemWriter(cur_checkpoint_dir)
        dcp.save(
            state_dict,
            checkpoint_id=str(cur_checkpoint_dir),
            process_group=mesh["fsdp2"].get_group(),
            storage_writer=writer,
        )

        if is_main_process():
            meta_dict = {
                "step": step,
                "epoch": epoch,
                "_accumulate_step": _accumulate_step,
            }
            if scheduler is not None:
                meta_dict["scheduler"] = scheduler.state_dict()
            if scaler is not None:
                meta_dict["scaler"] = scaler.state_dict()
            torch.save(meta_dict, cur_checkpoint_dir / "engine.pt")


def parallel_load(
    engine_instance: Engine,
    backend: str,
    mesh: DeviceMesh,
    ckpt_path: str,
):
    if backend == "ddp":
        engine_instance.model.module.load_state_dict(torch.load(Path(ckpt_path) / "model.pt", map_location="cpu"))
        engine_instance.optimizer.load_state_dict(torch.load(Path(ckpt_path) / "optimizer.pt", map_location="cpu"))
        engine_instance.scheduler.load_state_dict(torch.load(Path(ckpt_path) / "scheduler.pt", map_location="cpu"))
        engine_instance.scaler.load_state_dict(torch.load(Path(ckpt_path) / "scaler.pt", map_location="cpu"))
        engine_state = torch.load(Path(ckpt_path) / "engine.pt", map_location="cpu")
        engine_instance.step = engine_state["step"]
        engine_instance.epoch = engine_state["epoch"]
        engine_instance._accumulate_step = engine_state["_accumulate_step"]
        torch.set_rng_state(engine_state["rng_state"])
        torch.cuda.set_rng_state_all(engine_state["cuda_rng_state"])
    else:
        options = StateDictOptions(full_state_dict=False, cpu_offload=True, strict=False)
        model_sd = get_model_state_dict(engine_instance.model, options=options)
        optim_sd = get_optimizer_state_dict(engine_instance.model, engine_instance.optimizer, options=options)
        state_dict = {
            "model": model_sd,
            "optimizer": optim_sd,
        }

        dcp.load(state_dict, checkpoint_id=ckpt_path, process_group=mesh["fsdp2"].get_group())
        set_model_state_dict(engine_instance.model, state_dict["model"])
        set_optimizer_state_dict(engine_instance.model, engine_instance.optimizer, state_dict["optimizer"])

        engine_state = torch.load(Path(ckpt_path) / "engine.pt", map_location="cpu")
        engine_instance.scheduler.load_state_dict(engine_state["scheduler"])
        engine_instance.scaler.load_state_dict(engine_state["scaler"])
        engine_instance.step = engine_state["step"]
        engine_instance.epoch = engine_state["epoch"]
        engine_instance._accumulate_step = engine_state["_accumulate_step"]
