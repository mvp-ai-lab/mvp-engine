import os

import torch
import torch.distributed as dist

from mvp_engine.utils.log import simple_info
from mvp_engine.utils.misc import get_device


def _get_distributed_backend(device_type: str) -> str:
    """Select the distributed backend that matches the active accelerator."""
    if device_type == "cuda":
        return "nccl"
    if device_type == "npu":
        return "hccl"
    return "gloo"


def initialize_process_group():
    """Initialize the torch.distributed process group based on env variables.

    Uses RANK and WORLD_SIZE from the environment, selects the device for the
    current rank, and initializes a matching process group via env://.
    """
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    device = get_device(rank)

    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "cpu":
        simple_info("Using CPU for distributed training.")
        return
    elif device.type == "npu":
        torch.npu.set_device(device)

    if world_size <= 0:
        raise ValueError("WORLD_SIZE must be greater than 0 for distributed training.")

    simple_info(
        f"Parallel Process Group Initializing: [bold]rank {rank}/{world_size}[/bold] on [yellow]{device}[/yellow]..."
    )

    dist.init_process_group(
        backend=_get_distributed_backend(device.type),
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    simple_info(f"Process Group Initialized [bold]rank {rank}/{world_size}[/bold] on [yellow]{device}[/yellow]")
