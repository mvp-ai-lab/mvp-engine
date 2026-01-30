import os

import torch
import torch.distributed as dist

from ..log import simple_info
from ..misc import get_device


def prepare_ddp():
    """Prepare DistributedDataParallel (DDP) if distributed training is enabled."""

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    device = get_device(rank)

    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "cpu":
        pass
    elif device.type == "npu":
        torch.npu.set_device(device)

    if world_size <= 1:
        return

    simple_info(
        f"Setup DDP Initialization: [bold]rank {rank}/{world_size}[/bold] on [yellow]{device}[/yellow]..."
    )

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    message = f"DDP Initialized [bold]rank {rank}/{world_size}[/bold] on [yellow]{device}[/yellow]"
    simple_info(message)
