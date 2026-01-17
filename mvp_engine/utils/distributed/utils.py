import pickle
from typing import Any, Optional

import torch
import torch.distributed as dist


def get_rank() -> int:
    """Return the rank of the current process in the distributed group."""
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Return the number of processes in the distributed group."""
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """Check if the current process is the main process (rank 0)."""
    return get_rank() == 0


def broadcast_from_main(obj: Any, group: Optional[dist.ProcessGroup] = None) -> Any:
    """Broadcast an object from the main process to all other processes.
    
    Args:
        obj: The object to broadcast from rank 0.
        group: Optional process group. If None, creates a new gloo group.
    
    Returns:
        The broadcasted object on all ranks.
    """
    if not dist.is_initialized():
        return obj
    
    rank = get_rank()
    world_size = get_world_size()
    
    # Create a new gloo group if none provided
    if group is None:
        group = dist.new_group(backend="gloo", ranks=list(range(world_size)))
    
    if rank == 0:
        tensor = torch.tensor(bytearray(pickle.dumps(obj)), dtype=torch.uint8)
        size = torch.tensor([tensor.numel()], dtype=torch.long)
    else:
        size = torch.tensor([0], dtype=torch.long)

    dist.broadcast(size, src=0, group=group)

    if rank != 0:
        tensor = torch.empty(size.item(), dtype=torch.uint8)

    dist.broadcast(tensor, src=0, group=group)

    if rank != 0:
        obj = pickle.loads(tensor.cpu().numpy().tobytes())

    return obj