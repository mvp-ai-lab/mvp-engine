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


def build_mesh_shape_and_names(
    parallel_backend: str,
    world_size: int,
    mesh_cfg: Optional[dict[str, Any]] = None,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Build mesh shape and dim names for the given parallel backend.

    Args:
        parallel_backend: Distributed backend type, supports ``ddp`` and ``fsdp2``.
        world_size: Total number of distributed processes.
        mesh_cfg: Optional mesh config with ``dp_size``, ``fsdp2_size``, ``tp_size``.

    Returns:
        A tuple ``(mesh_shape, mesh_dim_names)``.
    """
    if parallel_backend == "ddp":
        return (world_size,), ("ddp",)

    if parallel_backend != "fsdp2":
        raise ValueError(f"Unsupported parallel backend: {parallel_backend}")

    mesh_cfg = mesh_cfg or {}

    tp_size = int(mesh_cfg.get("tp_size", 1))
    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}.")
    if world_size % tp_size != 0:
        raise ValueError(f"WORLD_SIZE({world_size}) must be divisible by tp_size({tp_size}).")

    fsdp2_size = int(mesh_cfg.get("fsdp2_size", 1))
    if fsdp2_size < 1:
        raise ValueError(f"fsdp2_size must be >= 1, got {fsdp2_size}.")
    if world_size % (fsdp2_size * tp_size) != 0:
        raise ValueError(f"WORLD_SIZE({world_size}) must be divisible by fsdp2_size({fsdp2_size})*tp_size({tp_size}).")

    dp_size = int(mesh_cfg.get("dp_size", world_size // (tp_size * fsdp2_size)))
    if dp_size * fsdp2_size * tp_size != world_size:
        raise ValueError(
            "Invalid mesh configuration: "
            f"dp_size({dp_size}) * fsdp2_size({fsdp2_size}) * tp_size({tp_size}) "
            f"must equal world_size({world_size})."
        )

    return (dp_size, fsdp2_size, tp_size), ("dp", "fsdp2", "tp")


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
