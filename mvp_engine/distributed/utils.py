import pickle
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


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


def infer_parallel_backend(mesh_cfg: Optional[dict[str, Any]] = None) -> str:
    """Infer the parallel backend from mesh config.

    Rules:
    - default to ``ddp`` when mesh is empty or absent;
    - if ``ddp_size`` is present, force ``ddp`` and ignore other mesh sizes;
    - otherwise use ``fsdp2``.
    """
    mesh_cfg = mesh_cfg or {}

    if mesh_cfg.get("ddp_size", None) is not None:
        return "ddp"
    if mesh_cfg:
        return "fsdp2"
    return "ddp"


def build_mesh_shape_and_names(
    parallel_backend: str,
    world_size: int,
    mesh_cfg: Optional[dict[str, Any]] = None,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Build mesh shape and dim names for the given parallel backend.

    Args:
        parallel_backend: Distributed backend type, supports ``ddp`` and ``fsdp2``.
        world_size: Total number of distributed processes.
        mesh_cfg: Optional mesh config with ``ddp_size``, ``dp_size``, ``fsdp2_size``,
            and ``tp_size``.

    Returns:
        A tuple ``(mesh_shape, mesh_dim_names)``.
    """
    if parallel_backend == "ddp":
        return (world_size,), ("ddp",)

    if parallel_backend != "fsdp2":
        raise ValueError(f"Unsupported parallel backend: {parallel_backend}")

    mesh_cfg = mesh_cfg or {}

    tp_size = int(mesh_cfg.get("tp_size", 1))
    fsdp2_size = int(mesh_cfg.get("fsdp2_size", 1))

    if tp_size == -1 and fsdp2_size == -1:
        raise ValueError("Only one of tp_size and fsdp2_size can be -1.")

    force_global_dp = False
    if tp_size == -1:
        tp_size = world_size
        force_global_dp = True
    elif tp_size < 1:
        raise ValueError(f"tp_size must be >= 1 or -1, got {tp_size}.")
    if world_size % tp_size != 0:
        raise ValueError(f"WORLD_SIZE({world_size}) must be divisible by tp_size({tp_size}).")

    if fsdp2_size == -1:
        fsdp2_size = world_size
        force_global_dp = True
    elif fsdp2_size < 1:
        raise ValueError(f"fsdp2_size must be >= 1 or -1, got {fsdp2_size}.")
    if world_size % (fsdp2_size * tp_size) != 0:
        raise ValueError(f"WORLD_SIZE({world_size}) must be divisible by fsdp2_size({fsdp2_size})*tp_size({tp_size}).")

    dp_size = 1 if force_global_dp else int(mesh_cfg.get("dp_size", world_size // (tp_size * fsdp2_size)))
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


def is_dtensor_tensor(tensor: torch.Tensor) -> bool:
    """Best-effort DTensor detection with a duck-typing fallback for tests."""
    if DTensor is not None and isinstance(tensor, DTensor):
        return True
    return hasattr(tensor, "to_local") and hasattr(tensor, "placements") and hasattr(tensor, "device_mesh")


def has_dtensor_parameters(parameters) -> bool:
    """Return True when any parameter tensor is backed by DTensor."""
    return any(is_dtensor_tensor(parameter) for parameter in parameters)


def to_local_dense_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the dense local tensor used for DTensor-aware math."""
    local_tensor = tensor.to_local() if is_dtensor_tensor(tensor) else tensor
    if getattr(local_tensor, "is_sparse", False):
        return local_tensor.coalesce().values()
    return local_tensor


def get_grad_scalar_device(parameters) -> torch.device:
    """Return the device used to host scalar norm accumulators."""
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        return to_local_dense_tensor(grad).device
    return torch.device("cpu")


def dtensor_reduce_groups(tensor: torch.Tensor) -> list:
    """Return process groups for shard/partial mesh dimensions of a DTensor."""
    groups = []
    device_mesh = tensor.device_mesh
    placements = tensor.placements

    for mesh_dim, placement in enumerate(placements):
        if not (placement.is_shard() or placement.is_partial()):
            continue
        if not hasattr(device_mesh, "get_group"):
            groups.append(None)
            continue

        try:
            group = device_mesh.get_group(mesh_dim=mesh_dim)
        except TypeError:
            try:
                group = device_mesh.get_group(mesh_dim)
            except TypeError:
                group = device_mesh.get_group()
        groups.append(group)

    return groups


def reduce_dtensor_scalar(tensor: torch.Tensor, dtensor_grad: torch.Tensor, reduce_op) -> torch.Tensor:
    """Reduce a local DTensor-derived scalar across shard/partial mesh dimensions."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor

    reduced = tensor
    for group in dtensor_reduce_groups(dtensor_grad):
        dist.all_reduce(reduced, op=reduce_op, group=group)
    return reduced


def scale_dtensor_grad_(grad: torch.Tensor, scale: torch.Tensor) -> None:
    """Scale a gradient tensor in-place, including DTensor local shards."""
    local_grad = grad.to_local() if is_dtensor_tensor(grad) else grad
    if getattr(local_grad, "is_sparse", False):
        local_grad = local_grad.coalesce()
        local_grad.values().mul_(scale.to(device=local_grad.device, dtype=local_grad.dtype))
        return
    local_grad.mul_(scale.to(device=local_grad.device, dtype=local_grad.dtype))
