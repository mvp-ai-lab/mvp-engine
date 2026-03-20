import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from mvp_engine.utils.log import simple_info


def initialize_device_mesh(
    device_type: str,
    mesh_cfg: dict[str, int],
) -> DeviceMesh:
    """
    Initialize device mesh for distributed training, compatible with both DDP and FSDP2.

    Args:
        device_type: Device type, "cuda" or "cpu"
        mesh_cfg: Dictionary specifying mesh dimensions and their sizes, e.g. {"replicate": world_size} for DDP.

    Returns:
        DeviceMesh: Initialized device mesh
    """
    world_size = dist.get_world_size()

    others = []
    to_be_infered_name = ""
    for dim_name, dim_size in mesh_cfg.items():
        if dim_size != -1:
            others.append(dim_size)
        else:
            to_be_infered_name = dim_name
    if len(others) < len(mesh_cfg) - 1:
        raise ValueError(
            "Insufficient dimension sizes specified for device mesh initialization, only one dimension can be inferred."
        )

    if to_be_infered_name:
        inferred_size = world_size // (1 if not others else torch.prod(torch.tensor(others)).item())
        mesh_cfg[to_be_infered_name] = inferred_size

    mesh_shape = list(mesh_cfg.values())
    mesh_dim_names = list(mesh_cfg.keys())

    simple_info(
        f"Device Mesh initializing: {' / '.join(mesh_dim_names)} ({' / '.join([str(x) for x in mesh_shape])})..."
    )

    device_mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
    )

    return device_mesh
