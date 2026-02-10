from typing import Optional, Tuple

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from mvp_engine.utils.log import simple_info


def initialize_device_mesh(
    device_type: str = "cuda",
    mesh_shape: Optional[Tuple[int, ...]] = None,
    mesh_dim_names: Optional[Tuple[str, ...]] = None,
) -> DeviceMesh:
    """
    Initialize device mesh for distributed training, compatible with both DDP and FSDP2.

    Args:
        device_type: Device type, "cuda" or "cpu"
        mesh_shape: Shape of the device mesh. If None, will create a 1D mesh with world_size.
                   For FSDP2, can use 2D mesh like (dp_size, fsdp_size).
        mesh_dim_names: Names for each mesh dimension, e.g., ("dp", "fsdp") for 2D mesh.

    Returns:
        DeviceMesh: Initialized device mesh

    Examples:
        # DDP mode (1D mesh)
        mesh = initialize_device_mesh("cuda")

        # FSDP2 mode (1D mesh)
        mesh = initialize_device_mesh("cuda", mesh_shape=(8,), mesh_dim_names=("fsdp",))
    """
    world_size = dist.get_world_size()

    # Default to 1D mesh with world_size for DDP compatibility
    if mesh_shape is None:
        mesh_shape = (world_size,)

    # Validate mesh shape
    mesh_size = 1
    for dim in mesh_shape:
        mesh_size *= dim

    if mesh_size != world_size:
        raise ValueError(f"Product of mesh_shape {mesh_shape} must equal world_size {world_size}")

    # Create device mesh
    if mesh_dim_names is None:
        if len(mesh_shape) == 1:
            mesh_dim_names = ("dp",)  # Default name for 1D mesh
        else:
            raise ValueError("mesh_dim_names must be provided for multi-dimensional meshes")

    simple_info(f"Device Mesh initializaing: {mesh_dim_names} ({mesh_shape})...")

    device_mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
    )

    return device_mesh
