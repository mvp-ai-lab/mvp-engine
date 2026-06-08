import fcntl
import os
import pickle
import socket
import struct
from itertools import product
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

MESH_DIM_REPLICATE = "replicate"
MESH_DIM_SHARD = "shard"
MESH_DIM_CONTEXT = "context"
MESH_DIM_TENSOR = "tensor"
MVP_ENGINE_MESH_ORDER = (MESH_DIM_REPLICATE, MESH_DIM_SHARD, MESH_DIM_CONTEXT, MESH_DIM_TENSOR)
MODEL_PARALLEL_DIMS = {MESH_DIM_TENSOR, MESH_DIM_CONTEXT}


def get_rank() -> int:
    """Return the rank of the current process in the distributed group."""
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """Return the local rank of the current process on this node."""
    return int(os.getenv("LOCAL_RANK", str(get_rank())))


def _get_ipv4_for_interface(interface: str) -> Optional[str]:
    """Return the IPv4 address for an interface, or ``None`` when unavailable."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        request = struct.pack("256s", interface[:15].encode("utf-8"))
        response = fcntl.ioctl(sock.fileno(), 0x8915, request)  # SIOCGIFADDR
        return socket.inet_ntoa(response[20:24])
    except OSError:
        return None
    finally:
        sock.close()


def _is_interface_up(interface: str) -> bool:
    """Return whether a network interface is marked ``up`` by the kernel."""
    try:
        with open(f"/sys/class/net/{interface}/operstate") as file:
            return file.read().strip() == "up"
    except OSError:
        return False


def guess_socket_interface() -> Optional[str]:
    """Guess a routable network interface for distributed communication."""
    master_addr = os.getenv("MASTER_ADDR")
    remote_ip = None

    if master_addr:
        try:
            remote_ip = socket.gethostbyname(master_addr)
        except OSError:
            remote_ip = None

    if remote_ip and not remote_ip.startswith("127."):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # UDP connect selects the outbound route without sending packets.
            sock.connect((remote_ip, 1))
            local_ip = sock.getsockname()[0]
            for _, interface in socket.if_nameindex():
                if _get_ipv4_for_interface(interface) == local_ip:
                    return interface
        except OSError:
            pass
        finally:
            sock.close()

    candidates = []
    for _, interface in socket.if_nameindex():
        if interface == "lo":
            continue
        address = _get_ipv4_for_interface(interface)
        if not address or address.startswith("127."):
            continue
        score = 0
        if _is_interface_up(interface):
            score += 100
        if interface.startswith(("bond", "ib", "en", "eth")):
            score += 10
        candidates.append((score, interface))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def configure_distributed_socket_ifnames(device_type: str) -> dict[str, str]:
    """Set socket interface env vars when the user has not configured them."""
    env_names = ["GLOO_SOCKET_IFNAME"]
    if device_type == "cuda":
        env_names.append("NCCL_SOCKET_IFNAME")

    configured = {}
    guessed_interface = None

    for env_name in env_names:
        env_value = os.getenv(env_name)
        if env_value:
            configured[env_name] = env_value
            continue
        if guessed_interface is None:
            guessed_interface = guess_socket_interface()
        if guessed_interface:
            os.environ[env_name] = guessed_interface
            configured[env_name] = guessed_interface

    return configured


def get_world_size() -> int:
    """Return the number of processes in the distributed group."""
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def initialize_device_mesh(
    device_type: str,
    mesh_cfg: dict[str, int],
) -> DeviceMesh:
    """Initialize the named DeviceMesh used by mvp-engine distributed runtimes."""
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

    from mvp_engine.utils.log import simple_info

    simple_info(
        f"Device Mesh initializing: {' / '.join(mesh_dim_names)} ({' / '.join([str(x) for x in mesh_shape])})..."
    )

    return init_device_mesh(
        device_type=device_type,
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
    )


def get_mesh_dim_names(device_mesh: DeviceMesh) -> tuple[str, ...]:
    """Return named DeviceMesh dimensions, or an empty tuple for unnamed meshes."""
    return tuple(getattr(device_mesh, "mesh_dim_names", ()) or ())


def get_mesh_shape(device_mesh: DeviceMesh) -> tuple[int, ...]:
    """Return the full DeviceMesh shape as plain ints."""
    return tuple(int(dim_size) for dim_size in device_mesh.shape)


def get_mesh_identity_key(device_mesh: DeviceMesh) -> tuple:
    """Return a stable key that identifies mesh topology and dimension names."""
    mesh = device_mesh.mesh.detach().cpu()
    return (
        device_mesh.device_type,
        tuple(int(dim_size) for dim_size in mesh.shape),
        tuple(int(rank) for rank in mesh.reshape(-1).tolist()),
        get_mesh_dim_names(device_mesh),
    )


def get_mesh_dim_size(device_mesh: DeviceMesh, dim_name: str) -> int:
    """Return the size of a named mesh dimension."""
    return int(device_mesh[dim_name].size())


def get_mesh_dim_group(device_mesh: DeviceMesh, mesh_dim: int | str) -> dist.ProcessGroup:
    """Return the process group for a mesh dimension index or name."""
    return device_mesh.get_group(mesh_dim)


def get_mesh_reduce_device(device_mesh: DeviceMesh) -> torch.device:
    """Return a device compatible with the mesh process group collectives."""
    device_type = device_mesh.device_type
    if device_type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(device_type)


def is_mesh_dim_active(device_mesh: DeviceMesh, dim_name: str) -> bool:
    """Return whether a named mesh dimension exists and has size greater than one."""
    return dim_name in get_mesh_dim_names(device_mesh) and get_mesh_dim_size(device_mesh, dim_name) > 1


def get_replicate_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """Return the replicate-only submesh used by DDP."""
    return device_mesh[MESH_DIM_REPLICATE]


def get_sharded_data_parallel_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """Return the replicate/shard submesh used by FSDP2."""
    return device_mesh[MESH_DIM_REPLICATE, MESH_DIM_SHARD]


def get_tensor_parallel_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """Return the tensor-parallel submesh."""
    return device_mesh[MESH_DIM_TENSOR]


def infer_mesh_parallel_backend(device_mesh: DeviceMesh) -> str:
    """Infer whether a DeviceMesh should use the DDP or FSDP2 checkpoint path."""
    mesh_dim_names = get_mesh_dim_names(device_mesh)
    if mesh_dim_names:
        if any(
            is_mesh_dim_active(device_mesh, dim_name)
            for dim_name in (MESH_DIM_SHARD, MESH_DIM_TENSOR, MESH_DIM_CONTEXT)
        ):
            return "fsdp2"
        return "ddp"

    mesh_shape = get_mesh_shape(device_mesh)
    if len(mesh_shape) <= 1:
        return "ddp"
    if any(dim_size > 1 for dim_size in mesh_shape[1:]):
        return "fsdp2"
    return "ddp"


class NamedDeviceMeshAdapter:
    """Adapt PyTorch named DeviceMesh for callers expecting name-based mesh access."""

    def __init__(self, device_mesh: DeviceMesh) -> None:
        self.device_mesh = device_mesh
        self.mesh_dim_names = get_mesh_dim_names(device_mesh)

    def size(self, dim_name: str) -> int:
        """Return the size of a named mesh dimension."""
        return get_mesh_dim_size(self.device_mesh, dim_name)

    def get_local_rank(self, dim_name: str) -> int:
        """Return the local rank within a named mesh dimension."""
        return int(self.device_mesh.get_local_rank(dim_name))


def get_mesh_world_size(device_mesh: DeviceMesh, dim_names: tuple[str, ...]) -> int:
    """Return the product of the requested mesh dimension sizes."""
    if not dim_names:
        return 1

    world_size = 1
    for dim_name in dim_names:
        world_size *= get_mesh_dim_size(device_mesh, dim_name)
    return int(world_size)


def get_data_parallel_dim_names(device_mesh: DeviceMesh) -> tuple[str, ...]:
    """Return mesh dimensions that contribute independent training samples."""
    return tuple(dim_name for dim_name in get_mesh_dim_names(device_mesh) if dim_name not in MODEL_PARALLEL_DIMS)


def get_data_parallel_world_size(device_mesh: DeviceMesh) -> int:
    """Return the mesh world size that contributes samples to one optimizer step."""
    return get_mesh_world_size(device_mesh, get_data_parallel_dim_names(device_mesh))


def get_data_parallel_rank(device_mesh: DeviceMesh) -> int:
    """Return this process's rank over data-parallel mesh dimensions."""
    group = get_data_parallel_group(device_mesh)
    if group is None:
        return 0
    return int(dist.get_rank(group=group))


def get_data_parallel_group(device_mesh: DeviceMesh) -> Optional[dist.ProcessGroup]:
    """Return the process group that contributes independent samples."""
    if not dist.is_available() or not dist.is_initialized():
        return None

    dp_dim_names = get_data_parallel_dim_names(device_mesh)
    if not dp_dim_names:
        return None

    if get_mesh_world_size(device_mesh, dp_dim_names) <= 1:
        return None

    if len(dp_dim_names) == 1:
        return device_mesh[dp_dim_names[0]].get_group()

    flat_name = "_".join(dp_dim_names)
    return device_mesh[dp_dim_names]._flatten(flat_name).get_group()


def get_context_parallel_size(device_mesh: DeviceMesh) -> int:
    """Return the active context-parallel mesh size."""
    if MESH_DIM_CONTEXT not in get_mesh_dim_names(device_mesh):
        return 1
    return get_mesh_dim_size(device_mesh, MESH_DIM_CONTEXT)


def get_context_parallel_rank(device_mesh: DeviceMesh) -> int:
    """Return this rank's coordinate inside the context mesh dimension."""
    if get_context_parallel_size(device_mesh) <= 1:
        return 0
    return int(device_mesh.get_local_rank(MESH_DIM_CONTEXT))


def get_context_parallel_group(device_mesh: DeviceMesh) -> Optional[dist.ProcessGroup]:
    """Return the active context-parallel process group."""
    if get_context_parallel_size(device_mesh) <= 1:
        return None
    return device_mesh[MESH_DIM_CONTEXT].get_group()


def get_mesh_group_ranks(device_mesh: DeviceMesh, dim_names: tuple[str, ...]) -> list[list[int]]:
    """Return rank groups spanning the requested mesh dimensions."""
    mesh_dim_names = get_mesh_dim_names(device_mesh)
    missing_dims = [dim_name for dim_name in dim_names if dim_name not in mesh_dim_names]
    if missing_dims:
        raise ValueError(f"DeviceMesh does not contain mesh dimensions: {missing_dims}.")

    selected = {mesh_dim_names.index(dim_name) for dim_name in dim_names}
    selected_indices = tuple(index for index in range(len(mesh_dim_names)) if index in selected)
    fixed_indices = tuple(index for index in range(len(mesh_dim_names)) if index not in selected)

    mesh = device_mesh.mesh.detach().cpu()
    fixed_ranges = [range(int(mesh.shape[index])) for index in fixed_indices]
    selected_ranges = [range(int(mesh.shape[index])) for index in selected_indices]

    rank_groups: list[list[int]] = []
    for fixed_coords in product(*fixed_ranges):
        ranks: list[int] = []
        for selected_coords in product(*selected_ranges):
            mesh_index = [0] * len(mesh_dim_names)
            for dim_index, coord in zip(fixed_indices, fixed_coords):
                mesh_index[dim_index] = coord
            for dim_index, coord in zip(selected_indices, selected_coords):
                mesh_index[dim_index] = coord
            ranks.append(int(mesh[tuple(mesh_index)].item()))
        rank_groups.append(ranks)
    return rank_groups


def validate_mvp_engine_mesh_order(device_mesh: DeviceMesh) -> None:
    """Validate that named mesh dimensions follow the mvp-engine runtime order."""
    mesh_dim_names = get_mesh_dim_names(device_mesh)
    if mesh_dim_names != MVP_ENGINE_MESH_ORDER:
        raise ValueError(
            "Long-context TP/CP compatibility requires mvp-engine mesh dimension order "
            f"{MVP_ENGINE_MESH_ORDER}, got {mesh_dim_names}."
        )


def configure_long_context_process_groups(device_mesh: DeviceMesh, config: dict | None) -> None:
    """Configure yunchang Ulysses/Ring process groups from the mvp-engine DeviceMesh."""
    if not dist.is_available() or not dist.is_initialized():
        return

    validate_mvp_engine_mesh_order(device_mesh)
    config = config or {}
    ulysses_degree = int(config.get("ulysses_degree", 1))
    ring_degree = int(config.get("ring_degree", 1))
    use_ulysses_low = bool(config.get("use_ulysses_low", True))

    context_groups = get_mesh_group_ranks(device_mesh, (MESH_DIM_CONTEXT,))
    ulysses_pg, ring_pg = create_yunchang_process_groups(
        context_groups,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        use_ulysses_low=use_ulysses_low,
    )

    try:
        from yunchang.globals import PROCESS_GROUP
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Long-context attention requires `pip install 'mvp_engine[long-context]'`.") from exc

    PROCESS_GROUP.ULYSSES_PG = ulysses_pg
    PROCESS_GROUP.RING_PG = ring_pg

    from mvp_engine.utils.log import logger

    logger.info(
        "Initialized long-context process groups from DeviceMesh "
        f"(context={get_context_parallel_size(device_mesh)}, "
        f"ulysses={ulysses_degree}, ring={ring_degree}, use_ulysses_low={use_ulysses_low})."
    )


def create_yunchang_process_groups(
    context_groups: list[list[int]],
    *,
    ulysses_degree: int,
    ring_degree: int,
    use_ulysses_low: bool,
) -> tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """Create yunchang Ulysses and Ring groups and return this rank's groups."""
    current_rank = dist.get_rank()
    selected_ulysses_pg = None
    selected_ring_pg = None

    for context_ranks in context_groups:
        ulysses_groups, ring_groups = get_yunchang_sequence_parallel_group_ranks(
            context_ranks,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            use_ulysses_low=use_ulysses_low,
        )
        if use_ulysses_low:
            selected_ulysses_pg = create_process_group_set_and_select(
                ulysses_groups,
                current_rank,
                selected_ulysses_pg,
            )
            selected_ring_pg = create_process_group_set_and_select(ring_groups, current_rank, selected_ring_pg)
        else:
            selected_ring_pg = create_process_group_set_and_select(ring_groups, current_rank, selected_ring_pg)
            selected_ulysses_pg = create_process_group_set_and_select(
                ulysses_groups,
                current_rank,
                selected_ulysses_pg,
            )

    if selected_ulysses_pg is None or selected_ring_pg is None:
        raise RuntimeError(f"Failed to create yunchang process groups for rank {current_rank}.")
    return selected_ulysses_pg, selected_ring_pg


def get_yunchang_sequence_parallel_group_ranks(
    context_ranks: list[int],
    *,
    ulysses_degree: int,
    ring_degree: int,
    use_ulysses_low: bool,
) -> tuple[list[list[int]], list[list[int]]]:
    """Return Ulysses and Ring rank groups for one context-parallel group."""
    if ulysses_degree * ring_degree != len(context_ranks):
        raise ValueError(
            "Long-context attention requires ulysses_degree * ring_degree to match each context group size."
        )

    if use_ulysses_low:
        ulysses_groups = [
            context_ranks[index * ulysses_degree : (index + 1) * ulysses_degree] for index in range(ring_degree)
        ]
        ring_groups = [context_ranks[index::ulysses_degree] for index in range(ulysses_degree)]
    else:
        ring_groups = [
            context_ranks[index * ring_degree : (index + 1) * ring_degree] for index in range(ulysses_degree)
        ]
        ulysses_groups = [context_ranks[index::ring_degree] for index in range(ring_degree)]
    return ulysses_groups, ring_groups


def create_process_group_set_and_select(
    rank_groups: list[list[int]],
    current_rank: int,
    selected_pg: dist.ProcessGroup | None,
) -> dist.ProcessGroup | None:
    """Create all process groups in a set and return the one containing this rank."""
    for ranks in rank_groups:
        group = dist.new_group(ranks=ranks)
        if current_rank in ranks:
            selected_pg = group
    return selected_pg


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
