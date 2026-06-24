import fcntl
import os
import pickle
import socket
import struct
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


def get_world_size(group: dist.ProcessGroup | None = None) -> int:
    """Return the number of processes in the distributed group."""
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


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
