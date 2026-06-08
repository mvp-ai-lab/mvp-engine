"""Long-context attention helpers built around yunchang USP attention."""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from mvp_engine.distributed.utils import (
    configure_long_context_process_groups,
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_size,
)
from mvp_engine.utils.log import logger

_HOOK_ATTR = "APPLY_LONG_CONTEXT_ATTENTION"
_BASIC_RING_IMPL_TYPES = {"basic", "basic_pytorch", "basic_flashinfer", "basic_npu"}


def is_long_context_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return whether long-context attention is enabled in backend kwargs."""
    return bool((config or {}).get("enabled", False))


def prepare_long_context_attention(
    model: nn.Module,
    device_mesh: DeviceMesh,
    config: Mapping[str, Any],
) -> None:
    """Initialize yunchang process groups and apply an optional model hook."""
    _validate_long_context_config(device_mesh, config)
    _set_yunchang_process_groups(device_mesh, config)

    hook = getattr(model.__class__, _HOOK_ATTR, None)
    if hook is None:
        raise ValueError(f"{model.__class__.__name__} must define {_HOOK_ATTR} to enable long-context attention.")
    if not callable(hook):
        raise TypeError(f"{model.__class__.__name__}.{_HOOK_ATTR} must be callable, got {type(hook)}.")
    hook(model, device_mesh, dict(config))


def build_long_context_attention(config: Mapping[str, Any], *, use_pack_qkv: bool = False) -> nn.Module:
    """Build a yunchang long-context attention module from backend config."""
    yunchang = _import_yunchang()
    from yunchang.kernels import AttnType

    attn_type = AttnType.from_string(str(config.get("attn_impl", "fa")))
    kwargs = {
        "ring_impl_type": str(config.get("ring_impl_type", "basic")),
        "use_sync": bool(config.get("use_sync", False)),
        "attn_type": attn_type,
    }
    if use_pack_qkv:
        return yunchang.LongContextAttentionQKVPacked(**kwargs)
    return yunchang.LongContextAttention(**kwargs)


def extract_local_sequence(
    value: torch.Tensor,
    device_mesh: DeviceMesh,
    config: Mapping[str, Any],
    *,
    dim: int = 1,
) -> torch.Tensor:
    """Extract this context rank's local sequence shard from a full sequence tensor."""
    context_size = get_context_parallel_size(device_mesh)
    if context_size <= 1:
        return value

    context_rank = get_context_parallel_rank(device_mesh)
    ring_impl_type = str(config.get("ring_impl_type", "basic"))
    if ring_impl_type == "basic" and dim != 1:
        raise ValueError("basic long-context extraction only supports dim=1.")
    if ring_impl_type == "basic":
        return value.chunk(context_size, dim=dim)[context_rank].contiguous()

    yunchang = _import_yunchang()
    extract_fn = yunchang.EXTRACT_FUNC_DICT[ring_impl_type]
    return extract_fn(
        value,
        context_rank,
        world_size=context_size,
        rd=int(config.get("ring_degree", 1)),
        ud=int(config.get("ulysses_degree", 1)),
        dim=dim,
    )


def get_basic_sequence_offset(local_sequence_length: int, device_mesh: DeviceMesh, config: Mapping[str, Any]) -> int:
    """Return the global sequence offset for contiguous basic ring extraction."""
    if str(config.get("ring_impl_type", "basic")) not in _BASIC_RING_IMPL_TYPES:
        raise ValueError("Only basic ring extraction has a contiguous per-rank sequence offset.")
    return get_context_parallel_rank(device_mesh) * int(local_sequence_length)


def get_local_sequence_position_indices(
    global_sequence_length: int,
    device_mesh: DeviceMesh,
    config: Mapping[str, Any],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return global token positions held by this context rank."""
    context_size = get_context_parallel_size(device_mesh)
    positions = torch.arange(global_sequence_length, device=device)
    if context_size <= 1:
        return positions
    if global_sequence_length % context_size != 0:
        raise ValueError("Global sequence length must be divisible by context size.")

    context_rank = get_context_parallel_rank(device_mesh)
    ring_impl_type = str(config.get("ring_impl_type", "basic"))
    if ring_impl_type in _BASIC_RING_IMPL_TYPES:
        return positions.chunk(context_size, dim=0)[context_rank]

    ulysses_degree = int(config.get("ulysses_degree", 1))
    ring_degree = int(config.get("ring_degree", 1))
    if ring_impl_type == "strip":
        if global_sequence_length % ring_degree != 0:
            raise ValueError("Strip long-context extraction requires sequence length divisible by ring_degree.")
        ordered_positions = positions.reshape(global_sequence_length // ring_degree, ring_degree)
        ordered_positions = ordered_positions.transpose(0, 1).reshape(global_sequence_length)
        return ordered_positions.chunk(context_size, dim=0)[context_rank]

    if ring_impl_type == "zigzag":
        if global_sequence_length % (2 * ring_degree) != 0:
            raise ValueError("Zigzag long-context extraction requires sequence length divisible by 2 * ring_degree.")
        ring_rank, ulysses_rank = _get_context_subranks(
            context_rank,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            use_ulysses_low=bool(config.get("use_ulysses_low", True)),
        )
        chunks = positions.chunk(2 * ring_degree, dim=0)
        local_positions = torch.cat([chunks[ring_rank], chunks[2 * ring_degree - ring_rank - 1]], dim=0)
        return local_positions.chunk(ulysses_degree, dim=0)[ulysses_rank]

    raise ValueError(f"Unsupported long-context ring_impl_type for position indices: {ring_impl_type}.")


def install_context_grad_sync(model: nn.Module, device_mesh: DeviceMesh) -> None:
    """Synchronize local-token parameter gradients across the context mesh."""
    context_size = get_context_parallel_size(device_mesh)
    if context_size <= 1:
        return
    if getattr(model, "_long_context_grad_sync_configured", False):
        return

    context_group = get_context_parallel_group(device_mesh)
    if context_group is None:
        return
    handles = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        handles.append(parameter.register_post_accumulate_grad_hook(_make_context_grad_sync_hook(context_group)))

    model._long_context_grad_sync_handles = handles
    model._long_context_grad_sync_configured = True
    logger.info(f"Installed long-context grad sync hooks on {len(handles)} parameters.")


def _validate_long_context_config(device_mesh: DeviceMesh, config: Mapping[str, Any]) -> None:
    context_size = get_context_parallel_size(device_mesh)
    ulysses_degree = int(config.get("ulysses_degree", 1))
    ring_degree = int(config.get("ring_degree", 1))
    if context_size <= 1:
        raise ValueError("Long-context attention requires parallel.mesh.context > 1.")
    if ulysses_degree * ring_degree != context_size:
        raise ValueError(
            "Long-context attention requires "
            "parallel.backend_kwargs.long_context.ulysses_degree * ring_degree == parallel.mesh.context."
        )


def _set_yunchang_process_groups(device_mesh: DeviceMesh, config: Mapping[str, Any]) -> None:
    _import_yunchang()
    if not dist.is_available() or not dist.is_initialized():
        return

    configure_long_context_process_groups(device_mesh, dict(config))


def _get_context_subranks(
    context_rank: int,
    *,
    ulysses_degree: int,
    ring_degree: int,
    use_ulysses_low: bool,
) -> tuple[int, int]:
    if use_ulysses_low:
        return context_rank // ulysses_degree, context_rank % ulysses_degree
    return context_rank % ring_degree, context_rank // ring_degree


def _make_context_grad_sync_hook(group: dist.ProcessGroup):
    state: dict[str, Any] = {}

    @torch.no_grad()
    def hook(parameter: torch.Tensor) -> None:
        grad = parameter.grad
        if grad is None:
            state.clear()
            return

        local_grad = _local_tensor(grad)
        grad_ref = state.get("grad_ref")
        if grad_ref is None or grad_ref() is not grad:
            state["grad_ref"] = weakref.ref(grad)
            state["synced"] = torch.zeros_like(local_grad)

        synced = state["synced"]
        delta = local_grad.detach().clone()
        delta.sub_(synced)
        dist.all_reduce(delta, op=dist.ReduceOp.SUM, group=group)
        local_grad.copy_(synced + delta)
        synced.copy_(local_grad)

    return hook


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "to_local"):
        local_tensor = tensor.to_local()
        wait = getattr(local_tensor, "wait", None)
        if callable(wait):
            local_tensor = wait()
        return local_tensor
    return tensor


def _import_yunchang():
    try:
        import yunchang
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Long-context attention requires `pip install 'mvp_engine[long-context]'`.") from exc
    return yunchang
