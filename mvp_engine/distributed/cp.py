"""Long-context attention helpers built around local Ulysses attention."""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from mvp_engine.distributed.ulysses import UlyssesAttention
from mvp_engine.distributed.utils import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_size,
)
from mvp_engine.utils.log import logger

_HOOK_ATTR = "APPLY_LONG_CONTEXT_ATTENTION"


def is_long_context_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return whether long-context attention is enabled in backend kwargs."""
    return bool((config or {}).get("enabled", False))


def prepare_long_context_attention(
    model: nn.Module,
    device_mesh: DeviceMesh,
    config: Mapping[str, Any],
) -> None:
    """Validate context mesh and apply a model long-context hook."""
    _validate_long_context_config(device_mesh)

    hook = getattr(model.__class__, _HOOK_ATTR, None)
    if hook is None:
        raise ValueError(f"{model.__class__.__name__} must define {_HOOK_ATTR} to enable long-context attention.")
    if not callable(hook):
        raise TypeError(f"{model.__class__.__name__}.{_HOOK_ATTR} must be callable, got {type(hook)}.")
    hook(model, device_mesh, dict(config))


def build_long_context_attention(config: Mapping[str, Any], device_mesh: DeviceMesh) -> nn.Module:
    """Build a local Ulysses long-context attention module from backend config."""
    sequence_process_group = get_context_parallel_group(device_mesh)
    if sequence_process_group is None and dist.is_available() and dist.is_initialized():
        raise RuntimeError("Ulysses attention requires an initialized context process group.")
    return UlyssesAttention(
        sequence_process_group=sequence_process_group,
        attn_impl=str(config.get("attn_impl", "fa")),
    )


def extract_local_sequence(
    value: torch.Tensor,
    device_mesh: DeviceMesh,
    *,
    dim: int = 1,
) -> torch.Tensor:
    """Extract this context rank's local sequence shard from a full sequence tensor."""
    context_size = get_context_parallel_size(device_mesh)
    if context_size <= 1:
        return value
    if int(value.shape[dim]) % context_size != 0:
        raise ValueError("Global sequence length must be divisible by context size.")

    context_rank = get_context_parallel_rank(device_mesh)
    return value.chunk(context_size, dim=dim)[context_rank].contiguous()


def get_local_sequence_position_indices(
    global_sequence_length: int,
    device_mesh: DeviceMesh,
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
    return positions.chunk(context_size, dim=0)[context_rank]


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


def _validate_long_context_config(device_mesh: DeviceMesh) -> None:
    context_size = get_context_parallel_size(device_mesh)
    if context_size <= 1:
        raise ValueError("Long-context attention requires parallel.mesh.context > 1.")


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
