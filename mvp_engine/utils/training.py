"""Training utilities for gradient accumulation and mixed precision."""

import math
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.distributed.tensor import DTensor
except Exception:  # pragma: no cover - runtime-dependent
    DTensor = ()


def _get_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local shard/replica tensor for a DTensor."""
    local_tensor = tensor.to_local()
    wait = getattr(local_tensor, "wait", None)
    if callable(wait):
        local_tensor = wait()
    return local_tensor


def _get_dtensor_placements_key(grad: torch.Tensor) -> tuple[str, ...]:
    """Represent DTensor placements so clipping can group identical layouts together."""
    return tuple(repr(placement) for placement in grad.placements)


def _get_dtensor_reduce_device(grad: torch.Tensor) -> torch.device:
    """Return a device compatible with the DTensor mesh process group collectives."""
    device_type = grad.device_mesh.device_type
    if device_type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(device_type)


def _get_dtensor_group_total_norm(group_grads: list[torch.Tensor], norm_type: float) -> torch.Tensor:
    """Compute the global norm for DTensor gradients without materializing full tensors."""
    local_tensors = [_get_local_tensor(grad).detach() for grad in group_grads]
    local_device = local_tensors[0].device
    reduce_device = _get_dtensor_reduce_device(group_grads[0])

    if math.isinf(norm_type):
        total_norm = (
            torch.stack(
                [
                    torch.linalg.vector_norm(local_tensor, ord=norm_type, dtype=torch.float32)
                    for local_tensor in local_tensors
                ]
            )
            .amax()
            .to(device=reduce_device)
        )
        reduce_op = dist.ReduceOp.MAX
    else:
        total_norm = (
            torch.stack(
                [
                    torch.linalg.vector_norm(local_tensor, ord=norm_type, dtype=torch.float32) ** norm_type
                    for local_tensor in local_tensors
                ]
            )
            .sum()
            .to(device=reduce_device)
        )
        reduce_op = dist.ReduceOp.SUM

    grad0 = group_grads[0]
    for mesh_dim, placement in enumerate(grad0.placements):
        if placement.is_shard() or placement.is_partial():
            dist.all_reduce(total_norm, op=reduce_op, group=grad0.device_mesh.get_group(mesh_dim))

    if not math.isinf(norm_type):
        total_norm = total_norm ** (1.0 / norm_type)

    return total_norm.to(device=local_device, dtype=torch.float32)


@torch.no_grad()
def _clip_dtensor_group_grads_with_norm_(
    group_grads: list[torch.Tensor],
    max_norm: float,
    total_norm: torch.Tensor,
) -> None:
    """Scale local DTensor shards/replicas in-place using a precomputed global norm."""
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    for grad in group_grads:
        local_tensor = _get_local_tensor(grad)
        local_tensor.mul_(clip_coef_clamped.to(device=local_tensor.device, dtype=local_tensor.dtype))


def _get_grad_mesh_key(grad: torch.Tensor) -> tuple:
    """Group gradients by DTensor mesh so mixed-mesh clipping can be reduced safely."""
    if not isinstance(grad, DTensor):
        return ("local",)

    mesh = grad.device_mesh
    return (
        "dtensor",
        mesh.device_type,
        tuple(mesh.mesh.shape),
        tuple(mesh.mesh.reshape(-1).tolist()),
        tuple(mesh.mesh_dim_names or ()),
        _get_dtensor_placements_key(grad),
    )


@dataclass
class GradientAccumulationState:
    """Track micro-batch progress within a gradient accumulation window."""

    gradient_accumulation_steps: int
    micro_step: int = 0

    def advance(self, skip_increase: bool = False) -> bool:
        """Advance one micro-batch and return whether gradients should sync."""
        if not skip_increase:
            self.micro_step += 1

        if self.micro_step % self.gradient_accumulation_steps == 0:
            self.micro_step = 0
            return True
        return False


@contextmanager
def accumulate_gradients(
    model: Union[torch.nn.Module, DDP],
    sync: bool,
) -> Generator[None, None, None]:
    """Context manager for gradient accumulation with optional DDP sync.

    When ``sync=False``, gradients are accumulated locally without
    all-reduce. When ``sync=True``, gradients are synchronized across
    processes (normal backward behavior).

    Args:
        model: The model (possibly wrapped in DDP or FSDP2).
        sync: Whether to synchronize gradients across processes.

    Yields:
        None
    """
    base_model = getattr(model, "_orig_mod", model)

    # DDP
    if isinstance(base_model, DDP):
        if sync:
            yield
        else:
            with base_model.no_sync():
                yield
        return

    # FSDP2
    if isinstance(base_model, FSDPModule):
        base_model.set_requires_gradient_sync(sync)
        yield
        return

    # Non-parallel
    yield


class GradientScaler:
    """Unified gradient scaler that works with fp16, bf16, and fp32.

    For fp16 on CUDA, uses torch.amp.GradScaler for loss scaling.
    For bf16 and fp32, acts as a pass-through (no scaling needed).

    Args:
        enabled: Whether mixed precision is enabled.
        dtype: The dtype for mixed precision (fp16, bf16, or fp32).
        device: The device type string (cuda, cpu, npu).
        init_scale: Initial scale factor for fp16.
        growth_factor: Factor to grow scale on successful steps.
        backoff_factor: Factor to reduce scale on overflow.
        growth_interval: Steps between scale growth attempts.
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ) -> None:
        self._enabled = enabled
        self._dtype = dtype
        self._device = device

        # Only use actual scaling for fp16 on CUDA
        self._use_scaler = enabled and dtype == torch.float16 and device == "cuda" and torch.cuda.is_available()

        if self._use_scaler:
            self._scaler = torch.amp.GradScaler(
                device=device,
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
                enabled=True,
            )
        else:
            self._scaler = None

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass.

        Args:
            loss: The loss tensor to scale.

        Returns:
            Scaled loss if using fp16, otherwise unchanged loss.
        """
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients before clipping.

        Args:
            optimizer: The optimizer whose gradients to unscale.
        """
        if self._scaler is not None:
            self._scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step optimizer with optional overflow checking.

        Args:
            optimizer: The optimizer to step.
        """
        if self._scaler is not None:
            self._scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self) -> None:
        """Update scale factor after optimizer step."""
        if self._scaler is not None:
            self._scaler.update()

    def get_scale(self) -> float:
        """Return current scale factor.

        Returns:
            Current scale factor, or 1.0 if scaling is disabled.
        """
        if self._scaler is not None:
            return self._scaler.get_scale()
        return 1.0

    def state_dict(self) -> dict:
        """Return scaler state for checkpointing.

        Returns:
            State dict of the underlying scaler, or empty dict.
        """
        if self._scaler is not None:
            return self._scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scaler state from checkpoint.

        Args:
            state_dict: State dict to load.
        """
        if self._scaler is not None and state_dict:
            self._scaler.load_state_dict(state_dict)


@torch.no_grad()
def clip_grad_norm_(
    parameters: nn.Module | Iterable[torch.Tensor] | torch.Tensor,
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """Clip gradient norm of parameters.

    A thin wrapper around torch.nn.utils.clip_grad_norm_ with
    better defaults and documentation.

    Args:
        parameters: Model parameters (or iterable of tensors).
        max_norm: Maximum norm value.
        norm_type: Type of norm (default L2).

    Returns:
        Total norm of the gradients before clipping.
    """
    if isinstance(parameters, nn.Module):
        parameters = parameters.parameters()

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return torch.tensor(0.0)

    grouped_parameters: dict[tuple, list[torch.Tensor]] = {}
    for param in parameters:
        group_key = _get_grad_mesh_key(param.grad)
        grouped_parameters.setdefault(group_key, []).append(param)

    group_total_norms: list[torch.Tensor] = []
    for group_parameters in grouped_parameters.values():
        group_grads = [param.grad for param in group_parameters]
        if isinstance(group_grads[0], DTensor):
            group_total_norm = _get_dtensor_group_total_norm(group_grads, norm_type)
        else:
            group_total_norm = torch.nn.utils.get_total_norm(group_grads, norm_type)
        group_total_norms.append(group_total_norm.to(device=group_grads[0].device, dtype=torch.float32))

    if math.isinf(norm_type):
        total_norm = torch.stack(group_total_norms).amax()
    else:
        total_norm = torch.linalg.vector_norm(torch.stack(group_total_norms), ord=norm_type)

    for group_parameters in grouped_parameters.values():
        group_grads = [param.grad for param in group_parameters]
        if isinstance(group_grads[0], DTensor):
            _clip_dtensor_group_grads_with_norm_(group_grads, max_norm, total_norm)
        else:
            torch.nn.utils.clip_grads_with_norm_(
                group_parameters,
                max_norm,
                total_norm,
                foreach=False,
            )

    return total_norm
