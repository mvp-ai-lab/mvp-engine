"""Training utilities for gradient accumulation and mixed precision."""

import math
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Generator, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.distributed.tensor import DTensor
except Exception:  # pragma: no cover - runtime-dependent
    DTensor = ()


def _materialize_total_norm(total_norm: torch.Tensor) -> torch.Tensor:
    """Convert DTensor total norms to local tensors so they can be combined safely."""
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()
    return total_norm


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
    )


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
        model: The model (possibly wrapped in DDP).
        sync: Whether to synchronize gradients across processes.

    Yields:
        None
    """
    if isinstance(model, DDP) and not sync:
        with model.no_sync():
            yield
    else:
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
        group_total_norm = torch.nn.utils.get_total_norm(group_grads, norm_type)
        group_total_norm = _materialize_total_norm(group_total_norm)
        group_total_norms.append(group_total_norm.to(device=group_grads[0].device, dtype=torch.float32))

    if math.isinf(norm_type):
        total_norm = torch.stack(group_total_norms).amax()
    else:
        total_norm = torch.linalg.vector_norm(torch.stack(group_total_norms), ord=norm_type)

    for group_parameters in grouped_parameters.values():
        torch.nn.utils.clip_grads_with_norm_(
            group_parameters,
            max_norm,
            total_norm,
            foreach=False,
        )

    return total_norm
