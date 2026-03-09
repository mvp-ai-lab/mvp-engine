"""Training utilities for gradient accumulation and mixed precision."""

import math
from contextlib import contextmanager
from typing import Generator, Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from mvp_engine.distributed.utils import (
    get_grad_scalar_device,
    is_dtensor_tensor,
    reduce_dtensor_scalar,
    scale_dtensor_grad_,
    to_local_dense_tensor,
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
    parameters,
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
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return torch.tensor(0.0)

    if any(is_dtensor_tensor(param.grad) for param in parameters):
        return _clip_grad_norm_for_dtensor(parameters, max_norm=max_norm, norm_type=norm_type)

    return torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm, norm_type=norm_type)


def _tensor_norm_power(grad: torch.Tensor, norm_type: float, device: torch.device) -> torch.Tensor:
    """Compute sum(|g|^p) for finite p or max(|g|) for inf-norm."""
    local_grad = to_local_dense_tensor(grad).detach()
    if local_grad.numel() == 0:
        return torch.zeros((), device=device, dtype=torch.float32)

    local_grad = local_grad.to(dtype=torch.float32)
    if math.isinf(norm_type):
        return local_grad.abs().max().to(device=device)
    return local_grad.abs().pow(norm_type).sum().to(device=device)


def _clip_grad_norm_for_dtensor(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """Clip gradients when DTensor grads are present without using foreach kernels."""
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    device = get_grad_scalar_device(parameters)

    ordinary_grads = [grad for grad in grads if not is_dtensor_tensor(grad)]
    dtensor_grads = [grad for grad in grads if is_dtensor_tensor(grad)]

    if math.isinf(norm_type):
        ordinary_norm = torch.zeros((), device=device, dtype=torch.float32)
        if ordinary_grads:
            ordinary_norm = torch.stack(
                [_tensor_norm_power(grad, norm_type=norm_type, device=device) for grad in ordinary_grads]
            ).max()

        dtensor_norm = torch.zeros((), device=device, dtype=torch.float32)
        if dtensor_grads:
            dtensor_norm = torch.stack(
                [
                    reduce_dtensor_scalar(
                        _tensor_norm_power(grad, norm_type=norm_type, device=device),
                        grad,
                        dist.ReduceOp.MAX,
                    )
                    for grad in dtensor_grads
                ]
            ).max()

        total_norm = torch.maximum(ordinary_norm, dtensor_norm)
    else:
        ordinary_power = torch.zeros((), device=device, dtype=torch.float32)
        if ordinary_grads:
            ordinary_power = torch.stack(
                [_tensor_norm_power(grad, norm_type=norm_type, device=device) for grad in ordinary_grads]
            ).sum()

        dtensor_power = torch.zeros((), device=device, dtype=torch.float32)
        if dtensor_grads:
            dtensor_power = torch.stack(
                [
                    reduce_dtensor_scalar(
                        _tensor_norm_power(grad, norm_type=norm_type, device=device),
                        grad,
                        dist.ReduceOp.SUM,
                    )
                    for grad in dtensor_grads
                ]
            ).sum()

        total_norm = (ordinary_power + dtensor_power).pow(1.0 / norm_type)

    clip_coef = torch.tensor(float(max_norm), device=device, dtype=torch.float32) / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    if clip_coef_clamped.item() < 1.0:
        for grad in grads:
            scale_dtensor_grad_(grad, clip_coef_clamped)

    return total_norm
