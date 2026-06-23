"""Reusable optimizer and scheduler utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class OptimKit:
    """Group reusable optimizer and scheduler construction utilities."""

    def build_optimizer(
        self,
        model: torch.nn.Module,
        optimizer: str,
        lr: float,
        weight_decay: float,
        **kwargs,
    ) -> torch.optim.Optimizer:
        """Build a torch optimizer over parameters that are still trainable."""
        import torch

        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise ValueError("No trainable parameters found.")

        assert hasattr(torch.optim, optimizer), f"Optimizer '{optimizer}' not found in torch.optim."
        return getattr(torch.optim, optimizer)(
            trainable_parameters,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs,
        )

    def build_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: str,
        **kwargs,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Build a Transformers learning-rate scheduler for the optimizer."""
        from transformers import get_scheduler

        return get_scheduler(
            name=lr_scheduler,
            optimizer=optimizer,
            **kwargs,
        )
