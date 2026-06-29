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
        lr_groups: list[tuple[tuple[str, ...], float]] | None = None,
        **kwargs,
    ) -> torch.optim.Optimizer:
        """Build a torch optimizer over parameters that are still trainable.

        Args:
            lr_groups: optional list of ``(name_prefixes, group_lr)``. A trainable
                parameter whose qualified name contains any of ``name_prefixes`` is
                placed in that group with ``group_lr``; the first matching group wins.
                Remaining trainable parameters use the default ``lr``. When ``None``
                (default), all trainable parameters share a single ``lr`` group.
        """
        import torch

        named_trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not named_trainable:
            raise ValueError("No trainable parameters found.")

        assert hasattr(torch.optim, optimizer), f"Optimizer '{optimizer}' not found in torch.optim."

        if not lr_groups:
            param_groups: list = [p for _, p in named_trainable]
        else:
            grouped: list[list] = [[] for _ in lr_groups]
            default: list = []
            for name, p in named_trainable:
                for index, (prefixes, _group_lr) in enumerate(lr_groups):
                    if any(prefix in name for prefix in prefixes):
                        grouped[index].append(p)
                        break
                else:
                    default.append(p)
            param_groups = []
            if default:
                param_groups.append({"params": default, "lr": lr})
            for (_prefixes, group_lr), params in zip(lr_groups, grouped):
                if params:
                    param_groups.append({"params": params, "lr": group_lr})

        return getattr(torch.optim, optimizer)(
            param_groups,
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
