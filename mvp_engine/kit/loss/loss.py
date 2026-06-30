"""Reusable scalar loss helpers."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from mvp_engine.utils.log import simple_info

if TYPE_CHECKING:
    import torch


class LossKit:
    """Group reusable scalar loss utilities."""

    def __init__(self, loss_guard: LossGuard | None = None) -> None:
        """Initialize scalar loss helpers and optional guard state."""
        self.loss_guard = loss_guard

    def build_loss_guard(
        self,
        *,
        spike_multiplier: float | None,
        window_size: int,
        min_history: int,
    ) -> LossGuard:
        """Build and store the scalar loss guard used by ``guard_loss``."""
        self.loss_guard = LossGuard(
            spike_multiplier=spike_multiplier,
            window_size=window_size,
            min_history=min_history,
        )
        return self.loss_guard

    def guard_loss(self, loss: torch.Tensor | float, *, step: int = 0) -> bool:
        """Return whether ``loss`` should participate in backward."""
        if self.loss_guard is None:
            return True
        return not self.loss_guard.check(loss, step=step)


class LossGuard:
    """Detect and skip scalar micro-batch losses with anomalous spikes."""

    def __init__(
        self,
        *,
        spike_multiplier: float | None,
        window_size: int,
        min_history: int,
    ) -> None:
        """Initialize the scalar-loss spike detector."""
        self.spike_multiplier = spike_multiplier
        self.min_history = min_history
        self.loss_history: deque[float] = deque(maxlen=window_size)

    def check(self, loss: torch.Tensor | float, *, step: int, token_count: int | None = None) -> bool:
        """Return whether the current loss should be skipped."""
        if self.spike_multiplier is None:
            return False

        current_loss = self._as_float(loss)
        if len(self.loss_history) < self.min_history:
            self.loss_history.append(current_loss)
            return False

        baseline = sum(self.loss_history) / len(self.loss_history)
        is_spike = current_loss > baseline * float(self.spike_multiplier)
        if is_spike:
            token_text = f", micro_tokens={token_count}" if token_count is not None else ""
            loss_factor = current_loss / baseline if baseline > 0 else float("inf")
            simple_info(
                "Loss spike skip at step "
                f"{step}: micro_loss={current_loss:.4f}, "
                f"baseline_loss={baseline:.4f}, "
                f"history_size={len(self.loss_history)}, "
                f"loss_factor={loss_factor:.2f}, "
                f"spike_multiplier={self.spike_multiplier}"
                f"{token_text}",
                level="warning",
            )
            return True

        self.loss_history.append(current_loss)
        return False

    @staticmethod
    def _as_float(loss: torch.Tensor | float | Any) -> float:
        """Convert a scalar tensor or Python value to a float."""
        import torch

        if isinstance(loss, torch.Tensor):
            return float(loss.detach().item())
        return float(loss)


LossKit.Guard = LossGuard
