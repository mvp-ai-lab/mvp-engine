"""Loss guards for OpenBee training."""

from __future__ import annotations

from collections import deque
from typing import Any

import torch

from mvp_engine.utils.log import simple_info


class LossGuard:
    """Detect and skip micro-batches with anomalously high loss."""

    def __init__(
        self,
        *,
        spike_multiplier: float | None,
        window_size: int,
        min_history: int,
    ) -> None:
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
        if isinstance(loss, torch.Tensor):
            return float(loss.detach().item())
        return float(loss)
