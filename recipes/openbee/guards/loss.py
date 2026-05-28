"""Loss guards for OpenBee training."""

from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.distributed as dist

from mvp_engine.utils.log import simple_info


class LossGuard:
    """Detect and skip micro-batches with anomalously high loss."""

    def __init__(
        self,
        *,
        spike_multiplier: float | None,
        window_size: int,
        min_history: int,
        group: dist.ProcessGroup | None = None,
        group_world_size: int | None = None,
    ) -> None:
        """Initialize the scalar-loss spike detector."""
        self.spike_multiplier = spike_multiplier
        self.min_history = min_history
        self.group = group
        self.group_world_size = group_world_size
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
        if isinstance(loss, torch.Tensor):
            return float(loss.detach().item())
        return float(loss)


class PerTokenLossGuard(LossGuard):
    """Detect spikes from per-token loss sums and valid token counts."""

    def check(
        self,
        loss_sum: torch.Tensor | float,
        token_count: int,
        *,
        step: int,
        device: torch.device,
    ) -> bool:
        """Return whether the current per-token micro-batch loss should be skipped."""
        if self.spike_multiplier is None:
            return False

        loss_stats = torch.stack(
            (
                self._as_tensor(loss_sum, device=device),
                torch.tensor(float(token_count), device=device, dtype=torch.float64),
            )
        )
        should_reduce = self.group_world_size is None or self.group_world_size > 1
        if should_reduce and dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM, group=self.group)

        global_token_count = int(loss_stats[1].item())
        if global_token_count <= 0:
            return False

        return super().check(
            loss_stats[0] / loss_stats[1],
            step=step,
            token_count=global_token_count,
        )

    @staticmethod
    def _as_tensor(value: torch.Tensor | float, *, device: torch.device) -> torch.Tensor:
        """Convert a per-rank loss sum into a float64 scalar tensor."""
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=device, dtype=torch.float64)
        return torch.tensor(float(value), device=device, dtype=torch.float64)
