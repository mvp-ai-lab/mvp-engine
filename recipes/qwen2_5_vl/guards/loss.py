"""Loss spike guard for qwen2_5_vl training."""

from __future__ import annotations

from collections import deque

import torch
import torch.distributed as dist

from mvp_engine.utils.log import simple_info


class PerTokenLossGuard:
    """Skip synchronized micro-batches whose mean token loss is an outlier."""

    def __init__(
        self,
        *,
        spike_multiplier: float | None,
        window_size: int,
        min_history: int,
        group: dist.ProcessGroup | None = None,
        group_world_size: int | None = None,
    ) -> None:
        """Initialize the rolling per-token loss detector."""
        self.spike_multiplier = spike_multiplier
        self.min_history = int(min_history)
        self.group = group
        self.group_world_size = group_world_size
        self.loss_history: deque[float] = deque(maxlen=int(window_size))

    def check(
        self,
        loss_sum: torch.Tensor | float,
        token_count: int,
        *,
        step: int,
        device: torch.device,
    ) -> bool:
        """Return whether this micro-batch loss should be skipped."""
        if self.spike_multiplier is None or token_count <= 0:
            return False

        stats = torch.stack(
            (
                _to_scalar_tensor(loss_sum, device=device),
                torch.tensor(float(token_count), device=device, dtype=torch.float64),
            )
        )
        if (self.group_world_size is None or self.group_world_size > 1) and dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=self.group)

        global_tokens = int(stats[1].item())
        if global_tokens <= 0:
            return False

        mean_loss = float((stats[0] / stats[1]).item())
        if len(self.loss_history) < self.min_history:
            self.loss_history.append(mean_loss)
            return False

        baseline = sum(self.loss_history) / len(self.loss_history)
        is_spike = baseline > 0 and mean_loss > baseline * float(self.spike_multiplier)
        if is_spike:
            simple_info(
                "Loss spike skip at step "
                f"{step}: micro_loss={mean_loss:.4f}, baseline_loss={baseline:.4f}, "
                f"history_size={len(self.loss_history)}, micro_tokens={global_tokens}",
                level="warning",
            )
            return True

        self.loss_history.append(mean_loss)
        return False


def _to_scalar_tensor(value: torch.Tensor | float, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=torch.float64)
    return torch.tensor(float(value), device=device, dtype=torch.float64)

