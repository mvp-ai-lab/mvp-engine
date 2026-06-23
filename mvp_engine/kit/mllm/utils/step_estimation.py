"""Optimizer-step estimation for packed MLLM datasets.

A packed MLLM sample bundles several source rows, so counting one full epoch of
packed samples is expensive. Given the dataset's total source-row count, this
kit consumes packed samples only until the packed/source compression ratio is
stable, then extrapolates the epoch's packed-sample total from that ratio. A
rank whose packed stream ends before the ratio stabilizes simply yields an exact
count for that rank.

The total source-row count is supplied by the caller (e.g. from dataset
metadata), so the kit never inspects the dataset's internals.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch
import torch.distributed as dist

from mvp_engine.utils.log import simple_info

from ...util.step_counting import resolve_reduce_device, samples_per_step


class Confidence(IntEnum):
    """Stability of a step estimate, ordered from least to most confident."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    VERY_HIGH = 3


# Upper bound on the windowed estimates' relative spread for each confidence level.
_SPREAD_THRESHOLDS: tuple[tuple[float, Confidence], ...] = (
    (0.001, Confidence.VERY_HIGH),
    (0.01, Confidence.HIGH),
    (0.05, Confidence.MEDIUM),
)


@dataclass(frozen=True, slots=True)
class StepEstimateResult:
    """Step count estimated from one finite pass over a packed MLLM dataset."""

    total_steps: int
    estimated_total_packed_samples: float
    seen_packed_samples: int
    seen_source_samples: int
    total_source_samples: int
    compression_ratio: float
    confidence: Confidence
    exact: bool


@dataclass(frozen=True, slots=True)
class _LocalEstimate:
    """One data-parallel rank's contribution to the global estimate."""

    total_steps: int
    estimated_packed_samples: float
    seen_packed_samples: int
    seen_source_samples: int
    confidence: Confidence
    exact: bool


class MLLMStepEstimationKit:
    """Estimate optimizer steps from packed MLLM compression statistics."""

    Confidence = Confidence
    Result = StepEstimateResult

    def estimate_total_steps(
        self,
        dataset: object,
        *,
        total_source_samples: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        data_parallel_world_size: int,
        data_parallel_group: dist.ProcessGroup | None = None,
        device: torch.device | None = None,
        target_confidence: Confidence = Confidence.HIGH,
        confidence_window_size: int = 100,
        sync_interval: int = 100,
    ) -> StepEstimateResult:
        """Estimate optimizer steps for one finite epoch across all data-parallel ranks.

        Args:
            total_source_samples: Total source rows in the whole dataset, summed
                across all data-parallel ranks. Split evenly to form each rank's
                source-row target.
        """
        if total_source_samples <= 0:
            raise ValueError("`total_source_samples` must be a positive dataset-wide source-row count.")
        if confidence_window_size <= 0:
            raise ValueError("`confidence_window_size` must be positive.")
        if sync_interval <= 0:
            raise ValueError("`sync_interval` must be positive.")
        per_step = samples_per_step(batch_size, gradient_accumulation_steps)
        source_target = math.ceil(total_source_samples / data_parallel_world_size)

        local = self._estimate_local(
            dataset,
            source_target=source_target,
            per_step=per_step,
            target_confidence=target_confidence,
            confidence_window_size=confidence_window_size,
            data_parallel_world_size=data_parallel_world_size,
            data_parallel_group=data_parallel_group,
            device=device,
            sync_interval=sync_interval,
        )
        result, all_ranks_nonempty = _reduce_estimate(
            local,
            total_source_samples=total_source_samples,
            dp_world_size=data_parallel_world_size,
            dp_group=data_parallel_group,
            device=device,
        )

        if result.seen_packed_samples <= 0:
            raise RuntimeError("MLLM step estimation found no packed training samples.")
        if not all_ranks_nonempty:
            raise RuntimeError("MLLM step estimation found a data-parallel rank with no packed samples.")

        simple_info(
            f"MLLM step estimation: exact={result.exact}, confidence={result.confidence.name}, "
            f"seen_packed={result.seen_packed_samples}, seen_source={result.seen_source_samples}, "
            f"total_source={result.total_source_samples}, compression_ratio={result.compression_ratio:.6f}, "
            f"estimated_total_packed={result.estimated_total_packed_samples:.2f}, "
            f"dp_world_size={data_parallel_world_size}, total_steps={result.total_steps}"
        )
        return result

    def _estimate_local(
        self,
        dataset: object,
        *,
        source_target: int,
        per_step: int,
        target_confidence: Confidence,
        confidence_window_size: int,
        data_parallel_world_size: int,
        data_parallel_group: dist.ProcessGroup | None,
        device: torch.device | None,
        sync_interval: int,
    ) -> _LocalEstimate:
        consume = getattr(dataset, "consume", None)
        if not callable(consume):
            raise TypeError("MLLMStepEstimationKit requires a dataset with a consume(factory) method.")

        return consume(
            lambda _context: _PackedSampleEstimator(
                source_target=source_target,
                per_step=per_step,
                target_confidence=target_confidence,
                window_size=confidence_window_size,
                dp_world_size=data_parallel_world_size,
                dp_group=data_parallel_group,
                device=device,
                sync_interval=sync_interval,
                source_sample_counter=self.count_source_samples,
            )
        )

    def count_source_samples(self, item: object) -> int:
        """Return how many source rows one packed MLLM sample represents."""
        source_sample_num_attr = getattr(item, "source_sample_num", None)
        if source_sample_num_attr is not None:
            return int(source_sample_num_attr)
        if not isinstance(item, dict) or "source_sample_num" not in item:
            return 1
        source_sample_num = item["source_sample_num"]
        if isinstance(source_sample_num, torch.Tensor):
            return int(source_sample_num.sum().item())
        return int(source_sample_num)


class _PackedSampleEstimator:
    """Consume packed samples until the stream ends or the estimate is confident."""

    def __init__(
        self,
        *,
        source_target: int,
        per_step: int,
        target_confidence: Confidence,
        window_size: int,
        dp_world_size: int,
        dp_group: dist.ProcessGroup | None,
        device: torch.device | None,
        sync_interval: int,
        source_sample_counter: Any,
    ) -> None:
        self.source_target = source_target
        self.per_step = per_step
        self.target_confidence = target_confidence
        self.dp_world_size = dp_world_size
        self.dp_group = dp_group
        self.sync_interval = sync_interval
        self.source_sample_counter = source_sample_counter
        self.seen_packed = 0
        self.seen_source = 0
        self.estimates: deque[float] = deque(maxlen=window_size)
        self.confidence = Confidence.LOW
        self.stopped_early = False
        self.global_stop = False
        self.reduce_device = (
            resolve_reduce_device(dp_world_size=dp_world_size, dp_group=dp_group, device=device)
            if dp_world_size > 1
            else None
        )

    def push(self, item: object) -> bool:
        """Consume one packed sample; stop early once the estimate is confident enough."""
        source_samples = int(self.source_sample_counter(item))
        if source_samples <= 0:
            raise RuntimeError("MLLM source-sample count must be positive.")
        self.seen_packed += 1
        self.seen_source += source_samples

        self.estimates.append(self.source_target * self.seen_packed / self.seen_source)
        if len(self.estimates) < self.estimates.maxlen:
            return True

        self.confidence = _estimate_confidence(self.estimates)
        if self.dp_world_size <= 1 and self.confidence >= self.target_confidence:
            self.stopped_early = True
            return False
        if self.dp_world_size > 1 and self.seen_packed % self.sync_interval == 0:
            return not self._sync_stop(local_done=False)
        return True

    def finish(self) -> _LocalEstimate:
        while self.dp_world_size > 1 and not self.global_stop:
            self._sync_stop(local_done=True)

        # Consuming the whole stream yields an exact packed count for this rank;
        # stopping early means the packed total is extrapolated from the ratio.
        exact = not self.stopped_early
        compression_ratio = self.seen_packed / self.seen_source if self.seen_source > 0 else 0.0
        estimated_packed = float(self.seen_packed) if exact else self.source_target * compression_ratio
        return _LocalEstimate(
            total_steps=math.ceil(estimated_packed / self.per_step),
            estimated_packed_samples=estimated_packed,
            seen_packed_samples=self.seen_packed,
            seen_source_samples=self.seen_source,
            confidence=Confidence.VERY_HIGH if exact else self.confidence,
            exact=exact,
        )

    def _sync_stop(self, *, local_done: bool) -> bool:
        local_ready = local_done or self.confidence >= self.target_confidence
        state = torch.tensor([int(local_ready), int(local_done)], dtype=torch.long, device=self.reduce_device)
        dist.all_reduce(state, op=dist.ReduceOp.MIN, group=self.dp_group)

        all_ready = bool(state[0].item())
        all_done = bool(state[1].item())
        self.global_stop = all_ready or all_done
        if self.global_stop and not local_done and not all_done:
            self.stopped_early = True
        return self.global_stop


def _estimate_confidence(estimates: Sequence[float]) -> Confidence:
    """Rate estimate stability by the relative spread across the window."""
    mean_estimate = sum(estimates) / len(estimates)
    if mean_estimate <= 0:
        return Confidence.LOW
    relative_spread = (max(estimates) - min(estimates)) / mean_estimate
    for threshold, confidence in _SPREAD_THRESHOLDS:
        if relative_spread <= threshold:
            return confidence
    return Confidence.LOW


def _reduce_estimate(
    local: _LocalEstimate,
    *,
    total_source_samples: int,
    dp_world_size: int,
    dp_group: dist.ProcessGroup | None,
    device: torch.device | None,
) -> tuple[StepEstimateResult, bool]:
    """Reduce per-rank estimates into global totals (counts=sum, steps=max, exact/confidence=min)."""
    if dp_world_size <= 1:
        ratio = local.seen_packed_samples / local.seen_source_samples if local.seen_source_samples > 0 else 0.0
        result = StepEstimateResult(
            total_steps=local.total_steps,
            estimated_total_packed_samples=local.estimated_packed_samples,
            seen_packed_samples=local.seen_packed_samples,
            seen_source_samples=local.seen_source_samples,
            total_source_samples=total_source_samples,
            compression_ratio=ratio,
            confidence=local.confidence,
            exact=local.exact,
        )
        return result, local.seen_packed_samples > 0

    reduce_device = resolve_reduce_device(dp_world_size=dp_world_size, dp_group=dp_group, device=device)
    counts = torch.tensor(
        [local.seen_packed_samples, local.seen_source_samples], dtype=torch.long, device=reduce_device
    )
    estimated_packed = torch.tensor(local.estimated_packed_samples, dtype=torch.float64, device=reduce_device)
    total_steps = torch.tensor(local.total_steps, dtype=torch.long, device=reduce_device)
    all_nonempty = torch.tensor(int(local.seen_packed_samples > 0), dtype=torch.long, device=reduce_device)
    flags = torch.tensor([int(local.exact), int(local.confidence)], dtype=torch.long, device=reduce_device)

    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=dp_group)
    dist.all_reduce(estimated_packed, op=dist.ReduceOp.SUM, group=dp_group)
    dist.all_reduce(total_steps, op=dist.ReduceOp.MAX, group=dp_group)
    dist.all_reduce(all_nonempty, op=dist.ReduceOp.MIN, group=dp_group)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=dp_group)

    seen_packed, seen_source = (int(value) for value in counts.tolist())
    exact = bool(flags[0].item())
    compression_ratio = seen_packed / seen_source if seen_source > 0 else 0.0
    result = StepEstimateResult(
        total_steps=int(total_steps.item()),
        estimated_total_packed_samples=float(estimated_packed.item()),
        seen_packed_samples=seen_packed,
        seen_source_samples=seen_source,
        total_source_samples=total_source_samples,
        compression_ratio=compression_ratio,
        confidence=Confidence.VERY_HIGH if exact else Confidence(int(flags[1].item())),
        exact=exact,
    )
    return result, bool(all_nonempty.item())
