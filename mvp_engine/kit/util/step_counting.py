"""Exact optimizer-step counting over a finite dataset.

This module also provides the small distributed helpers (`samples_per_step`,
`resolve_reduce_device`) shared by the MLLM step-estimation kit.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist

from mvp_engine.utils.log import simple_info

SampleCounter = Callable[[object], int]
"""Map one consumed dataset item to the number of training samples it represents."""


def samples_per_step(batch_size: int, gradient_accumulation_steps: int) -> int:
    """Return the number of training samples consumed per optimizer step."""
    value = batch_size * gradient_accumulation_steps
    if value <= 0:
        raise ValueError("`batch_size` and `gradient_accumulation_steps` must be positive.")
    return value


def resolve_reduce_device(
    *,
    dp_world_size: int,
    dp_group: dist.ProcessGroup | None,
    device: torch.device | None,
) -> torch.device:
    """Validate the data-parallel context and return the device for collective reductions."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Step counting requires an initialized torch.distributed group for data-parallel reduction.")
    if dp_group is None:
        raise ValueError("`data_parallel_group` is required when `data_parallel_world_size` > 1.")
    if dp_world_size != dist.get_world_size(dp_group):
        raise ValueError("`data_parallel_world_size` must match the data-parallel group size.")
    if device is None:
        raise ValueError("`device` is required when `data_parallel_world_size` > 1.")
    return device


@dataclass(frozen=True, slots=True)
class StepCountResult:
    """Exact step count from one full pass over a finite dataset."""

    total_steps: int
    total_samples: int
    samples_per_step: int


class StepCountingKit:
    """Count optimizer steps by fully consuming a finite dataset."""

    def __init__(self, *, sample_counter: SampleCounter = lambda _item: 1) -> None:
        """Configure how many training samples one consumed item represents."""
        self.sample_counter = sample_counter

    def count_total_steps(
        self,
        dataset: object,
        *,
        batch_size: int,
        gradient_accumulation_steps: int,
        data_parallel_world_size: int,
        data_parallel_group: dist.ProcessGroup | None = None,
        device: torch.device | None = None,
    ) -> StepCountResult:
        """Count optimizer steps for one finite epoch across all data-parallel ranks."""
        per_step = samples_per_step(batch_size, gradient_accumulation_steps)
        local_samples = self._count_local_samples(dataset)
        total_samples, total_steps, all_ranks_nonempty = _reduce_counts(
            local_samples,
            per_step=per_step,
            dp_world_size=data_parallel_world_size,
            dp_group=data_parallel_group,
            device=device,
        )

        if total_samples <= 0:
            raise RuntimeError("Step counting found no training samples.")
        if not all_ranks_nonempty:
            raise RuntimeError("Step counting found a data-parallel rank with no training samples.")

        simple_info(
            f"Step counting: total_samples={total_samples}, samples_per_step={per_step}, "
            f"dp_world_size={data_parallel_world_size}, total_steps={total_steps}"
        )
        return StepCountResult(total_steps=total_steps, total_samples=total_samples, samples_per_step=per_step)

    def _count_local_samples(self, dataset: object) -> int:
        consume = getattr(dataset, "consume", None)
        if not callable(consume):
            raise TypeError("StepCountingKit requires a dataset with a consume(factory) method.")
        return int(consume(lambda _context: _SampleCountConsumer(self.sample_counter)))


class _SampleCountConsumer:
    """Sum the sample weight of every item in a finite dataset stream."""

    def __init__(self, sample_counter: SampleCounter) -> None:
        self.sample_counter = sample_counter
        self.total_samples = 0

    def push(self, item: object) -> bool:
        count = int(self.sample_counter(item))
        if count <= 0:
            raise RuntimeError("Sample counter must return a positive integer.")
        self.total_samples += count
        return True

    def finish(self) -> int:
        return self.total_samples


def _reduce_counts(
    local_samples: int,
    *,
    per_step: int,
    dp_world_size: int,
    dp_group: dist.ProcessGroup | None,
    device: torch.device | None,
) -> tuple[int, int, bool]:
    """Reduce per-rank sample counts into global totals (samples=sum, steps=max, nonempty=min)."""
    local_steps = math.ceil(local_samples / per_step)
    if dp_world_size <= 1:
        return local_samples, local_steps, local_samples > 0

    reduce_device = resolve_reduce_device(dp_world_size=dp_world_size, dp_group=dp_group, device=device)
    total_samples = torch.tensor(local_samples, dtype=torch.long, device=reduce_device)
    total_steps = torch.tensor(local_steps, dtype=torch.long, device=reduce_device)
    all_nonempty = torch.tensor(int(local_samples > 0), dtype=torch.long, device=reduce_device)

    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM, group=dp_group)
    dist.all_reduce(total_steps, op=dist.ReduceOp.MAX, group=dp_group)
    dist.all_reduce(all_nonempty, op=dist.ReduceOp.MIN, group=dp_group)
    return int(total_samples.item()), int(total_steps.item()), bool(all_nonempty.item())
