"""Optimizer-step estimation for finite packed LLM datasets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from mvp_engine.utils.log import simple_info

from ...util.step_counting import StepCountingKit, samples_per_step


@dataclass(frozen=True, slots=True)
class LLMStepEstimateResult:
    """Exact step count from one finite pass over a packed text-LM dataset."""

    total_steps: int
    total_packed_samples: int
    samples_per_step: int
    exact: bool = True


class LLMStepEstimationKit:
    """Estimate optimizer steps for packed text-LM datasets."""

    Result = LLMStepEstimateResult

    def estimate_total_steps(
        self,
        dataset: object,
        *,
        batch_size: int,
        gradient_accumulation_steps: int,
        data_parallel_world_size: int,
        data_parallel_group: dist.ProcessGroup | None = None,
        device: torch.device | None = None,
    ) -> LLMStepEstimateResult:
        """Count packed samples exactly and convert them into optimizer steps."""
        per_step = samples_per_step(batch_size, gradient_accumulation_steps)
        result = StepCountingKit().count_total_steps(
            dataset,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            data_parallel_world_size=data_parallel_world_size,
            data_parallel_group=data_parallel_group,
            device=device,
        )
        simple_info(
            f"LLM step estimation: exact=True, packed_samples={result.total_samples}, "
            f"samples_per_step={per_step}, dp_world_size={data_parallel_world_size}, "
            f"total_steps={result.total_steps}"
        )
        return LLMStepEstimateResult(
            total_steps=result.total_steps,
            total_packed_samples=result.total_samples,
            samples_per_step=per_step,
        )
