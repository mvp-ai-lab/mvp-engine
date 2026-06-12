"""Packed-dataset step estimation helpers."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from mvp_dataset import Dataset

from mvp_engine.utils.log import simple_info

_CONFIDENCES = ("Low", "Med", "High", "VeryHigh")
_CONFIDENCE_RANKS = {confidence: rank for rank, confidence in enumerate(_CONFIDENCES)}


@dataclass(frozen=True, slots=True)
class StepEstimateResult:
    """Step-count statistics inferred from a packed dataset pipeline."""

    total_steps: int
    estimated_total_packed_samples: float
    seen_packed_samples: int
    seen_source_samples: int
    total_source_samples: int
    compression_ratio: float
    confidence: str
    exact: bool


class StepEstimateConsumer:
    """Consume packed samples until exact completion or estimate confidence."""

    def __init__(
        self,
        *,
        mode: str,
        local_source_samples: int | None,
        target_confidence: str,
        confidence_window_size: int,
    ) -> None:
        self.mode = mode
        self.local_source_samples = local_source_samples
        self.target_confidence = target_confidence
        self.confidence_window_size = int(confidence_window_size)
        self.seen_packed_samples = 0
        self.seen_source_samples = 0
        self.estimates: list[float] = []
        self.confidence = "Low"
        self.stopped_early = False

    def push(self, item: object) -> bool:
        """Consume one packed sample and stop when the estimate is stable enough."""
        self.seen_packed_samples += 1
        self.seen_source_samples += self._source_sample_num(item)

        if self.mode == "exact" or self.local_source_samples is None or self.seen_source_samples <= 0:
            return True

        compression_ratio = self.seen_packed_samples / self.seen_source_samples
        self.estimates.append(float(self.local_source_samples) * compression_ratio)
        if len(self.estimates) > self.confidence_window_size:
            self.estimates.pop(0)
        if len(self.estimates) < self.confidence_window_size:
            return True

        self.confidence = _compute_estimate_confidence(self.estimates)
        if _CONFIDENCE_RANKS[self.confidence] >= _CONFIDENCE_RANKS[self.target_confidence]:
            self.stopped_early = True
            return False
        return True

    def finish(self) -> tuple[int, int, str, bool]:
        """Return raw consumer counters."""
        return self.seen_packed_samples, self.seen_source_samples, self.confidence, not self.stopped_early

    @staticmethod
    def _source_sample_num(item: object) -> int:
        if isinstance(item, dict) and "source_sample_num" in item:
            return int(item["source_sample_num"])
        return 1


def estimate_total_steps(
    dataset: Dataset,
    *,
    batch_size: int,
    gradient_accumulation_steps: int,
    data_parallel_world_size: int,
    data_parallel_group: dist.ProcessGroup | None = None,
    device: torch.device | None = None,
    mode: str = "estimate",
    target_confidence: str = "High",
    confidence_window_size: int = 100,
    sync_interval: int = 10,
) -> StepEstimateResult:
    """Infer optimizer steps by consuming a finite packed dataset pipeline."""
    if mode not in {"estimate", "exact"}:
        raise ValueError("`mode` must be 'estimate' or 'exact'.")
    if target_confidence not in _CONFIDENCE_RANKS:
        raise ValueError(f"`target_confidence` must be one of {tuple(_CONFIDENCE_RANKS)}.")
    if confidence_window_size <= 0:
        raise ValueError("`confidence_window_size` must be positive.")
    if sync_interval <= 0:
        raise ValueError("`sync_interval` must be positive.")

    local_source_samples = _infer_local_source_samples(dataset)
    effective_mode = mode
    if effective_mode == "estimate" and local_source_samples is None:
        simple_info(
            "Step estimation could not infer local source samples from dataset metadata; falling back to exact.",
            level="warning",
        )
        effective_mode = "exact"

    consumer = StepEstimateConsumer(
        mode=effective_mode,
        local_source_samples=local_source_samples,
        target_confidence=target_confidence,
        confidence_window_size=confidence_window_size,
    )
    consumer_result = _consume_dataset(
        dataset,
        consumer,
        sync_interval=sync_interval,
        data_parallel_world_size=data_parallel_world_size,
        data_parallel_group=data_parallel_group,
        device=device,
    )

    local = _build_local_estimate(
        consumer_result=consumer_result,
        local_source_samples=local_source_samples,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    global_estimate = _aggregate_estimate(
        local,
        data_parallel_world_size=data_parallel_world_size,
        data_parallel_group=data_parallel_group,
        device=device,
    )

    if global_estimate.seen_packed_samples <= 0 or global_estimate.seen_source_samples <= 0:
        raise RuntimeError("Step estimation found no packed training samples.")
    if global_estimate.total_steps <= 0:
        raise RuntimeError("Step estimation found fewer packed samples than one optimization step requires.")

    simple_info(
        "Step estimation: "
        f"mode={effective_mode}, exact={global_estimate.exact}, confidence={global_estimate.confidence}, "
        f"seen_packed={global_estimate.seen_packed_samples}, "
        f"seen_source={global_estimate.seen_source_samples}, "
        f"total_source={global_estimate.total_source_samples}, "
        f"compression_ratio={global_estimate.compression_ratio:.6f}, "
        f"estimated_total_packed={global_estimate.estimated_total_packed_samples:.2f}, "
        f"dp_world_size={data_parallel_world_size}, "
        f"rank_policy=max, local_total_steps={local.total_steps}, "
        f"total_steps={global_estimate.total_steps}"
    )

    return global_estimate


def _compute_estimate_confidence(estimates: Sequence[float], *, min_points: int = 5) -> str:
    """Return stability confidence for recent total-packed estimates."""
    if len(estimates) < min_points:
        return "Low"

    mean_estimate = sum(estimates) / len(estimates)
    if mean_estimate <= 0:
        return "Low"

    relative_range = (max(estimates) - min(estimates)) / mean_estimate
    if relative_range <= 0.001:
        return "VeryHigh"
    if relative_range <= 0.01:
        return "High"
    if relative_range <= 0.05:
        return "Med"
    return "Low"


def _consume_dataset(
    dataset: Dataset,
    consumer: StepEstimateConsumer,
    *,
    sync_interval: int,
    data_parallel_world_size: int,
    data_parallel_group: dist.ProcessGroup | None,
    device: torch.device | None,
) -> tuple[int, int, str, bool]:
    """Consume in bounded chunks, with distributed ranks entering collectives regularly."""
    iterator = iter(dataset)
    exhausted = False
    stopped = False
    distributed = dist.is_available() and dist.is_initialized() and int(data_parallel_world_size) > 1
    sync_device = torch.device("cpu") if device is None else device

    while True:
        consumed = 0
        while consumed < sync_interval and not exhausted and not stopped:
            try:
                item = next(iterator)
            except StopIteration:
                exhausted = True
                break

            consumed += 1
            if consumer.push(item) is False:
                stopped = True
                break

        local_done = exhausted or stopped
        if not distributed:
            if local_done:
                break
            continue

        done_flag = torch.tensor(1 if local_done else 0, dtype=torch.long, device=sync_device)
        dist.all_reduce(done_flag, op=dist.ReduceOp.MIN, group=data_parallel_group)
        if bool(done_flag.item()):
            break

    return consumer.finish()


def _infer_local_source_samples(dataset: Dataset) -> int | None:
    """Best-effort source-row count for the current finite data-load slot."""
    if getattr(dataset, "_resample", False):
        return None

    context = getattr(dataset, "context", None)
    build_source_stream = getattr(dataset, "_build_source_stream", None)
    if callable(build_source_stream) and context is not None:
        try:
            source_stream = build_source_stream(context=context)
        except Exception:
            return None
        round_size = getattr(source_stream, "_round_size", None)
        if callable(round_size):
            return int(round_size(0))

    return None


def _build_local_estimate(
    *,
    consumer_result: tuple[int, int, str, bool],
    local_source_samples: int | None,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> StepEstimateResult:
    """Convert raw consumer counters into local step statistics."""
    seen_packed_samples, seen_source_samples, confidence, exact = consumer_result
    local_source_samples = seen_source_samples if local_source_samples is None or exact else local_source_samples
    compression_ratio = seen_packed_samples / seen_source_samples if seen_source_samples > 0 else 0.0
    estimated_total_packed_samples = (
        float(seen_packed_samples) if exact else float(local_source_samples) * compression_ratio
    )

    samples_per_step = int(batch_size) * int(gradient_accumulation_steps)
    if samples_per_step <= 0:
        raise RuntimeError("Step estimation cannot infer steps with non-positive samples per optimization step.")

    return StepEstimateResult(
        total_steps=math.ceil(estimated_total_packed_samples / samples_per_step),
        estimated_total_packed_samples=estimated_total_packed_samples,
        seen_packed_samples=seen_packed_samples,
        seen_source_samples=seen_source_samples,
        total_source_samples=int(local_source_samples),
        compression_ratio=compression_ratio,
        confidence="VeryHigh" if exact else confidence,
        exact=exact,
    )


def _aggregate_estimate(
    local: StepEstimateResult,
    *,
    data_parallel_world_size: int,
    data_parallel_group: dist.ProcessGroup | None,
    device: torch.device | None,
) -> StepEstimateResult:
    """Reduce local step-estimation stats across data-parallel ranks."""
    if not dist.is_available() or not dist.is_initialized() or int(data_parallel_world_size) <= 1:
        return local

    reduce_device = torch.device("cpu") if device is None else device
    counts = torch.tensor(
        [local.seen_packed_samples, local.seen_source_samples, local.total_source_samples],
        dtype=torch.long,
        device=reduce_device,
    )
    estimated_packed = torch.tensor(local.estimated_total_packed_samples, dtype=torch.float64, device=reduce_device)
    max_steps = torch.tensor(local.total_steps, dtype=torch.long, device=reduce_device)
    flags = torch.tensor(
        [1 if local.exact else 0, _CONFIDENCE_RANKS[local.confidence]],
        dtype=torch.long,
        device=reduce_device,
    )

    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=data_parallel_group)
    dist.all_reduce(estimated_packed, op=dist.ReduceOp.SUM, group=data_parallel_group)
    dist.all_reduce(max_steps, op=dist.ReduceOp.MAX, group=data_parallel_group)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=data_parallel_group)

    exact = bool(flags[0].item())
    seen_packed_samples = int(counts[0].item())
    seen_source_samples = int(counts[1].item())
    compression_ratio = seen_packed_samples / seen_source_samples if seen_source_samples > 0 else 0.0
    return StepEstimateResult(
        total_steps=int(max_steps.item()),
        estimated_total_packed_samples=float(estimated_packed.item()),
        seen_packed_samples=seen_packed_samples,
        seen_source_samples=seen_source_samples,
        total_source_samples=int(counts[2].item()),
        compression_ratio=compression_ratio,
        confidence="VeryHigh" if exact else _confidence_from_rank(int(flags[1].item())),
        exact=exact,
    )


def _confidence_from_rank(rank: int) -> str:
    if 0 <= rank < len(_CONFIDENCES):
        return _CONFIDENCES[rank]
    return "Low"
