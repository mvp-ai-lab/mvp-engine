"""Metric accumulation helpers for the OpenBee recipe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.distributed as dist

ReducerName = Literal["sum", "last", "max", "mean"]
AccumulatorName = Literal["sum", "add", "last", "max", "mean", "avg"]
DistributedReducerName = Literal["sum", "max", "mean", "avg", "none"]


@dataclass(slots=True)
class MetricAccumulator:
    """Accumulate named metrics with simple reducers.

    This helper is intentionally small and recipe-local. It collects metrics
    over an accumulation window and finalizes them into plain Python scalars
    for logging or downstream computation.
    """

    reducers: dict[str, ReducerName] = field(default_factory=dict)
    _values: dict[str, float | torch.Tensor | None] = field(default_factory=dict, init=False)
    _counts: dict[str, int] = field(default_factory=dict, init=False)

    def register(self, name: str, reducer: ReducerName) -> None:
        """Register a metric name with a supported reducer."""
        if name in self.reducers:
            raise ValueError(f"Metric '{name}' is already registered.")
        self.reducers[name] = reducer
        self._values[name] = None
        if reducer == "mean":
            self._counts[name] = 0

    def reset(self) -> None:
        """Reset all registered metric state for a new accumulation window."""
        for name, reducer in self.reducers.items():
            self._values[name] = None
            if reducer == "mean":
                self._counts[name] = 0

    def update(self, **metrics: Any) -> None:
        """Update registered metrics with the provided values."""
        for name, value in metrics.items():
            if name not in self.reducers:
                raise KeyError(f"Metric '{name}' is not registered.")
            if value is None:
                continue

            reducer = self.reducers[name]
            detached_value = self._detach_value(value)
            current_value = self._values[name]

            if reducer == "last":
                self._values[name] = detached_value
            elif reducer == "sum":
                self._values[name] = detached_value if current_value is None else current_value + detached_value
            elif reducer == "max":
                self._values[name] = (
                    detached_value
                    if current_value is None
                    else torch.maximum(
                        self._as_tensor(current_value),
                        self._as_tensor(detached_value),
                    )
                )
            elif reducer == "mean":
                self._values[name] = detached_value if current_value is None else current_value + detached_value
                self._counts[name] += 1
            else:  # pragma: no cover - guarded by register typing.
                raise ValueError(f"Unsupported reducer '{reducer}' for metric '{name}'.")

    def get(self, name: str) -> float | torch.Tensor | None:
        """Return the current raw state for a registered metric."""
        if name not in self.reducers:
            raise KeyError(f"Metric '{name}' is not registered.")
        return self._values[name]

    def finalize(self) -> dict[str, float]:
        """Finalize accumulated metrics into Python float values."""
        finalized: dict[str, float] = {}
        for name, reducer in self.reducers.items():
            value = self._values[name]
            if value is None:
                continue

            if reducer == "mean":
                count = self._counts.get(name, 0)
                if count <= 0:
                    continue
                value = value / count

            finalized[name] = self._to_float(value)
        return finalized

    @staticmethod
    def _detach_value(value: Any) -> float | torch.Tensor:
        """Detach tensor inputs while leaving Python scalars unchanged."""
        if isinstance(value, torch.Tensor):
            return value.detach()
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return value
        raise TypeError(f"Unsupported metric value type: {type(value).__name__}")

    @staticmethod
    def _as_tensor(value: float | torch.Tensor) -> torch.Tensor:
        """Convert scalar-like values to tensors for tensor-wise reductions."""
        if isinstance(value, torch.Tensor):
            return value
        return torch.tensor(value)

    @staticmethod
    def _to_float(value: float | torch.Tensor) -> float:
        """Convert a scalar tensor or Python scalar to float."""
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("MetricAccumulator can only finalize scalar tensors to float.")
            return float(value.item())
        return float(value)


class DistributedMetric:
    """One locally accumulated metric with an optional distributed value."""

    def __init__(
        self,
        *,
        name: str,
        device: torch.device,
        accumulate: AccumulatorName,
        reduce: DistributedReducerName,
    ) -> None:
        self.name = name
        self.device = device
        self.accumulate = self._normalize_accumulate(accumulate)
        self.reduce = self._normalize_reduce(reduce)
        self._local_sum: torch.Tensor | None = None
        self._count = 0
        self._global_value: float | None = None

    @property
    def local_value(self) -> float:
        """Return the local accumulated value."""
        return float(self._local_tensor().item())

    @property
    def global_value(self) -> float:
        """Return the cached distributed value from the latest reduce."""
        if self._global_value is None:
            raise RuntimeError(f"Metric '{self.name}' has not been reduced yet.")
        return self._global_value

    def reset(self) -> None:
        """Clear local and distributed state."""
        self._local_sum = None
        self._count = 0
        self._global_value = None

    def update(self, value: Any) -> None:
        """Update the local accumulated value."""
        if value is None:
            return

        tensor_value = self._as_tensor(value)
        self._global_value = None

        if self.accumulate == "last":
            self._local_sum = tensor_value
            self._count = 1
        elif self.accumulate == "sum":
            self._local_sum = tensor_value if self._local_sum is None else self._local_sum + tensor_value
            self._count += 1
        elif self.accumulate == "mean":
            self._local_sum = tensor_value if self._local_sum is None else self._local_sum + tensor_value
            self._count += 1
        elif self.accumulate == "max":
            self._local_sum = tensor_value if self._local_sum is None else torch.maximum(self._local_sum, tensor_value)
            self._count += 1
        else:  # pragma: no cover - guarded by normalization.
            raise ValueError(f"Unsupported accumulation method: {self.accumulate}")

    def set_global_tensor(self, value: torch.Tensor) -> None:
        """Cache a reduced scalar tensor as the metric global value."""
        self._global_value = float(value.item())

    def _local_tensor(self) -> torch.Tensor:
        if self._local_sum is None:
            raise RuntimeError(f"Metric '{self.name}' has no local value.")
        if self.accumulate == "mean":
            if self._count <= 0:
                raise RuntimeError(f"Metric '{self.name}' has no local updates.")
            return self._local_sum / float(self._count)
        return self._local_sum

    def _as_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Metric '{self.name}' only supports scalar tensors.")
            return value.detach().to(device=self.device, dtype=torch.float64)
        if isinstance(value, bool):
            return torch.tensor(float(value), device=self.device, dtype=torch.float64)
        if isinstance(value, int | float):
            return torch.tensor(float(value), device=self.device, dtype=torch.float64)
        raise TypeError(f"Unsupported metric value type for '{self.name}': {type(value).__name__}")

    @staticmethod
    def _normalize_accumulate(accumulate: AccumulatorName) -> Literal["sum", "last", "max", "mean"]:
        if accumulate == "add":
            return "sum"
        if accumulate == "avg":
            return "mean"
        if accumulate in {"sum", "last", "max", "mean"}:
            return accumulate
        raise ValueError(f"Unsupported accumulation method: {accumulate}")

    @staticmethod
    def _normalize_reduce(reduce: DistributedReducerName) -> Literal["sum", "max", "mean", "none"]:
        if reduce == "avg":
            return "mean"
        if reduce in {"sum", "max", "mean", "none"}:
            return reduce
        raise ValueError(f"Unsupported distributed reduce method: {reduce}")


@dataclass(slots=True)
class DistributedMetricAccumulator:
    """Accumulate named metrics locally and cache distributed reductions."""

    device: torch.device
    _metrics: dict[str, DistributedMetric] = field(default_factory=dict, init=False)

    def register(
        self,
        name: str,
        *,
        accumulate: AccumulatorName = "sum",
        reduce: DistributedReducerName = "sum",
    ) -> None:
        """Register a metric with local accumulation and distributed reduction modes."""
        if name in self._metrics:
            raise ValueError(f"Metric '{name}' is already registered.")
        self._metrics[name] = DistributedMetric(
            name=name,
            device=self.device,
            accumulate=accumulate,
            reduce=reduce,
        )

    def update(self, **metrics: Any) -> None:
        """Update registered metrics with the provided local values."""
        for name, value in metrics.items():
            if name not in self._metrics:
                raise KeyError(f"Metric '{name}' is not registered.")
            self._metrics[name].update(value)

    def reduce_all(self) -> None:
        """Reduce all registered metrics and cache their global values."""
        grouped_metrics: dict[str, list[DistributedMetric]] = {}
        for metric in self._metrics.values():
            if metric.reduce == "none":
                metric.set_global_tensor(metric._local_tensor())
            else:
                grouped_metrics.setdefault(metric.reduce, []).append(metric)

        for reduce_method, metrics in grouped_metrics.items():
            values = torch.stack([metric._local_tensor() for metric in metrics])
            if dist.is_available() and dist.is_initialized():
                reduce_op = dist.ReduceOp.MAX if reduce_method == "max" else dist.ReduceOp.SUM
                dist.all_reduce(values, op=reduce_op)
                if reduce_method == "mean":
                    values = values / float(dist.get_world_size())

            for metric, value in zip(metrics, values, strict=True):
                metric.set_global_tensor(value)

    def reset(self) -> None:
        """Reset all registered metric state."""
        for metric in self._metrics.values():
            metric.reset()

    def __getattr__(self, name: str) -> DistributedMetric:
        """Expose registered metrics as attributes."""
        if name in self._metrics:
            return self._metrics[name]
        raise AttributeError(f"{self.__class__.__name__!s} has no metric '{name}'.")
