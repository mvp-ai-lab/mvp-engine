"""Metric aggregation utilities for the logging subsystem."""

import gc
from typing import Dict, List, Literal, Optional, TypedDict, Union

import torch
import torch.distributed as dist


class MetricEntry(TypedDict):
    """Book-keeping structure for a named metric."""

    metric: "Metric"
    interval: int
    update_count: int


class Metric:
    """Collect scalar values and compute aggregated statistics."""

    def __init__(
        self,
        accumulation_size: int = 20,
        distributed: bool = False,
        support_nan: bool = True,
        dist_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        """Initialize a metric buffer and aggregation settings.

        Args:
            accumulation_size: Maximum number of recent values to retain.
            distributed: Whether to all-reduce across a process group.
            support_nan: If False, drop NaN updates and reuse last value.
            dist_group: Process group used when ``distributed`` is True.
        """
        self._buffer: List[Optional[Union[float, str]]] = []
        self._accumulation_size = accumulation_size
        self._distributed = distributed
        self._support_nan = support_nan
        self._dist_group = dist_group
        self._is_string = False

    def update(self, value: Optional[Union[int, float, str, torch.Tensor]]) -> None:
        """Add a new value to the buffer.

        Accepts scalars, tensors, strings, or ``None``. Strings flip the
        metric into string mode, where aggregation returns the last value.
        """
        if value is None:
            self._buffer.append(None)
        elif isinstance(value, str):
            self._is_string = True
            self._buffer.append(value)
        else:
            if not isinstance(value, (int, float, torch.Tensor)):
                raise TypeError(f"Metric only supports int, float, str, torch.Tensor or None, got {type(value)}")
            tensor_val = value if torch.is_tensor(value) else torch.tensor(value)
            if torch.isnan(tensor_val).item() and not self._support_nan:
                value = self._buffer[-1] if self._buffer else None
            self._buffer.append(float(value) if value is not None else None)

        if len(self._buffer) > self._accumulation_size + 1:
            self._buffer.pop(0)

    def clear(self) -> None:
        """Clear the buffer and release references."""
        del self._buffer
        self._buffer = []
        gc.collect()

    def _collect(self) -> List[Union[float, str]]:
        """Return the recent non-``None`` values respecting accumulation size."""
        if not self._buffer:
            return []
        slice_vals = self._buffer[-min(len(self._buffer), self._accumulation_size) :]
        return [v for v in slice_vals if v is not None]

    def mean(self) -> Union[float, str]:
        """Compute the (distributed) mean, or last value when in string mode."""
        collected = self._collect()
        if not collected:
            return "" if self._is_string else 0.0

        # For string metrics, return the last value
        if self._is_string:
            return collected[-1]

        values = torch.tensor(collected, dtype=torch.float64)
        count = torch.tensor([values.numel()], dtype=torch.float64)
        total = values.sum()

        if self._distributed and self._dist_group is not None:
            dist.all_reduce(total, group=self._dist_group)
            dist.all_reduce(count, group=self._dist_group)

        return float(total / count)

    def sum(self) -> Union[float, str]:
        """Compute the (distributed) sum, or last value when in string mode."""
        collected = self._collect()
        if not collected:
            return "" if self._is_string else 0.0

        # For string metrics, return the last value
        if self._is_string:
            return collected[-1]

        if torch.cuda.is_available():
            with torch.cuda.stream(torch.cuda.Stream()):
                values = torch.tensor(collected, dtype=torch.float32).cuda()
                total = values.sum()
        else:
            values = torch.tensor(collected, dtype=torch.float32)
            total = values.sum()

        if self._distributed and self._dist_group is not None:
            dist.all_reduce(total, group=self._dist_group)

        return float(total)


class MetricAggregator:
    """Registry of named metrics with interval-based collection."""

    def __init__(
        self,
        dist_group: Optional[dist.ProcessGroup] = None,
        default_interval: int = 20,
        default_accumulation_size: int = 20,
    ) -> None:
        """Create a metric registry.

        Args:
            dist_group: Process group used for distributed reductions.
            default_interval: Fallback interval for collection cadence.
        """
        self._metrics: Dict[str, MetricEntry] = {}
        self._dist_group = dist_group
        self._default_interval = default_interval
        self._default_accumulation_size = default_accumulation_size

    def add(
        self,
        name: str,
        accumulation_size: Optional[int] = None,
        interval: Optional[int] = None,
        distributed: Optional[bool] = None,
        support_nan: bool = True,
    ) -> None:
        """Register a metric for tracking.

        Args:
            name: Unique metric identifier.
            accumulation_size: History length to retain per metric.
            interval: Collection interval; defaults to ``default_interval``.
            distributed: Force distributed reduction flag; None mirrors ``dist.is_initialized``.
            support_nan: If False, drop NaN updates and reuse last numeric value.
        """
        if name in self._metrics:
            return

        self._metrics[name] = {
            "metric": Metric(
                accumulation_size=(self._default_accumulation_size if accumulation_size is None else accumulation_size),
                distributed=dist.is_initialized() if distributed is None else distributed,
                support_nan=support_nan,
                dist_group=self._dist_group,
            ),
            "interval": interval or self._default_interval,
            "update_count": 0,
        }

    def update(self, metrics: Dict[str, Optional[Union[int, float, str, torch.Tensor]]]) -> None:
        """Update registered metrics with new values.

        Args:
            metrics: Mapping from metric name to the latest observation. Values
                can be numeric, tensors, strings, or ``None``.
        """
        for name, value in metrics.items():
            if name not in self._metrics:
                self.add(name)
            self._metrics[name]["metric"].update(value)
            self._metrics[name]["update_count"] += 1

    def collect(
        self,
        metric_names: List[str],
        accumulate: Literal["mean", "sum"] = "mean",
    ) -> Dict[str, Union[float, str]]:
        """Collect aggregated values for metrics that reached their interval.

        Args:
            metric_names: Names to check for collection readiness.
            accumulate: Aggregation mode; either ``"mean"`` or ``"sum"``.

        Returns:
            A mapping of collected metric values keyed by name.
        """
        collected: Dict[str, Union[float, str]] = {}

        for name in metric_names:
            if name not in self._metrics:
                continue

            entry = self._metrics[name]
            if entry["update_count"] >= entry["interval"]:
                entry["update_count"] = 0
                if accumulate == "mean":
                    collected[name] = entry["metric"].mean()
                elif accumulate == "sum":
                    collected[name] = entry["metric"].sum()
                else:
                    raise ValueError(f"Unknown accumulation method: {accumulate}")

        return collected

    def collect_all(self, accumulate: Literal["mean", "sum"] = "mean") -> Dict[str, Union[float, str]]:
        """Collect aggregated values for all registered metrics.

        Args:
            accumulate: Aggregation mode; either ``"mean"`` or ``"sum"``.

        Returns:
            All current aggregated metric values keyed by name.
        """
        collected: Dict[str, Union[float, str]] = {}

        for name, entry in self._metrics.items():
            if accumulate == "mean":
                collected[name] = entry["metric"].mean()
            elif accumulate == "sum":
                collected[name] = entry["metric"].sum()
            else:
                raise ValueError(f"Unknown accumulation method: {accumulate}")

        return collected
