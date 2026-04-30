"""Metric accumulation helpers for the PanguVL recipe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

ReducerName = Literal["sum", "last", "max", "mean"]


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
        if isinstance(value, int | float):
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
