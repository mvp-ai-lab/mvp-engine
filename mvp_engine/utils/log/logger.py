"""Lightweight logging wrapper integrating log backends and metrics.

This module exposes `Logger` which coordinates metric aggregation
(`MetricAggregator`) and forwards messages/metrics to registered
`Backend` implementations.

Types:
- Metric values can be numeric (`float`) or `str` (last-value metrics).
"""

from typing import List, Mapping, Optional, Union

import torch
import torch.distributed as dist

from mvp_engine.utils.distributed.utils import get_world_size
from mvp_engine.utils.log.backend.backend import Backend
from mvp_engine.utils.log.metric import MetricAggregator


class Logger:
    """Coordinator for metric aggregation and backend forwarding.

    Args:
        backends: List of `Backend` instances that receive logs/metrics.
        interval: Default aggregation interval (passed to MetricAggregator).
    """

    def __init__(self, backends: List[Backend], interval: int = 20) -> None:
        if get_world_size() > 1:
            gloo_group = dist.new_group(backend="gloo")
        else:
            gloo_group = None

        self.metrics = MetricAggregator(dist_group=gloo_group, default_interval=interval)
        self.step: int = 0
        self.backends: List[Backend] = backends

    def log_config(self, config: dict) -> None:
        """Forward configuration dict to all backends.

        Args:
            config: Configuration mapping (usually from OmegaConf).
        """
        for backend in self.backends:
            backend.log_config(config)

    def summary(self) -> None:
        """Collect all metrics and forward them using the last known step."""
        collected = self.metrics.collect_all()
        for backend in self.backends:
            backend.log_metrics(collected, self.step)

    def add_metric(
        self,
        name: str,
        accumulation_size: int = 20,
        interval: Optional[int] = None,
        distributed: Optional[bool] = None,
        support_nan: bool = True,
    ) -> None:
        """Register a single metric for tracking.

        Args:
            name: Metric name.
            accumulation_size: Buffer size for the metric.
            interval: Optional reporting interval (defaults to logger interval).
            distributed: Whether to perform distributed reduction.
            support_nan: Whether NaN values are allowed.
        """
        self.metrics.add(
            name=name,
            accumulation_size=accumulation_size,
            interval=interval,
            distributed=distributed,
            support_nan=support_nan,
        )

    def add_metrics(
        self,
        names: List[str],
        accumulation_size: int = 20,
        interval: Optional[int] = None,
        distributed: Optional[bool] = None,
        support_nan: bool = True,
    ) -> None:
        """Register multiple metrics by name."""
        for name in names:
            self.add_metric(
                name=name,
                accumulation_size=accumulation_size,
                interval=interval,
                distributed=distributed,
                support_nan=support_nan,
            )

    def log_metrics(
        self,
        metrics: Mapping[str, Union[int, float, str, torch.Tensor]],
        step: int,
        epoch: Optional[int] = None,
    ) -> None:
        """Update aggregator with new values and forward aggregated metrics.

        Args:
            metrics: Mapping of metric names to latest observed values.
            step: Current training step.
            epoch: Optional epoch index.
        """
        self.metrics.update(metrics)
        self.step = step

        collected = self.metrics.collect(metric_names=list(metrics.keys()))

        if collected:
            for backend in self.backends:
                backend.log_metrics(collected, step, epoch)

    def destroy(self) -> None:
        """Call `destroy()` on all registered backends."""
        for backend in self.backends:
            backend.destroy()

    def info(self, message: str) -> None:
        """Log an informational message via backends."""
        for backend in self.backends:
            backend.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message via backends."""
        for backend in self.backends:
            backend.warning(message)

    def error(self, message: str) -> None:
        """Log an error message via backends."""
        for backend in self.backends:
            backend.error(message)
