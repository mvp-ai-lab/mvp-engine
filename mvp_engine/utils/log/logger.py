"""Lightweight logging wrapper integrating log backends and metrics.

This module exposes `Logger` which coordinates metric aggregation
(`MetricAggregator`) and forwards messages/metrics to registered
`Backend` implementations.

Types:
- Metric values can be numeric (`float`) or `str` (last-value metrics).
"""

import os
from enum import IntEnum
from typing import List, Mapping, Optional, Union

import torch
import torch.distributed as dist

from mvp_engine.distributed.utils import get_world_size
from mvp_engine.utils.log.backend.backend import Backend
from mvp_engine.utils.log.metric import MetricAggregator


class LogLevel(IntEnum):
    """Severity order for logger message filtering."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40


LOG_LEVEL_ALIASES = {
    "debug": LogLevel.DEBUG,
    "info": LogLevel.INFO,
    "warn": LogLevel.WARNING,
    "warning": LogLevel.WARNING,
    "error": LogLevel.ERROR,
}


def parse_log_level(level: str) -> LogLevel:
    """Convert a string log level to ``LogLevel``."""
    normalized_level = level.strip().lower()
    if normalized_level not in LOG_LEVEL_ALIASES:
        raise ValueError(f"Invalid log level '{level}'. Supported levels: {', '.join(LOG_LEVEL_ALIASES.keys())}.")
    return LOG_LEVEL_ALIASES[normalized_level]


class Logger:
    """Coordinator for metric aggregation and backend forwarding.

    Args:
        backends: List of `Backend` instances that receive logs/metrics.
        interval: Default aggregation interval (passed to MetricAggregator).
    """

    def __init__(
        self,
        backends: List[Backend],
        interval: int = 20,
        accumulation_size: int = 20,
        level: LogLevel = LogLevel.INFO,
    ) -> None:
        if get_world_size() > 1:
            os.environ["GLOO_LOG_LEVEL"] = "ERROR"
            gloo_group = dist.new_group(backend="gloo")
        else:
            gloo_group = None

        self.metrics = MetricAggregator(
            dist_group=gloo_group,
            default_interval=interval,
            default_accumulation_size=accumulation_size,
        )
        self.step: int = 0
        self.total_steps: Optional[int] = None
        self.backends: List[Backend] = backends
        self.level: LogLevel = level

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
            backend.log_metrics(collected, self.step, total_steps=self.total_steps)

    def add_metric(
        self,
        name: str,
        accumulation_size: Optional[int] = None,
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
        accumulation_size: Optional[int] = None,
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
        total_steps: Optional[int] = None,
    ) -> None:
        """Update aggregator with new values and forward aggregated metrics.

        Args:
            metrics: Mapping of metric names to latest observed values.
            step: Current training step.
            epoch: Optional epoch index.
            total_steps: Optional total number of training steps for progress display.
        """
        self.metrics.update(metrics)
        self.step = step
        effective_total_steps = total_steps if total_steps is not None else self.total_steps
        self.total_steps = effective_total_steps

        collected = self.metrics.collect(metric_names=list(metrics.keys()))

        if collected:
            for backend in self.backends:
                backend.log_metrics(collected, step, epoch, effective_total_steps)

    def destroy(self) -> None:
        """Call `destroy()` on all registered backends and clear the global instance."""
        for backend in self.backends:
            backend.destroy()

        # Avoid leaving a destroyed logger registered as the global instance.
        from mvp_engine.utils.log import logger as global_logger

        if global_logger._instance is self:
            global_logger._instance = None

    def debug(self, message: str) -> None:
        """Log a debug message via backends."""
        if self.level > LogLevel.DEBUG:
            return
        for backend in self.backends:
            backend.debug(message)

    def info(self, message: str) -> None:
        """Log an informational message via backends."""
        if self.level > LogLevel.INFO:
            return
        for backend in self.backends:
            backend.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message via backends."""
        if self.level > LogLevel.WARNING:
            return
        for backend in self.backends:
            backend.warning(message)

    def error(self, message: str) -> None:
        """Log an error message via backends."""
        if self.level > LogLevel.ERROR:
            return
        for backend in self.backends:
            backend.error(message)
