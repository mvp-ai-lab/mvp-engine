import subprocess
import time
from collections import deque
from typing import Optional

import torch
import torch.nn as nn

from mvp_engine.distributed.utils import is_main_process
from mvp_engine.utils.log import simple_info


class Timer:
    """A timer utility for tracking per-batch time and estimating ETA.

    This timer records the time taken for each batch/iteration, maintains
    a running average over a sliding window, and can estimate the remaining
    time to completion.

    Attributes:
        window_size: Number of recent samples to use for computing averages.
        start_time: Timestamp when the timer was started/reset.
        batch_times: Deque storing recent batch durations.
        total_batches: Total number of batches (for ETA calculation).
        current_batch: Current batch index.

    Example:
        >>> timer = Timer(total_batches=1000, window_size=100)
        >>> timer.start()
        >>> for i in range(1000):
        ...     # do work
        ...     timer.tick()
        ...     print(f"Batch time: {timer.batch_time:.3f}s, ETA: {timer.eta_string}")
    """

    def __init__(
        self,
        total_batches: Optional[int] = None,
        window_size: int = 100,
    ):
        """Initialize the Timer.

        Args:
            total_batches: Total number of batches for the training run.
                If provided, enables ETA estimation.
            window_size: Number of recent samples to use for computing
                the running average. Default is 100.
        """
        self.window_size = window_size
        self.total_batches = total_batches
        self.batch_times: deque[float] = deque(maxlen=window_size)

        self._start_time: Optional[float] = None
        self._last_tick: Optional[float] = None
        self._current_batch: int = 0
        self._total_elapsed: float = 0.0

    def start(self) -> "Timer":
        """Start or reset the timer.

        Returns:
            Self for method chaining.
        """
        self._start_time = time.perf_counter()
        self._last_tick = self._start_time
        self._current_batch = 0
        self._total_elapsed = 0.0
        self.batch_times.clear()
        return self

    def tick(self, batch_increment: int = 1) -> float:
        """Record the completion of one or more batches.

        Args:
            batch_increment: Number of batches completed since last tick.
                Default is 1.

        Returns:
            Time elapsed since last tick in seconds.
        """
        if self._last_tick is None:
            self.start()

        current_time = time.perf_counter()
        elapsed = current_time - self._last_tick
        self._last_tick = current_time
        self._current_batch += batch_increment
        self._total_elapsed = current_time - self._start_time

        # Record per-batch time
        per_batch_time = elapsed / batch_increment if batch_increment > 0 else elapsed
        for _ in range(batch_increment):
            self.batch_times.append(per_batch_time)

        return elapsed

    def reset(self) -> "Timer":
        """Reset the timer to initial state.

        Returns:
            Self for method chaining.
        """
        return self.start()

    def set_progress(self, current_batch: int, total_batches: Optional[int] = None) -> None:
        """Manually set the current progress.

        Useful when resuming from a checkpoint.

        Args:
            current_batch: Current batch/step number.
            total_batches: Optional new total number of batches.
        """
        self._current_batch = current_batch
        if total_batches is not None:
            self.total_batches = total_batches

    @property
    def batch_time(self) -> float:
        """Average time per batch over the recent window.

        Returns:
            Average batch time in seconds, or 0.0 if no data.
        """
        if not self.batch_times:
            return 0.0
        return sum(self.batch_times) / len(self.batch_times)

    @property
    def batch_time_latest(self) -> float:
        """Time taken for the most recent batch.

        Returns:
            Latest batch time in seconds, or 0.0 if no data.
        """
        if not self.batch_times:
            return 0.0
        return self.batch_times[-1]

    @property
    def total_elapsed(self) -> float:
        """Total time elapsed since timer was started.

        Returns:
            Total elapsed time in seconds.
        """
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    @property
    def current_batch(self) -> int:
        """Current batch/iteration number.

        Returns:
            Current batch index.
        """
        return self._current_batch

    @property
    def remaining_batches(self) -> int:
        """Number of batches remaining.

        Returns:
            Remaining batches, or 0 if total_batches is not set.
        """
        if self.total_batches is None:
            return 0
        return max(0, self.total_batches - self._current_batch)

    @property
    def eta(self) -> float:
        """Estimated time of arrival (remaining time) in seconds.

        Returns:
            Estimated remaining time in seconds, or 0.0 if cannot estimate.
        """
        if self.total_batches is None or self.batch_time == 0:
            return 0.0
        return self.remaining_batches * self.batch_time

    @property
    def eta_string(self) -> str:
        """Human-readable ETA string.

        Returns:
            Formatted string like "1d 2h 30m 45s" or "N/A" if cannot estimate.
        """
        return format_time(self.eta) if self.eta > 0 else "N/A"

    @property
    def elapsed_string(self) -> str:
        """Human-readable elapsed time string.

        Returns:
            Formatted string like "1d 2h 30m 45s".
        """
        return format_time(self.total_elapsed)

    @property
    def throughput(self) -> float:
        """Batches per second (throughput).

        Returns:
            Throughput in batches/second, or 0.0 if no data.
        """
        if self.batch_time == 0:
            return 0.0
        return 1.0 / self.batch_time

    def get_stats(self) -> dict:
        """Get a dictionary of timing statistics.

        Returns:
            Dict containing batch_time, batch_time_latest, elapsed,
            eta, throughput, current_batch, and remaining_batches.
        """
        return {
            "batch_time": self.batch_time,
            "batch_time_latest": self.batch_time_latest,
            "elapsed": self.total_elapsed,
            "elapsed_str": self.elapsed_string,
            "eta": self.eta,
            "eta_str": self.eta_string,
            "throughput": self.throughput,
            "current_batch": self.current_batch,
            "remaining_batches": self.remaining_batches,
        }


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted string like "00:00:00"
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def freeze(module: nn.Module):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def find_optimizable_params(module: nn.Module):
    for p in module.parameters():
        if p.requires_grad:
            yield p


def get_device(index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        try:
            import torch_npu  # noqa: F401

            return torch.device(f"npu:{index}")
        except ImportError:
            return torch.device("cpu")


def get_git_info():
    branch = "None"
    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.STDOUT,
            )
            .strip()
            .decode("utf-8")
        )
    except subprocess.CalledProcessError:
        pass

    commit_hash = "None"
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT).decode("utf-8").strip()
        )
    except subprocess.CalledProcessError:
        pass

    return {"branch": branch, "commit_hash": commit_hash}


def calculate_model_size(model):
    if is_main_process():
        model_size = sum(p.numel() for p in model.parameters())
        simple_info(f" - Model size: {model_size / 1e9:.4f} B")
        trainable_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
        simple_info(f" - Trainable model size: {trainable_size / 1e9:.4f} B")
