import time
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional


class Timer:
    """Track global progress ETA and scoped duration samples."""

    def __init__(self, total_progress: Optional[float] = None, window_size: int = 100):
        self.window_size = window_size
        self.total_progress = total_progress
        self.current_progress = 0.0
        self.progress_times: deque[float] = deque(maxlen=window_size)
        self.scope_times: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window_size))

        self._start_time: Optional[float] = None
        self._last_progress_tick: Optional[float] = None
        self._scope_start_times: dict[str, float] = {}

    def start(self) -> "Timer":
        """Start or reset the timer."""
        start_time = time.perf_counter()
        self._start_time = start_time
        self._last_progress_tick = start_time
        self.current_progress = 0.0
        self.progress_times.clear()
        self.scope_times.clear()
        self._scope_start_times.clear()
        return self

    def reset(self) -> "Timer":
        """Reset the timer to initial state."""
        return self.start()

    def set_progress(self, current_progress: float, total_progress: Optional[float] = None) -> None:
        """Manually set current and optional total progress."""
        self.current_progress = current_progress
        if total_progress is not None:
            self.total_progress = total_progress

    def tick(self, progress_increment: float = 1.0) -> float:
        """Advance global progress and record time per progress unit."""
        if progress_increment < 0:
            raise ValueError("progress_increment must be non-negative.")
        if self._last_progress_tick is None:
            self.start()

        current_time = time.perf_counter()
        elapsed = current_time - self._last_progress_tick
        self._last_progress_tick = current_time

        if progress_increment > 0:
            self.current_progress += progress_increment
            self.progress_times.append(elapsed / progress_increment)

        return elapsed

    def start_scope(self, scope: str) -> None:
        """Start timing a scope."""
        self._scope_start_times[scope] = time.perf_counter()

    def end_scope(self, scope: str) -> float:
        """End timing a scope and record the elapsed duration."""
        start_time = self._scope_start_times.pop(scope, None)
        if start_time is None:
            raise ValueError(f"Scope '{scope}' has not been started.")

        elapsed = time.perf_counter() - start_time
        self.scope_times[scope].append(elapsed)
        return elapsed

    @contextmanager
    def scope(self, scope: str) -> Iterator[None]:
        """Record elapsed time for a scope around a block."""
        self.start_scope(scope)
        try:
            yield
        finally:
            self.end_scope(scope)

    def record_scope(self, scope: str, elapsed: float) -> None:
        """Record a pre-measured scope duration."""
        if elapsed < 0:
            raise ValueError("elapsed must be non-negative.")
        self.scope_times[scope].append(elapsed)

    @property
    def total_elapsed(self) -> float:
        """Total elapsed time since timer start."""
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    @property
    def remaining_progress(self) -> float:
        """Remaining global progress before completion."""
        if self.total_progress is None:
            return 0.0
        return max(0.0, self.total_progress - self.current_progress)

    @property
    def progress_time(self) -> float:
        """Average seconds per global progress unit over the recent window."""
        if not self.progress_times:
            return 0.0
        return sum(self.progress_times) / len(self.progress_times)

    @property
    def progress_time_latest(self) -> float:
        """Latest seconds per global progress unit sample."""
        if not self.progress_times:
            return 0.0
        return self.progress_times[-1]

    @property
    def eta(self) -> float:
        """Estimated remaining time in seconds."""
        if self.total_progress is None or self.progress_time == 0:
            return 0.0
        return self.remaining_progress * self.progress_time

    @property
    def eta_string(self) -> str:
        """Human-readable ETA string."""
        return format_time(self.eta) if self.eta > 0 else "N/A"

    @property
    def elapsed_string(self) -> str:
        """Human-readable elapsed time string."""
        return format_time(self.total_elapsed)

    def get_scope_time(self, scope: str) -> float:
        """Average elapsed time for a scope."""
        times = self.scope_times.get(scope)
        if not times:
            return 0.0
        return sum(times) / len(times)

    def get_scope_time_latest(self, scope: str) -> float:
        """Latest elapsed time sample for a scope."""
        times = self.scope_times.get(scope)
        if not times:
            return 0.0
        return times[-1]

    def get_stats(self) -> dict:
        """Get progress and scope timing statistics."""
        return {
            "elapsed": self.total_elapsed,
            "elapsed_str": self.elapsed_string,
            "eta": self.eta,
            "eta_str": self.eta_string,
            "current_progress": self.current_progress,
            "remaining_progress": self.remaining_progress,
            "progress_time": self.progress_time,
            "progress_time_latest": self.progress_time_latest,
            "scope_times": {scope: self.get_scope_time(scope) for scope in self.scope_times},
            "scope_times_latest": {scope: self.get_scope_time_latest(scope) for scope in self.scope_times},
        }


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
