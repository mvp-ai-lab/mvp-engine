"""Post-pack skip helpers for OpenBee fast resume."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from mvp_dataset.core import Assembler, RuntimeContext

SkipMode = Literal["off", "pre_calculate", "perform"]


class SkipRecorder(Assembler[Any, dict[str, int]]):
    """Replace each post-pack output with a lightweight worker-slot marker."""

    def __init__(self, *, worker_slot: int) -> None:
        self.worker_slot = int(worker_slot)

    def push(self, sample: Any) -> Iterable[dict[str, int]]:
        del sample
        return [{"worker_slot": self.worker_slot}]

    def finish(self, *, drop_last: bool = False) -> Iterable[dict[str, int]]:
        del drop_last
        return []


class SkipByWorker(Assembler[Any, Any]):
    """Drop the first N post-pack outputs assigned to this worker slot."""

    def __init__(self, *, worker_slot: int, skip_counts: Mapping[int, int]) -> None:
        self.worker_slot = int(worker_slot)
        normalized_counts: dict[int, int] = {}
        for slot, count in skip_counts.items():
            if int(count) < 0:
                raise ValueError(f"skip_counts values must be non-negative, got {int(count)}.")
            normalized_counts[int(slot)] = int(count)
        self.remaining = normalized_counts.get(self.worker_slot, 0)

    def push(self, sample: Any) -> Iterable[Any]:
        if self.remaining > 0:
            self.remaining -= 1
            return []
        return [sample]

    def finish(self, *, drop_last: bool = False) -> Iterable[Any]:
        del drop_last
        return []


def build_skip_recorder(assemble_context: RuntimeContext) -> SkipRecorder:
    """Create a worker-local post-pack skip recorder."""

    return SkipRecorder(worker_slot=assemble_context.slot)


def build_skip_by_worker(assemble_context: RuntimeContext, skip_counts: Mapping[int, int]) -> SkipByWorker:
    """Create a worker-local post-pack skipper."""

    return SkipByWorker(worker_slot=assemble_context.slot, skip_counts=skip_counts)
