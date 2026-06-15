"""Invalid-sample gating utilities for the PanguVL recipe."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from mvp_dataset.core import Assembler, RuntimeContext
from mvp_dataset.core.resume import ResumeStateError, stable_fingerprint

from .types import ModelInputs


def build_skipped_sample() -> ModelInputs:
    """Return an empty sample sentinel that downstream stages can ignore safely."""
    return {
        "input_ids": torch.empty(0, dtype=torch.long),
        "attention_mask": torch.empty(0, dtype=torch.long),
        "labels": torch.empty(0, dtype=torch.long),
    }


class InvalidSampleGateAssembler(Assembler[dict[str, Any], dict[str, Any]]):
    """Block invalid PanguVL sample sentinels before batching.

    ``process_sample`` returns an empty-tensor sentinel for invalid rows so the
    dataset pipeline can continue. This gate removes those sentinels from the
    stream before the dataloader batches samples, which avoids propagating
    ``None`` batches into the training loop.
    """

    def push(self, sample: dict[str, Any]) -> Iterable[dict[str, Any]]:
        if int(sample["input_ids"].size(0)) <= 0:
            return []
        return [sample]

    def finish(self, *, drop_last: bool = False) -> Iterable[dict[str, Any]]:
        del drop_last
        return []

    def state_dict(self) -> dict[str, object]:
        """Return the resumable state for this stateless gate."""
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore the gate from a resumable state dictionary."""
        if state:
            raise ResumeStateError("[InvalidResumeState] invalid-sample gate state must be empty")

    def fingerprint(self) -> str:
        """Return a stable fingerprint for resume compatibility checks."""
        return stable_fingerprint({"kind": "panguvl-invalid-sample-gate", "version": 1})


def build_invalid_sample_gate_assembler(assemble_context: RuntimeContext) -> InvalidSampleGateAssembler:
    """Build an assembler that gates invalid-sample sentinels out of the stream."""
    del assemble_context
    return InvalidSampleGateAssembler()
