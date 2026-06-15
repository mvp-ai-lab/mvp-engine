"""Invalid-sample gating utilities for the OpenBee recipe."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from mvp_dataset.core import Assembler, RuntimeContext

from .types import ModelInputs


def build_skipped_sample() -> ModelInputs:
    """Return an empty sample sentinel that downstream stages can ignore safely."""
    return {
        "input_ids": torch.empty(0, dtype=torch.long),
        "attention_mask": torch.empty(0, dtype=torch.long),
        "labels": torch.empty(0, dtype=torch.long),
    }


class InvalidSampleGateAssembler(Assembler[dict[str, Any], dict[str, Any]]):
    """Block invalid OpenBee sample sentinels before batching.

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


def build_invalid_sample_gate_assembler(assemble_context: RuntimeContext) -> InvalidSampleGateAssembler:
    """Build an assembler that gates invalid-sample sentinels out of the stream."""
    del assemble_context
    return InvalidSampleGateAssembler()
