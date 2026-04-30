"""Shared runtime patch types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PatchStatus = Literal["applied", "skipped", "failed"]


@dataclass(frozen=True)
class PatchResult:
    """Result from applying a runtime patch."""

    name: str
    status: PatchStatus
    reason: str
