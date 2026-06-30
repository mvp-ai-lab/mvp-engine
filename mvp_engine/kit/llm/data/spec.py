"""Declarative specs for text-only LM data pipeline construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .schema import LLMSchemaHandler
from .tokenization import LLMTokenizationHandler


@dataclass(frozen=True, slots=True)
class LLMSourceSpec:
    """Source dataset settings for a text-only LM data pipeline."""

    dataset_path: str
    dataset_source: str = "lance"
    seed: int = 42
    resample: bool = True

    def __post_init__(self) -> None:
        """Validate source options."""
        if not self.dataset_path:
            raise ValueError("LLMSourceSpec.dataset_path must not be empty.")


@dataclass(frozen=True, slots=True)
class LLMSampleSpec:
    """Per-recipe handlers that define source schema and token behavior."""

    schema_handler: LLMSchemaHandler
    tokenization_handler: LLMTokenizationHandler


@dataclass(frozen=True, slots=True)
class LLMPackingSpec:
    """Token-stream packing configuration consumed by the text-LM packer."""

    max_seq_len: int
    tail_policy: Literal["drop", "pad"] = "drop"
    isolate_attention: bool = False
    isolate_position_ids: bool = False
    assembler_cls: type[Any] | None = None

    def __post_init__(self) -> None:
        """Validate packing options."""
        if self.max_seq_len <= 0:
            raise ValueError("LLMPackingSpec.max_seq_len must be positive.")
        if self.tail_policy not in {"drop", "pad"}:
            raise ValueError("LLMPackingSpec.tail_policy must be one of drop/pad.")


@dataclass(frozen=True, slots=True)
class LLMLoaderSpec:
    """TorchLoader batching options for a text-only LM data pipeline."""

    batch_size: int
    num_workers: int = 0
    pin_memory: bool | None = None
    persistent_workers: bool = False
    multiprocessing_context: str = "spawn"
    drop_last: bool = True

    def __post_init__(self) -> None:
        """Validate loader options."""
        if self.batch_size <= 0:
            raise ValueError("LLMLoaderSpec.batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("LLMLoaderSpec.num_workers must be non-negative.")


@dataclass(frozen=True, slots=True)
class LLMDistributionSpec:
    """Distributed placement options passed into the mvp-dataset runtime context."""

    device_mesh: object | None = None
    dp_dims: str | Sequence[str] | None = None


@dataclass(frozen=True, slots=True)
class LLMDataSpec:
    """Complete declarative input consumed by ``LLMDataKit``."""

    source: LLMSourceSpec
    sample: LLMSampleSpec
    packing: LLMPackingSpec
    loader: LLMLoaderSpec
    distribution: LLMDistributionSpec | None = None
