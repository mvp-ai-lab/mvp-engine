"""Declarative specs for MLLM data pipeline construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .media import MLLMMediaHandler
from .schema import MLLMSchemaHandler
from .tokenization import MLLMTokenizationHandler


@dataclass(frozen=True, slots=True)
class MLLMSourceSpec:
    """Source dataset settings, including resampling and reference resolution.

    Attributes:
        dataset_path: Path or URI passed to ``Dataset.from_source``.
        dataset_source: mvp-dataset source type, such as ``"lance"``.
        ref_columns: Source columns that should be resolved when ``resolve_refs`` is enabled.
        seed: Runtime seed passed to the dataset context.
        resample: Whether the source dataset should be infinite/resampled.
        resolve_refs: Whether the dataset should materialize reference columns before model-input conversion.
    """

    dataset_path: str
    dataset_source: str = "lance"
    ref_columns: tuple[str, ...] = ("images",)
    seed: int = 42
    resample: bool = True
    resolve_refs: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize source options.

        Raises:
            ValueError: If ``dataset_path`` is empty.
        """
        if not self.dataset_path:
            raise ValueError("MLLMSourceSpec.dataset_path must not be empty.")
        object.__setattr__(self, "ref_columns", tuple(self.ref_columns))


@dataclass(frozen=True, slots=True)
class MLLMSampleSpec:
    """Per-model handlers that define single-sample schema, media, and token behavior.

    Attributes:
        schema_handler: Converts source rows into canonical ``MLLMSegment`` and ``MLLMMediaSlot`` objects.
        media_handler: Renders media placeholders and loads model-specific media tensors.
        tokenization_handler: Converts rendered segments into ``input_ids``, ``labels``, and ``attention_mask``.
    """

    schema_handler: MLLMSchemaHandler
    media_handler: MLLMMediaHandler
    tokenization_handler: MLLMTokenizationHandler


@dataclass(frozen=True, slots=True)
class MLLMPackingSpec:
    """Sequence-packing configuration consumed by the mvp-dataset assembler.

    Attributes:
        max_seq_len: Maximum packed sequence length.
        algorithm: Logical packing algorithm name.
        selection_strategy: How pending samples are selected for open packs.
        open_pack_limit: Maximum number of in-flight packs.
        buffer_size: Number of pending samples tolerated before emitting packs.
        block_causal: Whether packed samples should use block-causal attention masks downstream.
        assembler_cls: Optional custom assembler class implementing the same push/finish contract.
    """

    max_seq_len: int
    algorithm: str = "multi_pack"
    selection_strategy: str = "best_fit"
    open_pack_limit: int = 8
    buffer_size: int = 64
    block_causal: bool = True
    assembler_cls: type[Any] | None = None

    def __post_init__(self) -> None:
        """Validate packing options.

        Raises:
            ValueError: If any packing option is outside the supported range.
        """
        if self.max_seq_len <= 0:
            raise ValueError("MLLMPackingSpec.max_seq_len must be positive.")
        if self.algorithm not in {"multi_pack", "packed"}:
            raise ValueError("MLLMPackingSpec.algorithm must be 'multi_pack' or 'packed'.")
        if self.selection_strategy not in {"random", "best_fit"}:
            raise ValueError("MLLMPackingSpec.selection_strategy must be one of random/best_fit.")
        if self.open_pack_limit <= 0:
            raise ValueError("MLLMPackingSpec.open_pack_limit must be positive.")
        if self.buffer_size < 0:
            raise ValueError("MLLMPackingSpec.buffer_size must be non-negative.")


@dataclass(frozen=True, slots=True)
class MLLMLoaderSpec:
    """TorchLoader batching options for an MLLM data pipeline.

    Attributes:
        batch_size: Number of packed samples per batch.
        num_workers: TorchLoader worker count.
        pin_memory: Optional explicit pin-memory setting; ``None`` lets ``MLLMDataKit`` infer it from device.
        persistent_workers: Whether worker processes stay alive across epochs.
        multiprocessing_context: Multiprocessing start method passed to TorchLoader.
        drop_last: Whether the final partial batch is dropped.
    """

    batch_size: int
    num_workers: int = 0
    pin_memory: bool | None = None
    persistent_workers: bool = False
    multiprocessing_context: str = "spawn"
    drop_last: bool = True

    def __post_init__(self) -> None:
        """Validate loader options.

        Raises:
            ValueError: If ``batch_size`` or ``num_workers`` is invalid.
        """
        if self.batch_size <= 0:
            raise ValueError("MLLMLoaderSpec.batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("MLLMLoaderSpec.num_workers must be non-negative.")


@dataclass(frozen=True, slots=True)
class MLLMDistributionSpec:
    """Distributed placement options passed into the mvp-dataset runtime context.

    Attributes:
        device_mesh: Optional mesh object used by distributed dataset sharding.
        dp_dims: Mesh dimension name or names that represent data parallelism.
    """

    device_mesh: object | None = None
    dp_dims: str | Sequence[str] | None = None


@dataclass(frozen=True, slots=True)
class MLLMDataSpec:
    """Complete declarative input consumed by ``MLLMDataKit``.

    Attributes:
        source: Dataset source and reference-resolution settings.
        sample: Per-model single-sample handlers.
        packing: Sequence-packing settings.
        loader: Batch-loader settings.
        distribution: Optional distributed placement settings.
    """

    source: MLLMSourceSpec
    sample: MLLMSampleSpec
    packing: MLLMPackingSpec
    loader: MLLMLoaderSpec
    distribution: MLLMDistributionSpec | None = None
