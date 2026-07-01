"""Sequence packing utilities for multimodal language model data."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from mvp_dataset.core import Assembler

from .sample import MLLMPack, MLLMSample
from .spec import MLLMPackingSpec

try:
    from mvp_dataset.core.resume import stable_fingerprint
except ImportError:

    def stable_fingerprint(value: object) -> str:
        """Return a stable JSON fingerprint when the installed mvp_dataset lacks one."""
        encoded = json.dumps(_normalize_fingerprint_value(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _normalize_fingerprint_value(value: object) -> object:
    """Normalize common Python values into a deterministic JSON payload."""
    if dataclasses.is_dataclass(value):
        return _normalize_fingerprint_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalize_fingerprint_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_fingerprint_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(slots=True)
class _OpenPack:
    """Mutable packing state for one in-flight packed sample."""

    samples: list[MLLMSample] = field(default_factory=list)
    total_length: int = 0
    insertion_order: int = 0


@dataclass(slots=True)
class _PendingSample:
    """One buffered sample and its cached token length."""

    sample: MLLMSample
    length: int


class MLLMPackingAssembler(Assembler[MLLMSample, MLLMPack]):
    """Pack tokenized MLLM samples into longer training sequences.

    The assembler follows the mvp-dataset streaming assembler contract: ``push``
    receives one sample at a time, and ``finish`` flushes buffered state at the end
    of a finite stream.
    """

    def __init__(
        self,
        spec: MLLMPackingSpec,
        assemble_context: Any | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        """Configure the streaming packer and its sample-selection strategy.

        Args:
            spec: Packing configuration.
            assemble_context: Optional mvp-dataset assembler context. Its
                ``sample_shuffle_seed`` is used when ``seed`` is omitted.
            seed: Optional explicit RNG seed for random sample selection.
        """
        if seed is None:
            seed = getattr(assemble_context, "sample_shuffle_seed", 0)

        self.spec = spec
        self.rng = random.Random(seed)
        self.open_packs: list[_OpenPack] = []
        self.pending_samples: list[_PendingSample] = []
        self.next_open_pack_order = 0

    def push(self, sample: MLLMSample) -> Iterable[MLLMPack]:
        """Buffer one processed sample and emit any packs made ready by it.

        Args:
            sample: Tokenized sample to place into a pack.

        Returns:
            Packs that became ready after accepting the sample.

        Raises:
            ValueError: If the sample is longer than ``spec.max_seq_len``.
        """
        sample_length = sample.token_length
        if sample_length <= 0:
            return []
        if sample_length > self.spec.max_seq_len:
            raise ValueError(f"Sample length {sample_length} exceeds max_seq_len {self.spec.max_seq_len}.")
        if sample_length == self.spec.max_seq_len:
            return [self._pack_samples([sample])]

        pending = _PendingSample(sample=sample, length=sample_length)
        if self.spec.selection_strategy == "random":
            self.pending_samples.append(pending)
        else:
            insert_index = len(self.pending_samples)
            while insert_index > 0 and self.pending_samples[insert_index - 1].length < sample_length:
                insert_index -= 1
            self.pending_samples.insert(insert_index, pending)

        if len(self.pending_samples) > self.spec.buffer_size:
            return self._drain_pool_to_buffer_limit()
        return []

    def finish(self, *, drop_last: bool = False) -> Iterable[MLLMPack]:
        """Flush buffered samples and optionally drop unfinished open packs.

        Args:
            drop_last: Whether to discard incomplete packs when the finite stream ends.

        Returns:
            Remaining packs emitted from buffered state.
        """
        emitted: list[MLLMPack] = []
        if drop_last:
            pending_count = -1
            while self.pending_samples and pending_count != len(self.pending_samples):
                pending_count = len(self.pending_samples)
                if len(self.open_packs) < self.spec.open_pack_limit:
                    self._open_pack_from_pending()
                emitted.extend(self._assign_pending_to_packs())
                emitted.extend(self._flush_ready_packs())
            emitted.extend(self._flush_ready_packs())
        else:
            emitted.extend(self._drain_pool_to_buffer_limit(buffer_limit=0))
            for pack in sorted(self.open_packs, key=lambda pack: pack.insertion_order):
                emitted.append(self._pack_samples(pack.samples))

        self.pending_samples.clear()
        self.open_packs.clear()
        return emitted

    def state_dict(self) -> dict[str, object]:
        """Return resumable packing state.

        Returns:
            Serialized open packs, pending samples, insertion counter, and RNG state.
        """
        return {
            "open_packs": list(self.open_packs),
            "pending_samples": list(self.pending_samples),
            "next_open_pack_order": self.next_open_pack_order,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore packing state.

        Args:
            state: State previously returned by ``state_dict``.
        """
        self.open_packs = list(state.get("open_packs", []))
        self.pending_samples = list(state.get("pending_samples", []))
        self.next_open_pack_order = int(state.get("next_open_pack_order", 0))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.setstate(rng_state)

    def fingerprint(self) -> str:
        """Return a stable resume fingerprint for this assembler configuration.

        Returns:
            Stable string fingerprint used by mvp-dataset resume checks.
        """
        return stable_fingerprint(
            {
                "algorithm": self.spec.algorithm,
                "max_seq_len": self.spec.max_seq_len,
                "selection_strategy": self.spec.selection_strategy,
                "open_pack_limit": self.spec.open_pack_limit,
                "buffer_size": self.spec.buffer_size,
                "block_causal": self.spec.block_causal,
            }
        )

    def _assign_pending_to_packs(self) -> list[MLLMPack]:
        """Place buffered samples into open packs and emit packs that become ready."""
        if not self.pending_samples:
            return []

        emitted: list[MLLMPack] = []
        made_progress = True
        while self.pending_samples and made_progress:
            made_progress = False
            indices: range | list[int] = range(len(self.pending_samples))
            if self.spec.selection_strategy == "random":
                indices = list(indices)
                self.rng.shuffle(indices)

            assigned_indices: list[int] = []
            for idx in indices:
                sample_length = self.pending_samples[idx].length
                chosen_index = self._choose_open_pack_index(sample_length)
                if chosen_index is not None:
                    chosen_pack = self.open_packs[chosen_index]
                    chosen_pack.samples.append(self.pending_samples[idx].sample)
                    chosen_pack.total_length += sample_length
                    assigned_indices.append(idx)
                    made_progress = True
                elif len(self.open_packs) < self.spec.open_pack_limit:
                    self._add_open_pack(
                        _OpenPack(samples=[self.pending_samples[idx].sample], total_length=sample_length)
                    )
                    assigned_indices.append(idx)
                    made_progress = True

            for idx in sorted(assigned_indices, reverse=True):
                del self.pending_samples[idx]
            emitted.extend(self._flush_ready_packs())

        return emitted

    def _flush_ready_packs(self) -> list[MLLMPack]:
        """Emit open packs that exactly reached the configured maximum length."""
        emitted: list[MLLMPack] = []
        remaining: list[_OpenPack] = []
        for pack in self.open_packs:
            if pack.total_length == self.spec.max_seq_len:
                emitted.append(self._pack_samples(pack.samples))
            else:
                remaining.append(pack)
        self.open_packs = remaining
        return emitted

    def _close_most_filled_pack(self) -> MLLMPack:
        """Close and return the currently fullest open pack."""
        if not self.open_packs:
            raise RuntimeError("Cannot close a pack when no open packs exist.")
        chosen_index = min(
            range(len(self.open_packs)),
            key=lambda idx: (
                self.spec.max_seq_len - self.open_packs[idx].total_length,
                self.open_packs[idx].insertion_order,
            ),
        )
        return self._pack_samples(self.open_packs.pop(chosen_index).samples)

    def _open_pack_from_pending(self) -> None:
        """Create one open pack from the pending-sample pool."""
        if not self.pending_samples:
            raise RuntimeError("Cannot open a pack from an empty sample pool.")
        sample_index = self.rng.randrange(len(self.pending_samples)) if self.spec.selection_strategy == "random" else 0
        pending = self.pending_samples.pop(sample_index)
        self._add_open_pack(_OpenPack(samples=[pending.sample], total_length=pending.length))

    def _drain_pool_to_buffer_limit(self, *, buffer_limit: int | None = None) -> list[MLLMPack]:
        """Emit packs until the pending pool is within the requested buffer limit."""
        if buffer_limit is None:
            buffer_limit = self.spec.buffer_size

        emitted: list[MLLMPack] = []
        emitted.extend(self._assign_pending_to_packs())
        emitted.extend(self._flush_ready_packs())

        while self.pending_samples and len(self.pending_samples) > buffer_limit:
            if not self.open_packs:
                self._open_pack_from_pending()
            emitted.append(self._close_most_filled_pack())
            emitted.extend(self._assign_pending_to_packs())
            emitted.extend(self._flush_ready_packs())

        return emitted

    def _add_open_pack(self, pack: _OpenPack) -> None:
        """Append a newly opened pack and assign its stable insertion order."""
        pack.insertion_order = self.next_open_pack_order
        self.next_open_pack_order += 1
        self.open_packs.append(pack)

    def _choose_open_pack_index(self, sample_length: int) -> int | None:
        """Return the open pack that should receive a sample of the given length."""
        candidate_indices = [
            index
            for index, pack in enumerate(self.open_packs)
            if self.spec.max_seq_len - pack.total_length >= sample_length
        ]
        if not candidate_indices:
            return None
        if self.spec.selection_strategy == "random":
            return self.rng.choice(candidate_indices)
        return min(
            candidate_indices,
            key=lambda index: (
                self.spec.max_seq_len - self.open_packs[index].total_length,
                self.open_packs[index].insertion_order,
            ),
        )

    def _pack_samples(self, samples: list[MLLMSample]) -> MLLMPack:
        """Wrap a non-empty sample group as one finalized MLLM pack."""
        if not samples:
            raise RuntimeError("Cannot build an empty MLLM pack.")
        return MLLMPack(samples=list(samples))


def build_packed_block_causal_mask(
    pack_segment_ids: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a 4D additive mask that isolates packed samples for eager/SDPA backends.

    Args:
        pack_segment_ids: 2D tensor whose equal nonzero ids mark tokens from the
            same source sample inside a packed sequence.
        dtype: Floating dtype for the additive attention mask.

    Returns:
        A mask of shape ``[batch, 1, seq, seq]`` with zero for allowed attention
        positions and the minimum representable value for blocked positions.

    Raises:
        ValueError: If ``pack_segment_ids`` is not 2D.
    """
    if pack_segment_ids.ndim != 2:
        raise ValueError(f"Expected 2D pack_segment_ids, got shape {tuple(pack_segment_ids.shape)}.")

    batch_size, sequence_length = pack_segment_ids.shape
    token_positions = torch.arange(sequence_length, device=pack_segment_ids.device)
    causal_mask = token_positions.unsqueeze(0) <= token_positions.unsqueeze(1)

    valid_tokens = pack_segment_ids.ne(0)
    same_segment = pack_segment_ids.unsqueeze(-1) == pack_segment_ids.unsqueeze(-2)
    allowed = valid_tokens.unsqueeze(-1) & valid_tokens.unsqueeze(-2) & same_segment & causal_mask.unsqueeze(0)

    min_dtype = torch.finfo(dtype).min
    attention_mask = torch.full(
        (batch_size, 1, sequence_length, sequence_length),
        min_dtype,
        dtype=dtype,
        device=pack_segment_ids.device,
    )
    attention_mask.masked_fill_(allowed.unsqueeze(1), 0)
    return attention_mask
