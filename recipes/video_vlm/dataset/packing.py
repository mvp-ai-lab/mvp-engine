"""Packing utilities for the Video VLM recipe."""

from __future__ import annotations

import random
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from mvp_dataset.core import Assembler, RuntimeContext


@dataclass(slots=True)
class _OpenPack:
    """Mutable packing state for one in-flight packed sample."""

    samples: list[dict[str, Any]] = field(default_factory=list)
    total_length: int = 0
    insertion_order: int = 0


@dataclass(slots=True)
class _PendingSample:
    """One buffered sample and its cached token length."""

    sample: dict[str, Any]
    length: int


class PackedSampleAssembler(Assembler[dict[str, Any], dict[str, Any] | list[dict[str, Any]]]):
    """Assemble processed samples into longer packed sequences."""

    def __init__(
        self,
        *,
        max_length: int,
        selection_strategy: Literal["random", "best_fit"] = "best_fit",
        open_pack_limit: int = 8,
        pack_buffer_size: int = 64,
        seed: int = 0,
        defer_finalize: bool = False,
    ) -> None:
        """Configure the streaming packer and its sample-selection strategy."""
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}.")
        if open_pack_limit <= 0:
            raise ValueError(f"open_pack_limit must be positive, got {open_pack_limit}.")
        if pack_buffer_size < 0:
            raise ValueError(f"pack_buffer_size must be non-negative, got {pack_buffer_size}.")
        if selection_strategy not in {"random", "best_fit"}:
            raise ValueError(f"selection_strategy must be one of random/best_fit, got {selection_strategy!r}.")

        self.max_length = max_length
        self.selection_strategy = selection_strategy
        self.open_pack_limit = open_pack_limit
        self.pack_buffer_size = pack_buffer_size
        self.keep_pending_sorted = self.selection_strategy != "random"
        self.rng = random.Random(seed)
        self.open_packs: list[_OpenPack] = []
        self.open_pack_remaining: list[int] = []
        self.pending_samples: list[_PendingSample] = []
        self.next_open_pack_order = 0
        self.defer_finalize = defer_finalize

    def push(self, sample: dict[str, Any]) -> Iterable[dict[str, Any] | list[dict[str, Any]]]:
        """Buffer one processed sample and emit any packs made ready by it."""
        sample_length = int(sample["input_ids"].size(0))
        if sample_length <= 0:
            return []

        if sample_length >= self.max_length:
            return [self._pack_samples([sample])]

        pending = _PendingSample(sample=sample, length=sample_length)
        if not self.keep_pending_sorted:
            self.pending_samples.append(pending)
        else:
            insert_index = len(self.pending_samples)
            while insert_index > 0 and self.pending_samples[insert_index - 1].length < sample_length:
                insert_index -= 1
            self.pending_samples.insert(insert_index, pending)

        if len(self.pending_samples) > self.pack_buffer_size:
            return self._drain_pool_to_buffer_limit()
        return []

    def finish(self, *, drop_last: bool = False) -> Iterable[dict[str, Any] | list[dict[str, Any]]]:
        """Flush buffered samples and optionally drop unfinished open packs."""
        emitted: list[dict[str, Any] | list[dict[str, Any]]] = []
        emitted.extend(self._drain_pool_to_buffer_limit(buffer_limit=0))

        while self.pending_samples:
            if not self.open_packs:
                self._open_pack_from_pending()
            emitted.append(self._close_most_filled_pack())
            emitted.extend(self._assign_pending_to_packs())
            emitted.extend(self._flush_ready_packs())

        emitted.extend(self._flush_ready_packs())

        if not drop_last:
            if self.selection_strategy == "best_fit":
                for pack in sorted(self.open_packs, key=lambda pack: pack.insertion_order):
                    emitted.append(self._pack_samples(pack.samples))
            else:
                while self.open_packs:
                    emitted.append(self._pack_samples(self._pop_open_pack(0).samples))

        self.pending_samples.clear()
        self.open_packs.clear()
        self.open_pack_remaining.clear()
        return emitted

    def _assign_pending_to_packs(self) -> list[dict[str, Any] | list[dict[str, Any]]]:
        """Assign pending samples into existing or newly opened packs."""
        if not self.pending_samples:
            return []

        max_length = self.max_length
        open_pack_limit = self.open_pack_limit
        is_random = self.selection_strategy == "random"
        rng = self.rng
        open_packs = self.open_packs
        emitted: list[dict[str, Any] | list[dict[str, Any]]] = []
        made_progress = True

        while self.pending_samples and made_progress:
            made_progress = False
            pending = self.pending_samples
            indices = range(len(pending))
            if is_random:
                indices = list(indices)
                rng.shuffle(indices)

            assigned_indices: list[int] = []
            for idx in indices:
                sample_length = pending[idx].length

                chosen_pack: _OpenPack | None = None
                if is_random:
                    candidate_count = 0
                    for pack in open_packs:
                        if max_length - pack.total_length >= sample_length:
                            candidate_count += 1
                            if rng.randrange(candidate_count) == 0:
                                chosen_pack = pack
                else:
                    chosen_index = bisect_left(self.open_pack_remaining, sample_length)
                    if chosen_index < len(open_packs):
                        chosen_pack = open_packs[chosen_index]

                if chosen_pack is not None:
                    chosen_pack.samples.append(pending[idx].sample)
                    chosen_pack.total_length += sample_length
                    if not is_random:
                        self.open_pack_remaining[chosen_index] = self.max_length - chosen_pack.total_length
                        self._restore_best_fit_order(chosen_index)
                    assigned_indices.append(idx)
                    made_progress = True
                elif len(open_packs) < open_pack_limit:
                    self._add_open_pack(_OpenPack(samples=[pending[idx].sample], total_length=sample_length))
                    assigned_indices.append(idx)
                    made_progress = True

            for idx in sorted(assigned_indices, reverse=True):
                del pending[idx]

            emitted.extend(self._flush_ready_packs())

        return emitted

    def _flush_ready_packs(self) -> list[dict[str, Any] | list[dict[str, Any]]]:
        """Emit packs that have exactly reached ``max_length``."""
        emitted: list[dict[str, Any] | list[dict[str, Any]]] = []
        if self.selection_strategy == "best_fit":
            while self.open_pack_remaining and self.open_pack_remaining[0] == 0:
                emitted.append(self._pack_samples(self._pop_open_pack(0).samples))
            return emitted

        remaining: list[_OpenPack] = []
        for pack in self.open_packs:
            if pack.total_length == self.max_length:
                emitted.append(self._pack_samples(pack.samples))
            else:
                remaining.append(pack)
        self.open_packs = remaining
        return emitted

    def _close_most_filled_pack(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Finalize and remove the most-filled currently open pack."""
        if not self.open_packs:
            raise RuntimeError("Cannot close a pack when no open packs exist.")

        if self.selection_strategy == "best_fit":
            return self._pack_samples(self._pop_open_pack(0).samples)

        chosen_index = min(
            range(len(self.open_packs)),
            key=lambda idx: self.max_length - self.open_packs[idx].total_length,
        )
        pack = self._pop_open_pack(chosen_index)
        return self._pack_samples(pack.samples)

    def _open_pack_from_pending(self) -> None:
        """Start a new open pack from one pending sample."""
        if not self.pending_samples:
            raise RuntimeError("Cannot open a pack from an empty sample pool.")

        sample_index = self.rng.randrange(len(self.pending_samples)) if self.selection_strategy == "random" else 0
        pending = self.pending_samples.pop(sample_index)
        self._add_open_pack(_OpenPack(samples=[pending.sample], total_length=pending.length))

    def _drain_pool_to_buffer_limit(
        self,
        *,
        buffer_limit: int | None = None,
    ) -> list[dict[str, Any] | list[dict[str, Any]]]:
        """Emit packs until the pending-sample pool is within the buffer limit."""
        if buffer_limit is None:
            buffer_limit = self.pack_buffer_size

        emitted: list[dict[str, Any] | list[dict[str, Any]]] = []
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
        """Insert an open pack and maintain best-fit ordering when needed."""
        pack.insertion_order = self.next_open_pack_order
        self.next_open_pack_order += 1
        self.open_packs.append(pack)
        if self.selection_strategy == "best_fit":
            self.open_pack_remaining.append(self.max_length - pack.total_length)
            self._restore_best_fit_order(len(self.open_packs) - 1)

    def _pop_open_pack(self, index: int) -> _OpenPack:
        """Remove and return an open pack by index."""
        pack = self.open_packs.pop(index)
        if self.selection_strategy == "best_fit":
            del self.open_pack_remaining[index]
        return pack

    def _restore_best_fit_order(self, index: int) -> None:
        """Move one best-fit pack until remaining-capacity order is restored."""
        if self.selection_strategy != "best_fit":
            return

        pack = self.open_packs[index]
        remaining = self.open_pack_remaining[index]
        while index > 0 and self.open_pack_remaining[index - 1] > remaining:
            self.open_packs[index] = self.open_packs[index - 1]
            self.open_pack_remaining[index] = self.open_pack_remaining[index - 1]
            index -= 1

        self.open_packs[index] = pack
        self.open_pack_remaining[index] = remaining

    def _pack_samples(self, samples: list[dict[str, Any]]) -> dict[str, Any] | list[dict[str, Any]]:
        """Return deferred sample groups or a finalized packed sample."""
        if self.defer_finalize:
            return [dict(sample) for sample in samples]
        return finalize_packed_sample_group(samples)


def finalize_packed_sample_group(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert a packed sample group into the model-facing packed sample dict."""
    if not samples:
        raise ValueError("Cannot finalize an empty packed sample group.")

    packed_sample: dict[str, Any] = {
        "input_ids": torch.cat([sample["input_ids"] for sample in samples], dim=0),
        "attention_mask": torch.cat([sample["attention_mask"] for sample in samples], dim=0),
        "labels": torch.cat([sample["labels"] for sample in samples], dim=0),
        "pack_segment_ids": torch.cat(
            [
                torch.full_like(sample["input_ids"], fill_value=index + 1, dtype=torch.long)
                for index, sample in enumerate(samples)
            ],
            dim=0,
        ),
    }

    pixel_values = [sample["pixel_values"] for sample in samples if sample.get("pixel_values") is not None]
    if pixel_values:
        packed_sample["pixel_values"] = torch.cat(pixel_values, dim=0)
    else:
        packed_sample["pixel_values"] = None

    image_grid_thw = [sample["image_grid_thw"] for sample in samples if sample.get("image_grid_thw") is not None]
    if image_grid_thw:
        packed_sample["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
    else:
        packed_sample["image_grid_thw"] = None

    packed_sample["source_sample_num"] = len(samples)

    return packed_sample


def build_packed_sample_assembler(
    assemble_context: RuntimeContext,
    *,
    max_length: int,
    selection_strategy: str,
    open_pack_limit: int,
    pack_buffer_size: int,
    defer_finalize: bool = False,
) -> PackedSampleAssembler:
    """Build one packing assembler instance for a dataset iterator."""
    return PackedSampleAssembler(
        max_length=max_length,
        selection_strategy=selection_strategy,
        open_pack_limit=open_pack_limit,
        pack_buffer_size=pack_buffer_size,
        seed=assemble_context.sample_shuffle_seed,
        defer_finalize=defer_finalize,
    )
