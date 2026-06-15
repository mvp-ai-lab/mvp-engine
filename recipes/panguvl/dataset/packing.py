"""Packing utilities for the PanguVL recipe."""

from __future__ import annotations

import random
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from mvp_dataset.core import Assembler, RuntimeContext
from mvp_dataset.core.resume import ResumeStateError, stable_fingerprint


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


class PackedSampleAssembler(Assembler[dict[str, Any], dict[str, Any]]):
    """Assemble processed samples into longer packed sequences."""

    def __init__(
        self,
        *,
        max_length: int,
        selection_strategy: Literal["random", "best_fit"] = "best_fit",
        open_pack_limit: int = 8,
        pack_buffer_size: int = 64,
        seed: int = 0,
    ) -> None:
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

    def push(self, sample: dict[str, Any]) -> Iterable[dict[str, Any]]:
        sample_length = self._get_sample_length(sample)
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

    def finish(self, *, drop_last: bool = False) -> Iterable[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
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

    def state_dict(self) -> dict[str, object]:
        """Return the resumable state for this streaming packer."""
        return {
            "pending_samples": [
                {
                    "sample": pending.sample,
                    "length": pending.length,
                }
                for pending in self.pending_samples
            ],
            "open_packs": [
                {
                    "samples": list(pack.samples),
                    "total_length": pack.total_length,
                    "insertion_order": pack.insertion_order,
                }
                for pack in self.open_packs
            ],
            "open_pack_remaining": list(self.open_pack_remaining),
            "next_open_pack_order": self.next_open_pack_order,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore this packer from a resumable state dictionary."""
        pending_samples = state.get("pending_samples")
        open_packs = state.get("open_packs")
        open_pack_remaining = state.get("open_pack_remaining")
        next_open_pack_order = state.get("next_open_pack_order")
        rng_state = state.get("rng_state")

        if not isinstance(pending_samples, list):
            raise ResumeStateError("[InvalidResumeState] packed assembler pending_samples must be a list")
        if not isinstance(open_packs, list):
            raise ResumeStateError("[InvalidResumeState] packed assembler open_packs must be a list")
        if not isinstance(open_pack_remaining, list):
            raise ResumeStateError("[InvalidResumeState] packed assembler open_pack_remaining must be a list")
        if not isinstance(next_open_pack_order, int):
            raise ResumeStateError("[InvalidResumeState] packed assembler next_open_pack_order must be an int")

        restored_pending: list[_PendingSample] = []
        for item in pending_samples:
            if not isinstance(item, dict):
                raise ResumeStateError("[InvalidResumeState] packed assembler pending item must be a dict")
            sample = item.get("sample")
            length = item.get("length")
            if not isinstance(sample, dict) or not isinstance(length, int):
                raise ResumeStateError("[InvalidResumeState] packed assembler pending item is malformed")
            restored_pending.append(_PendingSample(sample=sample, length=length))

        restored_packs: list[_OpenPack] = []
        for item in open_packs:
            if not isinstance(item, dict):
                raise ResumeStateError("[InvalidResumeState] packed assembler open pack must be a dict")
            samples = item.get("samples")
            total_length = item.get("total_length")
            insertion_order = item.get("insertion_order")
            if (
                not isinstance(samples, list)
                or not isinstance(total_length, int)
                or not isinstance(insertion_order, int)
            ):
                raise ResumeStateError("[InvalidResumeState] packed assembler open pack is malformed")
            restored_packs.append(
                _OpenPack(
                    samples=[dict(sample) for sample in samples if isinstance(sample, dict)],
                    total_length=total_length,
                    insertion_order=insertion_order,
                )
            )
            if len(restored_packs[-1].samples) != len(samples):
                raise ResumeStateError("[InvalidResumeState] packed assembler open pack samples must be dicts")

        if not all(isinstance(value, int) for value in open_pack_remaining):
            raise ResumeStateError("[InvalidResumeState] packed assembler open_pack_remaining must contain ints")
        if self.selection_strategy == "best_fit" and len(open_pack_remaining) != len(restored_packs):
            raise ResumeStateError("[InvalidResumeState] packed assembler open_pack_remaining length mismatch")

        try:
            self.rng.setstate(rng_state)
        except (TypeError, ValueError) as exc:
            raise ResumeStateError("[InvalidResumeState] packed assembler rng_state is malformed") from exc

        self.pending_samples = restored_pending
        self.open_packs = restored_packs
        self.open_pack_remaining = list(open_pack_remaining)
        self.next_open_pack_order = next_open_pack_order

    def fingerprint(self) -> str:
        """Return a stable fingerprint for resume compatibility checks."""
        return stable_fingerprint(
            {
                "kind": "panguvl-packed-sample-assembler",
                "version": 1,
                "class": self.__class__.__name__,
                "max_length": self.max_length,
                "selection_strategy": self.selection_strategy,
                "open_pack_limit": self.open_pack_limit,
                "pack_buffer_size": self.pack_buffer_size,
            }
        )

    def _assign_pending_to_packs(self) -> list[dict[str, Any]]:
        if not self.pending_samples:
            return []

        max_length = self.max_length
        open_pack_limit = self.open_pack_limit
        is_random = self.selection_strategy == "random"
        rng = self.rng
        open_packs = self.open_packs
        emitted: list[dict[str, Any]] = []
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

    def _flush_ready_packs(self) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
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

    def _close_most_filled_pack(self) -> dict[str, Any]:
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
        if not self.pending_samples:
            raise RuntimeError("Cannot open a pack from an empty sample pool.")

        sample_index = self.rng.randrange(len(self.pending_samples)) if self.selection_strategy == "random" else 0
        pending = self.pending_samples.pop(sample_index)
        self._add_open_pack(_OpenPack(samples=[pending.sample], total_length=pending.length))

    def _drain_pool_to_buffer_limit(self, *, buffer_limit: int | None = None) -> list[dict[str, Any]]:
        if buffer_limit is None:
            buffer_limit = self.pack_buffer_size

        emitted: list[dict[str, Any]] = []
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
        pack.insertion_order = self.next_open_pack_order
        self.next_open_pack_order += 1
        self.open_packs.append(pack)
        if self.selection_strategy == "best_fit":
            self.open_pack_remaining.append(self.max_length - pack.total_length)
            self._restore_best_fit_order(len(self.open_packs) - 1)

    def _pop_open_pack(self, index: int) -> _OpenPack:
        pack = self.open_packs.pop(index)
        if self.selection_strategy == "best_fit":
            del self.open_pack_remaining[index]
        return pack

    def _restore_best_fit_order(self, index: int) -> None:
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

    def _pack_samples(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if len(samples) == 1:
            sample = dict(samples[0])
            sample["pack_segment_ids"] = torch.ones_like(sample["input_ids"], dtype=torch.long)
            return sample

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

        image_grid_thw = [sample["image_grid_thw"] for sample in samples if sample.get("image_grid_thw") is not None]
        if image_grid_thw:
            packed_sample["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

        return packed_sample

    def _get_sample_length(self, sample: dict[str, Any]) -> int:
        """Return the token length of one processed sample."""
        return int(sample["input_ids"].size(0))


class PackedLengthAssembler(PackedSampleAssembler):
    """Assemble token lengths with the same packing decisions as sample packing."""

    def _get_sample_length(self, sample: dict[str, Any]) -> int:
        """Return the precomputed token length of one lightweight sample."""
        return int(sample["length"])

    def _pack_samples(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the packed length and source-sample count."""
        return {
            "length": sum(int(sample["length"]) for sample in samples),
            "sample_count": len(samples),
        }


def build_packed_sample_assembler(
    assemble_context: RuntimeContext,
    *,
    max_length: int,
    selection_strategy: str,
    open_pack_limit: int,
    pack_buffer_size: int,
) -> PackedSampleAssembler:
    """Build one packing assembler instance for a dataset iterator."""
    return PackedSampleAssembler(
        max_length=max_length,
        selection_strategy=selection_strategy,
        open_pack_limit=open_pack_limit,
        pack_buffer_size=pack_buffer_size,
        seed=assemble_context.sample_shuffle_seed,
    )
