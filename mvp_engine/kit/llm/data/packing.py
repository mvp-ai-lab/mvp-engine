"""Sequential token-stream packing utilities for text-only language model data."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from mvp_dataset.core import Assembler
from mvp_dataset.core.resume import stable_fingerprint

from .sample import LLMPack, LLMSample
from .spec import LLMPackingSpec


class LLMPackingAssembler(Assembler[LLMSample, LLMPack]):
    """Concatenate samples in order and split the token stream into fixed-size packs."""

    def __init__(
        self,
        spec: LLMPackingSpec,
        assemble_context: Any | None = None,
    ) -> None:
        """Configure the sequential token-stream packer."""
        del assemble_context
        self.spec = spec
        self.pending_input_ids: list[int] = []
        self.pending_labels: list[int] = []
        self.pending_sample_ids: list[int] = []
        self.next_sample_id = 1
        self.pad_token_id: int | None = None
        self.ignore_index: int = -100

    def push(self, sample: LLMSample) -> Iterable[LLMPack]:
        """Append one tokenized sample to the stream and emit full packs."""
        if sample.token_length <= 0:
            return []

        tokenizer = sample.tokenization_handler.tokenizer
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            self.pad_token_id = int(pad_token_id)
        self.ignore_index = int(sample.tokenization_handler.ignore_index)

        sample_id = self.next_sample_id
        self.next_sample_id += 1
        self.pending_input_ids.extend(sample.input_ids)
        self.pending_labels.extend(sample.labels)
        self.pending_sample_ids.extend([sample_id] * sample.token_length)
        return self._drain_full_packs()

    def finish(self, *, drop_last: bool = False) -> Iterable[LLMPack]:
        """Flush the final stream tail according to the configured tail policy."""
        emitted = self._drain_full_packs()
        if self.pending_input_ids and self.spec.tail_policy == "pad" and not drop_last:
            emitted.append(self._build_pack(len(self.pending_input_ids), pad_to_length=self.spec.max_seq_len))

        self.pending_input_ids.clear()
        self.pending_labels.clear()
        self.pending_sample_ids.clear()
        return emitted

    def state_dict(self) -> dict[str, object]:
        """Return resumable packing state."""
        return {
            "pending_input_ids": list(self.pending_input_ids),
            "pending_labels": list(self.pending_labels),
            "pending_sample_ids": list(self.pending_sample_ids),
            "next_sample_id": self.next_sample_id,
            "pad_token_id": self.pad_token_id,
            "ignore_index": self.ignore_index,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore resumable packing state."""
        self.pending_input_ids = list(state.get("pending_input_ids", []))
        self.pending_labels = list(state.get("pending_labels", []))
        self.pending_sample_ids = list(state.get("pending_sample_ids", []))
        self.next_sample_id = int(state.get("next_sample_id", 1))
        pad_token_id = state.get("pad_token_id")
        self.pad_token_id = int(pad_token_id) if pad_token_id is not None else None
        self.ignore_index = int(state.get("ignore_index", -100))

    def fingerprint(self) -> str:
        """Return a stable resume fingerprint for this assembler configuration."""
        return stable_fingerprint(
            {
                "max_seq_len": self.spec.max_seq_len,
                "tail_policy": self.spec.tail_policy,
                "isolate_attention": self.spec.isolate_attention,
                "isolate_position_ids": self.spec.isolate_position_ids,
            }
        )

    def _drain_full_packs(self) -> list[LLMPack]:
        """Emit all currently available full-length stream packs."""
        emitted = []
        while len(self.pending_input_ids) >= self.spec.max_seq_len:
            emitted.append(self._build_pack(self.spec.max_seq_len))
        return emitted

    def _build_pack(self, length: int, *, pad_to_length: int | None = None) -> LLMPack:
        """Build one pack from the stream head and remove consumed tokens."""
        input_ids = self.pending_input_ids[:length]
        labels = self.pending_labels[:length]
        sample_ids = self.pending_sample_ids[:length]
        del self.pending_input_ids[:length]
        del self.pending_labels[:length]
        del self.pending_sample_ids[:length]

        pad_length = 0 if pad_to_length is None else pad_to_length - length
        if pad_length < 0:
            raise ValueError("Cannot pad a pack to a length shorter than its token count.")
        if pad_length:
            if self.pad_token_id is None:
                raise ValueError("Tokenizer must expose pad_token_id when LLMPackingSpec.tail_policy='pad'.")
            input_ids = input_ids + [self.pad_token_id] * pad_length
            labels = labels + [self.ignore_index] * pad_length

        attention_mask = self._build_attention_mask(sample_ids, pad_length)
        pack_segment_ids = self._build_pack_segment_ids(sample_ids, pad_length)
        position_ids = self._build_position_ids(sample_ids, pad_length) if self.spec.isolate_position_ids else None
        return LLMPack(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            pack_segment_ids=pack_segment_ids,
            position_ids=position_ids,
            source_sample_num=len(set(sample_ids)),
        )

    def _build_attention_mask(self, sample_ids: list[int], pad_length: int) -> list[int]:
        """Build the 1D token-validity mask used for token counting."""
        return [1] * len(sample_ids) + [0] * pad_length

    def _build_pack_segment_ids(self, sample_ids: list[int], pad_length: int) -> list[int]:
        """Build segment ids consumed by packed attention-mask preparation."""
        if not self.spec.isolate_attention:
            return [1] * len(sample_ids) + [0] * pad_length

        segment_ids = []
        current_sample_id = None
        current_segment_id = 0
        for sample_id in sample_ids:
            if sample_id != current_sample_id:
                current_sample_id = sample_id
                current_segment_id += 1
            segment_ids.append(current_segment_id)
        return segment_ids + [0] * pad_length

    def _build_position_ids(self, sample_ids: list[int], pad_length: int) -> list[int]:
        """Build optional isolated position ids for visible sample boundaries."""
        position_ids = []
        current_sample_id = None
        current_position = 0
        for sample_id in sample_ids:
            if sample_id != current_sample_id:
                current_sample_id = sample_id
                current_position = 0
            position_ids.append(current_position)
            current_position += 1
        return position_ids + [0] * pad_length


def build_packed_block_causal_mask(
    pack_segment_ids: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a 4D additive mask from stream segment ids for eager/SDPA backends."""
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
