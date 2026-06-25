"""General dense-sequence helpers for context parallelism."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F

from mvp_engine.distributed import ParallelMesh
from mvp_engine.distributed.cp import SeqAllToAll4D


@dataclass(frozen=True)
class CPSequenceSpec:
    """Describe how one dense batch tensor is padded and sliced for CP."""

    key: str
    dim: int
    pad_value: int | float | bool = 0
    pad_scale: int = 1


class CPKit:
    """Prepare dense token batches for context-parallel model execution."""

    def __init__(self, parallel_mesh: ParallelMesh) -> None:
        """Bind the context-parallel role used for rank-local slicing."""
        self.parallel_mesh = parallel_mesh
        self.context_size = parallel_mesh.cp.world_size
        self.context_group = parallel_mesh.cp.group
        self.context_rank = parallel_mesh.cp.rank

    def local_sequence_indices(self, global_seq_len: int, *, device: torch.device | None = None) -> torch.Tensor:
        """Return the global sequence positions owned by this context rank."""
        if global_seq_len < 0:
            raise ValueError(f"global_seq_len must be non-negative, got {global_seq_len}.")

        positions = torch.arange(global_seq_len, device=device)
        if self.context_size <= 1:
            return positions
        if global_seq_len % self.context_size != 0:
            raise ValueError(
                f"Global sequence length ({global_seq_len}) must be divisible by context size ({self.context_size})."
            )
        return positions.chunk(self.context_size, dim=0)[self.context_rank]

    def pad_sequence_batch(
        self,
        batch: Mapping[str, Any],
        specs: Iterable[CPSequenceSpec],
        *,
        target_multiple: int | None = None,
    ) -> dict[str, Any]:
        """Pad selected dense token fields to a sequence-length multiple."""
        result = dict(batch)
        multiple = self.context_size if target_multiple is None else int(target_multiple)
        if multiple <= 1:
            return result

        for spec in specs:
            value = result.get(spec.key)
            if not isinstance(value, torch.Tensor):
                continue
            if not -value.ndim <= spec.dim < value.ndim:
                raise ValueError(f"Invalid sequence dim {spec.dim} for {spec.key} with shape {tuple(value.shape)}.")

            sequence_dim = spec.dim % value.ndim
            pad_scale = int(spec.pad_scale)
            if pad_scale <= 0:
                raise ValueError(f"pad_scale must be positive for {spec.key}, got {spec.pad_scale}.")

            seq_len = int(value.shape[sequence_dim])
            if seq_len % pad_scale != 0:
                raise ValueError(
                    f"{spec.key} sequence length ({seq_len}) must be divisible by pad_scale ({pad_scale})."
                )

            pad_len = (-seq_len) % (multiple * pad_scale)
            if pad_len == 0:
                continue

            pad = [0, 0] * value.ndim
            pad[2 * (value.ndim - sequence_dim - 1) + 1] = pad_len
            result[spec.key] = F.pad(value, tuple(pad), value=spec.pad_value)

        return result

    def slice_sequence_batch(
        self,
        batch: Mapping[str, Any],
        specs: Iterable[CPSequenceSpec],
    ) -> dict[str, Any]:
        """Slice selected dense token fields and refresh local token-count metadata."""
        result = dict(batch)
        if self.context_size <= 1:
            return result

        attention_mask = result.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim > 2:
            raise ValueError(
                "CPKit.slice_sequence_batch does not support prebuilt multi-dimensional attention_mask; "
                "use packed cu_seq_lens metadata or a 2D token mask."
            )

        for spec in specs:
            value = result.get(spec.key)
            if not isinstance(value, torch.Tensor):
                continue
            if not -value.ndim <= spec.dim < value.ndim:
                raise ValueError(f"Invalid sequence dim {spec.dim} for {spec.key} with shape {tuple(value.shape)}.")

            sequence_dim = spec.dim % value.ndim
            pad_scale = int(spec.pad_scale)
            if pad_scale <= 0:
                raise ValueError(f"pad_scale must be positive for {spec.key}, got {spec.pad_scale}.")

            seq_len = int(value.shape[sequence_dim])
            if seq_len % (self.context_size * pad_scale) != 0:
                raise ValueError(
                    f"{spec.key} sequence length ({seq_len}) must be divisible by "
                    f"context_size * pad_scale ({self.context_size * pad_scale})."
                )

            indices = self.local_sequence_indices(seq_len, device=value.device)
            result[spec.key] = value.index_select(sequence_dim, indices)

        shift_labels = result.get("shift_labels")
        if isinstance(shift_labels, torch.Tensor):
            result["effective_tokens"] = int(shift_labels.ne(-100).sum().item())

        pack_segment_ids = result.get("pack_segment_ids")
        if isinstance(pack_segment_ids, torch.Tensor):
            result["total_tokens"] = int(pack_segment_ids.ne(0).sum().item())

        return result

    def gather_sequence(self, value: torch.Tensor, *, sequence_dim: int = 0) -> torch.Tensor:
        """Gather equal-sized context-local sequence chunks into global sequence order."""
        if self.context_size <= 1:
            return value
        if value.shape[sequence_dim] <= 0:
            raise ValueError("Cannot gather an empty context-local sequence.")

        local = value.movedim(sequence_dim, 0).contiguous()
        gathered = dist_nn.all_gather(local, group=self.context_group)
        return torch.cat(tuple(gathered), dim=0).movedim(0, sequence_dim).contiguous()

    def gather_seq_scatter_hidden(
        self,
        value: torch.Tensor,
        *,
        sequence_dim: int,
        hidden_dim: int = -1,
    ) -> torch.Tensor:
        """Convert local-sequence/full-hidden tensors to full-sequence/hidden-sharded layout."""
        if self.context_size <= 1:
            return value

        sequence_dim = sequence_dim % value.ndim
        hidden_dim = hidden_dim % value.ndim
        if sequence_dim == hidden_dim:
            raise ValueError("sequence_dim and hidden_dim must refer to different dimensions.")
        if value.shape[hidden_dim] % self.context_size != 0:
            raise ValueError(
                f"Hidden dimension ({value.shape[hidden_dim]}) must be divisible by context size ({self.context_size})."
            )

        local = value.movedim((sequence_dim, hidden_dim), (-2, -1)).contiguous()
        batch_shape = local.shape[:-2]
        local_seq_len = local.shape[-2]
        hidden_size = local.shape[-1]
        local = local.reshape(-1, local_seq_len, self.context_size, hidden_size // self.context_size)
        gathered = SeqAllToAll4D.apply(self.context_group, local, 2, 1, False)
        global_seq_len = gathered.shape[1]
        gathered = gathered.reshape(*batch_shape, global_seq_len, hidden_size // self.context_size)
        return gathered.movedim((-2, -1), (sequence_dim, hidden_dim)).contiguous()

    def scatter_seq_gather_hidden(
        self,
        value: torch.Tensor,
        *,
        sequence_dim: int,
        hidden_dim: int = -1,
    ) -> torch.Tensor:
        """Convert full-sequence/hidden-sharded tensors to local-sequence/full-hidden layout."""
        if self.context_size <= 1:
            return value

        sequence_dim = sequence_dim % value.ndim
        hidden_dim = hidden_dim % value.ndim
        if sequence_dim == hidden_dim:
            raise ValueError("sequence_dim and hidden_dim must refer to different dimensions.")
        if value.shape[sequence_dim] % self.context_size != 0:
            raise ValueError(
                f"Sequence dimension ({value.shape[sequence_dim]}) must be divisible by context size "
                f"({self.context_size})."
            )

        global_value = value.movedim((sequence_dim, hidden_dim), (-2, -1)).contiguous()
        batch_shape = global_value.shape[:-2]
        global_seq_len = global_value.shape[-2]
        hidden_size = global_value.shape[-1]
        global_value = global_value.reshape(-1, global_seq_len, 1, hidden_size)
        scattered = SeqAllToAll4D.apply(self.context_group, global_value, 1, 2, False)
        local_seq_len = scattered.shape[1]
        scattered = scattered.reshape(*batch_shape, local_seq_len, hidden_size * self.context_size)
        return scattered.movedim((-2, -1), (sequence_dim, hidden_dim)).contiguous()
