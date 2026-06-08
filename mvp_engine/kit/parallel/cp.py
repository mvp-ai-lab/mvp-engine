"""Context-parallel batch layout helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from mvp_engine.distributed.cp import get_local_sequence_position_indices
from mvp_engine.distributed.utils import (
    get_context_parallel_group,
    get_context_parallel_size,
)

_BASIC_RING_IMPL_TYPES = {"basic", "basic_pytorch", "basic_flashinfer", "basic_npu"}


@dataclass(frozen=True, slots=True)
class CPBatchLayout:
    """Context-parallel sequence layout metadata for one local rank."""

    global_seq_len: int
    unpadded_global_seq_len: int
    local_seq_len: int
    context_size: int
    ring_impl_type: str
    local_position_indices: torch.Tensor
    local_loss_tokens: int


@dataclass(frozen=True, slots=True)
class CPCausalBatch:
    """Prepared context-local causal batch plus its padded global view."""

    local_batch: dict[str, Any]
    global_batch: dict[str, Any]
    layout: CPBatchLayout


@dataclass(frozen=True, slots=True)
class CPCrossEntropyLoss:
    """Context-parallel cross-entropy loss and detached logging statistics."""

    loss: torch.Tensor
    local_loss_sum: torch.Tensor
    global_loss_sum: torch.Tensor
    local_valid_tokens: torch.Tensor
    global_valid_tokens: torch.Tensor


class CPKit:
    """Group reusable context-parallel batch and loss utilities."""

    def prepare_causal_batch(
        self,
        batch: Mapping[str, Any],
        *,
        device_mesh: DeviceMesh,
        config: Mapping[str, Any],
        pad_token_id: int,
        ignore_index: int = -100,
        input_ids_key: str = "input_ids",
        label_key: str = "labels",
        attention_mask_key: str = "attention_mask",
        segment_ids_key: str = "pack_segment_ids",
        position_ids_key: str = "position_ids",
        token_aligned_keys: Iterable[str] = ("input_ids", "attention_mask", "pack_segment_ids", "mm_token_type_ids"),
        drop_attention_mask: bool = True,
        require_unpadded: bool = True,
    ) -> CPCausalBatch:
        """Pad, globally shift labels, and extract this rank's context-local causal batch."""
        return prepare_cp_causal_batch(
            batch,
            device_mesh=device_mesh,
            config=config,
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
            input_ids_key=input_ids_key,
            label_key=label_key,
            attention_mask_key=attention_mask_key,
            segment_ids_key=segment_ids_key,
            position_ids_key=position_ids_key,
            token_aligned_keys=token_aligned_keys,
            drop_attention_mask=drop_attention_mask,
            require_unpadded=require_unpadded,
        )

    def compute_cross_entropy_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        device_mesh: DeviceMesh,
        ignore_index: int = -100,
    ) -> CPCrossEntropyLoss:
        """Compute local-token backward loss with context-reduced token counts for CP."""
        return compute_cp_cross_entropy_loss(
            logits,
            labels,
            device_mesh=device_mesh,
            ignore_index=ignore_index,
        )


def prepare_cp_causal_batch(
    batch: Mapping[str, Any],
    *,
    device_mesh: DeviceMesh,
    config: Mapping[str, Any],
    pad_token_id: int,
    ignore_index: int = -100,
    input_ids_key: str = "input_ids",
    label_key: str = "labels",
    attention_mask_key: str = "attention_mask",
    segment_ids_key: str = "pack_segment_ids",
    position_ids_key: str = "position_ids",
    token_aligned_keys: Iterable[str] = ("input_ids", "attention_mask", "pack_segment_ids", "mm_token_type_ids"),
    drop_attention_mask: bool = True,
    require_unpadded: bool = True,
) -> CPCausalBatch:
    """Pad, globally shift labels, and extract this rank's context-local causal batch."""
    if input_ids_key not in batch:
        raise KeyError(f"CP batch requires {input_ids_key!r}.")
    if label_key not in batch:
        raise KeyError(f"CP batch requires {label_key!r}.")

    input_ids = batch[input_ids_key]
    labels = batch[label_key]
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError(
            f"Expected 2D tensor {input_ids_key}, got {type(input_ids)} {getattr(input_ids, 'shape', None)}."
        )
    if not isinstance(labels, torch.Tensor) or labels.shape != input_ids.shape:
        raise ValueError(f"{label_key} must be a tensor with the same shape as {input_ids_key}.")

    if require_unpadded and attention_mask_key in batch and batch[attention_mask_key] is not None:
        attention_mask = batch[attention_mask_key]
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
            raise ValueError(f"{attention_mask_key} must match {input_ids_key} when present.")
        if not torch.all(attention_mask != 0):
            raise ValueError("Context-parallel causal batches currently require unpadded attention masks.")

    context_size = get_context_parallel_size(device_mesh)
    if context_size <= 1:
        local_batch = dict(batch)
        shifted_labels = _build_global_next_token_labels(
            labels,
            segment_ids=batch.get(segment_ids_key),
            ignore_index=ignore_index,
        )
        positions = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long)
        local_batch[label_key] = shifted_labels
        local_batch[position_ids_key] = positions.unsqueeze(0).expand(input_ids.shape[0], -1)
        if drop_attention_mask:
            local_batch.pop(attention_mask_key, None)
        layout = CPBatchLayout(
            global_seq_len=int(input_ids.shape[1]),
            unpadded_global_seq_len=int(input_ids.shape[1]),
            local_seq_len=int(input_ids.shape[1]),
            context_size=1,
            ring_impl_type=str(config.get("ring_impl_type", "basic")),
            local_position_indices=positions,
            local_loss_tokens=int(shifted_labels.ne(ignore_index).sum().item()),
        )
        return CPCausalBatch(local_batch=local_batch, global_batch=dict(batch), layout=layout)

    global_batch = _pad_global_batch(
        batch,
        global_seq_len=int(input_ids.shape[1]),
        target_multiple=_get_layout_multiple(config, context_size),
        pad_values={
            input_ids_key: int(pad_token_id),
            label_key: int(ignore_index),
            attention_mask_key: 1,
            segment_ids_key: 0,
        },
        token_aligned_keys=set(token_aligned_keys) | {input_ids_key, label_key, attention_mask_key, segment_ids_key},
    )

    padded_input_ids = global_batch[input_ids_key]
    global_seq_len = int(padded_input_ids.shape[1])
    local_position_indices = get_local_sequence_position_indices(
        global_seq_len,
        device_mesh,
        config,
        device=padded_input_ids.device,
    )

    shifted_labels = _build_global_next_token_labels(
        global_batch[label_key],
        segment_ids=global_batch.get(segment_ids_key),
        ignore_index=ignore_index,
    )

    local_batch = dict(global_batch)
    for key in token_aligned_keys:
        value = global_batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and int(value.shape[1]) == global_seq_len:
            local_batch[key] = _index_local_sequence(value, local_position_indices)

    local_batch[label_key] = _index_local_sequence(shifted_labels, local_position_indices)
    local_batch[position_ids_key] = local_position_indices.unsqueeze(0).expand(padded_input_ids.shape[0], -1)
    if drop_attention_mask:
        local_batch.pop(attention_mask_key, None)

    layout = CPBatchLayout(
        global_seq_len=global_seq_len,
        unpadded_global_seq_len=int(input_ids.shape[1]),
        local_seq_len=int(local_batch[input_ids_key].shape[1]),
        context_size=context_size,
        ring_impl_type=str(config.get("ring_impl_type", "basic")),
        local_position_indices=local_position_indices,
        local_loss_tokens=int(local_batch[label_key].ne(ignore_index).sum().item()),
    )
    return CPCausalBatch(local_batch=local_batch, global_batch=global_batch, layout=layout)


def compute_cp_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    device_mesh: DeviceMesh,
    ignore_index: int = -100,
) -> CPCrossEntropyLoss:
    """Compute local-token backward loss with context-reduced token counts for CP."""
    if logits.ndim != 3:
        raise ValueError(f"Expected 3D logits [batch, seq, vocab], got shape {tuple(logits.shape)}.")
    if labels.shape != logits.shape[:2]:
        raise ValueError(
            f"Labels shape {tuple(labels.shape)} must match logits batch/sequence {tuple(logits.shape[:2])}."
        )

    local_loss_sum = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels.reshape(-1).to(logits.device),
        ignore_index=ignore_index,
        reduction="sum",
    )
    local_valid_tokens = labels.ne(ignore_index).sum().to(device=local_loss_sum.device, dtype=local_loss_sum.dtype)
    global_valid_tokens = local_valid_tokens.detach().clone()
    global_loss_sum = local_loss_sum.detach().clone()

    context_size = get_context_parallel_size(device_mesh)
    if context_size > 1 and dist.is_available() and dist.is_initialized():
        context_group = get_context_parallel_group(device_mesh)
        dist.all_reduce(global_valid_tokens, op=dist.ReduceOp.SUM, group=context_group)
        dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM, group=context_group)

    if global_valid_tokens.item() <= 0:
        raise ValueError("Context-parallel batch has no supervised labels across context ranks.")

    return CPCrossEntropyLoss(
        loss=local_loss_sum / global_valid_tokens,
        local_loss_sum=local_loss_sum,
        global_loss_sum=global_loss_sum,
        local_valid_tokens=local_valid_tokens,
        global_valid_tokens=global_valid_tokens,
    )


def _get_layout_multiple(config: Mapping[str, Any], context_size: int) -> int:
    ring_impl_type = str(config.get("ring_impl_type", "basic"))
    if ring_impl_type == "zigzag":
        return 2 * int(config.get("ring_degree", 1)) * int(config.get("ulysses_degree", 1))
    if ring_impl_type in _BASIC_RING_IMPL_TYPES or ring_impl_type == "strip":
        return int(context_size)
    raise ValueError(f"Unsupported context-parallel ring_impl_type for batch layout: {ring_impl_type}.")


def _pad_global_batch(
    batch: Mapping[str, Any],
    *,
    global_seq_len: int,
    target_multiple: int,
    pad_values: Mapping[str, int],
    token_aligned_keys: set[str],
) -> dict[str, Any]:
    if target_multiple <= 0:
        raise ValueError("target_multiple must be positive.")
    pad_len = (target_multiple - global_seq_len % target_multiple) % target_multiple
    if pad_len == 0:
        return dict(batch)

    padded = dict(batch)
    for key in token_aligned_keys:
        value = padded.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and int(value.shape[1]) == global_seq_len:
            padded[key] = _pad_sequence_dim(value, pad_len=pad_len, pad_value=int(pad_values.get(key, 0)))
    return padded


def _pad_sequence_dim(value: torch.Tensor, *, pad_len: int, pad_value: int) -> torch.Tensor:
    pad_shape = list(value.shape)
    pad_shape[1] = int(pad_len)
    pad = value.new_full(pad_shape, pad_value)
    return torch.cat([value, pad], dim=1)


def _build_global_next_token_labels(
    labels: torch.Tensor,
    *,
    segment_ids: torch.Tensor | None,
    ignore_index: int,
) -> torch.Tensor:
    shifted_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    if segment_ids is None:
        return shifted_labels

    shifted_segment_ids = F.pad(segment_ids, (0, 1), value=0)[..., 1:].contiguous()
    same_segment = segment_ids.ne(0) & shifted_segment_ids.eq(segment_ids)
    return shifted_labels.masked_fill(~same_segment, ignore_index)


def _index_local_sequence(value: torch.Tensor, local_position_indices: torch.Tensor) -> torch.Tensor:
    return value.index_select(1, local_position_indices.to(device=value.device)).contiguous()
