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


@dataclass(frozen=True, slots=True)
class CPBatchLayout:
    """Context-parallel sequence layout metadata for one local rank."""

    global_seq_len: int
    unpadded_global_seq_len: int
    local_seq_len: int
    context_size: int
    local_position_indices: torch.Tensor
    local_loss_tokens: int
    split_strategy: str = "text"
    rank_order_position_indices: torch.Tensor | None = None
    media_spans: tuple["CPMediaSpan", ...] = ()


@dataclass(frozen=True, slots=True)
class CPMediaSpan:
    """One packed visual media span used by multimodal context splitting."""

    media_id: int
    start: int
    end: int
    seq_len: int
    grid_thw: tuple[int, int, int] | None = None
    num_frames: int | None = None
    media_type: str = "image"


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
        split_strategy: str = "text",
        global_packed_seq_params_key: str = "global_packed_seq_params",
        image_grid_thw_key: str = "image_grid_thw",
        video_grid_thw_key: str = "video_grid_thw",
        mm_token_type_ids_key: str = "mm_token_type_ids",
        temporal_patch_size: int | None = None,
    ) -> CPCausalBatch:
        """Pad, globally shift labels, and extract this rank's context-local causal batch."""
        return prepare_cp_causal_batch(
            batch,
            device_mesh=device_mesh,
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
            split_strategy=split_strategy,
            global_packed_seq_params_key=global_packed_seq_params_key,
            image_grid_thw_key=image_grid_thw_key,
            video_grid_thw_key=video_grid_thw_key,
            mm_token_type_ids_key=mm_token_type_ids_key,
            temporal_patch_size=temporal_patch_size,
        )

    def prepare_packed_causal_batch(
        self,
        batch: Mapping[str, Any],
        *,
        device_mesh: DeviceMesh,
        pad_token_id: int,
        ignore_index: int = -100,
        input_ids_key: str = "input_ids",
        label_key: str = "labels",
        attention_mask_key: str = "attention_mask",
        segment_ids_key: str = "pack_segment_ids",
        position_ids_key: str = "position_ids",
        token_aligned_keys: Iterable[str] = ("input_ids", "attention_mask", "pack_segment_ids", "mm_token_type_ids"),
        require_unpadded: bool = True,
        cu_seq_lens_q_key: str = "cu_seq_lens_q",
        cu_seq_lens_k_key: str = "cu_seq_lens_k",
        max_length_q_key: str = "max_length_q",
        max_length_k_key: str = "max_length_k",
        split_strategy: str = "text",
        global_packed_seq_params_key: str = "global_packed_seq_params",
        image_grid_thw_key: str = "image_grid_thw",
        video_grid_thw_key: str = "video_grid_thw",
        mm_token_type_ids_key: str = "mm_token_type_ids",
        temporal_patch_size: int | None = None,
    ) -> CPCausalBatch:
        """Prepare a packed block-causal batch for context-parallel attention."""
        return prepare_cp_packed_causal_batch(
            batch,
            device_mesh=device_mesh,
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
            input_ids_key=input_ids_key,
            label_key=label_key,
            attention_mask_key=attention_mask_key,
            segment_ids_key=segment_ids_key,
            position_ids_key=position_ids_key,
            token_aligned_keys=token_aligned_keys,
            require_unpadded=require_unpadded,
            cu_seq_lens_q_key=cu_seq_lens_q_key,
            cu_seq_lens_k_key=cu_seq_lens_k_key,
            max_length_q_key=max_length_q_key,
            max_length_k_key=max_length_k_key,
            split_strategy=split_strategy,
            global_packed_seq_params_key=global_packed_seq_params_key,
            image_grid_thw_key=image_grid_thw_key,
            video_grid_thw_key=video_grid_thw_key,
            mm_token_type_ids_key=mm_token_type_ids_key,
            temporal_patch_size=temporal_patch_size,
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
    split_strategy: str = "text",
    global_packed_seq_params_key: str = "global_packed_seq_params",
    image_grid_thw_key: str = "image_grid_thw",
    video_grid_thw_key: str = "video_grid_thw",
    mm_token_type_ids_key: str = "mm_token_type_ids",
    temporal_patch_size: int | None = None,
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

    normalized_strategy = _normalize_split_strategy(split_strategy)
    if normalized_strategy not in {"text", "multimodal"}:
        raise ValueError("split_strategy must be either 'text' or 'multimodal'.")

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
        if position_ids_key not in local_batch or not isinstance(local_batch[position_ids_key], torch.Tensor):
            local_batch[position_ids_key] = positions.unsqueeze(0).expand(input_ids.shape[0], -1)
        if drop_attention_mask:
            local_batch.pop(attention_mask_key, None)
        layout = CPBatchLayout(
            global_seq_len=int(input_ids.shape[1]),
            unpadded_global_seq_len=int(input_ids.shape[1]),
            local_seq_len=int(input_ids.shape[1]),
            context_size=1,
            local_position_indices=positions,
            local_loss_tokens=int(shifted_labels.ne(ignore_index).sum().item()),
            split_strategy=normalized_strategy,
        )
        return CPCausalBatch(local_batch=local_batch, global_batch=dict(batch), layout=layout)

    if normalized_strategy == "multimodal":
        return _prepare_cp_multimodal_causal_batch(
            batch,
            device_mesh=device_mesh,
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
            input_ids_key=input_ids_key,
            label_key=label_key,
            attention_mask_key=attention_mask_key,
            segment_ids_key=segment_ids_key,
            position_ids_key=position_ids_key,
            token_aligned_keys=token_aligned_keys,
            global_packed_seq_params_key=global_packed_seq_params_key,
            image_grid_thw_key=image_grid_thw_key,
            video_grid_thw_key=video_grid_thw_key,
            mm_token_type_ids_key=mm_token_type_ids_key,
            temporal_patch_size=temporal_patch_size,
        )

    if _has_multimodal_cp_metadata(
        batch,
        global_packed_seq_params_key=global_packed_seq_params_key,
        image_grid_thw_key=image_grid_thw_key,
        video_grid_thw_key=video_grid_thw_key,
        mm_token_type_ids_key=mm_token_type_ids_key,
    ):
        raise ValueError(
            "Context-parallel text split cannot be used with multimodal media metadata; "
            "pass split_strategy='multimodal' to keep image/video spans on valid boundaries."
        )

    global_batch = _pad_global_batch(
        batch,
        global_seq_len=int(input_ids.shape[1]),
        target_multiple=context_size,
        pad_values={
            input_ids_key: int(pad_token_id),
            label_key: int(ignore_index),
            attention_mask_key: 1,
            segment_ids_key: 0,
            position_ids_key: 0,
        },
        token_aligned_keys=set(token_aligned_keys)
        | {input_ids_key, label_key, attention_mask_key, segment_ids_key, position_ids_key},
        batch_size=int(input_ids.shape[0]),
        position_ids_key=position_ids_key,
    )

    padded_input_ids = global_batch[input_ids_key]
    global_seq_len = int(padded_input_ids.shape[1])
    local_position_indices = get_local_sequence_position_indices(
        global_seq_len,
        device_mesh,
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
    position_ids = global_batch.get(position_ids_key)
    if isinstance(position_ids, torch.Tensor):
        local_batch[position_ids_key] = _index_position_ids(
            position_ids,
            local_position_indices,
            batch_size=int(padded_input_ids.shape[0]),
            global_seq_len=global_seq_len,
            position_ids_key=position_ids_key,
        )
    else:
        local_batch[position_ids_key] = local_position_indices.unsqueeze(0).expand(padded_input_ids.shape[0], -1)
    if drop_attention_mask:
        local_batch.pop(attention_mask_key, None)

    layout = CPBatchLayout(
        global_seq_len=global_seq_len,
        unpadded_global_seq_len=int(input_ids.shape[1]),
        local_seq_len=int(local_batch[input_ids_key].shape[1]),
        context_size=context_size,
        local_position_indices=local_position_indices,
        local_loss_tokens=int(local_batch[label_key].ne(ignore_index).sum().item()),
        split_strategy="text",
    )
    return CPCausalBatch(local_batch=local_batch, global_batch=global_batch, layout=layout)


def prepare_cp_packed_causal_batch(
    batch: Mapping[str, Any],
    *,
    device_mesh: DeviceMesh,
    pad_token_id: int,
    ignore_index: int = -100,
    input_ids_key: str = "input_ids",
    label_key: str = "labels",
    attention_mask_key: str = "attention_mask",
    segment_ids_key: str = "pack_segment_ids",
    position_ids_key: str = "position_ids",
    token_aligned_keys: Iterable[str] = ("input_ids", "attention_mask", "pack_segment_ids", "mm_token_type_ids"),
    require_unpadded: bool = True,
    cu_seq_lens_q_key: str = "cu_seq_lens_q",
    cu_seq_lens_k_key: str = "cu_seq_lens_k",
    max_length_q_key: str = "max_length_q",
    max_length_k_key: str = "max_length_k",
    split_strategy: str = "text",
    global_packed_seq_params_key: str = "global_packed_seq_params",
    image_grid_thw_key: str = "image_grid_thw",
    video_grid_thw_key: str = "video_grid_thw",
    mm_token_type_ids_key: str = "mm_token_type_ids",
    temporal_patch_size: int | None = None,
) -> CPCausalBatch:
    """Prepare packed block-causal inputs for Ulysses context parallelism.

    Token tensors are context-local in ``local_batch``. Packed ``cu_seq_lens``
    metadata remains global because Ulysses gathers sequence before attention.
    """
    if segment_ids_key not in batch:
        raise KeyError(f"Packed CP batch requires {segment_ids_key!r}.")

    stale_metadata_keys = {
        cu_seq_lens_q_key,
        cu_seq_lens_k_key,
        max_length_q_key,
        max_length_k_key,
    }
    causal_batch = prepare_cp_causal_batch(
        {key: value for key, value in batch.items() if key not in stale_metadata_keys},
        device_mesh=device_mesh,
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
        input_ids_key=input_ids_key,
        label_key=label_key,
        attention_mask_key=attention_mask_key,
        segment_ids_key=segment_ids_key,
        position_ids_key=position_ids_key,
        token_aligned_keys=token_aligned_keys,
        drop_attention_mask=True,
        require_unpadded=require_unpadded,
        split_strategy=split_strategy,
        global_packed_seq_params_key=global_packed_seq_params_key,
        image_grid_thw_key=image_grid_thw_key,
        video_grid_thw_key=video_grid_thw_key,
        mm_token_type_ids_key=mm_token_type_ids_key,
        temporal_patch_size=temporal_patch_size,
    )

    global_segment_ids = causal_batch.global_batch[segment_ids_key]
    rank_order_indices = causal_batch.layout.rank_order_position_indices
    if rank_order_indices is not None:
        global_segment_ids = _index_local_sequence(global_segment_ids, rank_order_indices)
    cu_seq_lens, max_length = _build_packed_cu_seqlens(global_segment_ids, segment_ids_key=segment_ids_key)

    local_batch = dict(causal_batch.local_batch)
    local_batch[attention_mask_key] = None
    local_batch[cu_seq_lens_q_key] = cu_seq_lens
    local_batch[cu_seq_lens_k_key] = cu_seq_lens
    local_batch[max_length_q_key] = max_length
    local_batch[max_length_k_key] = max_length

    global_batch = dict(causal_batch.global_batch)
    global_batch[attention_mask_key] = None
    global_batch[cu_seq_lens_q_key] = cu_seq_lens
    global_batch[cu_seq_lens_k_key] = cu_seq_lens
    global_batch[max_length_q_key] = max_length
    global_batch[max_length_k_key] = max_length

    return CPCausalBatch(local_batch=local_batch, global_batch=global_batch, layout=causal_batch.layout)


def _prepare_cp_multimodal_causal_batch(
    batch: Mapping[str, Any],
    *,
    device_mesh: DeviceMesh,
    pad_token_id: int,
    ignore_index: int,
    input_ids_key: str,
    label_key: str,
    attention_mask_key: str,
    segment_ids_key: str,
    position_ids_key: str,
    token_aligned_keys: Iterable[str],
    global_packed_seq_params_key: str,
    image_grid_thw_key: str,
    video_grid_thw_key: str,
    mm_token_type_ids_key: str,
    temporal_patch_size: int | None,
) -> CPCausalBatch:
    input_ids = batch[input_ids_key]
    if int(input_ids.shape[0]) != 1:
        raise ValueError("Multimodal context splitting currently requires a packed batch size of 1.")

    context_size = get_context_parallel_size(device_mesh)
    unpadded_global_seq_len = int(input_ids.shape[1])
    media_spans = _build_media_spans(
        batch,
        global_seq_len=unpadded_global_seq_len,
        global_packed_seq_params_key=global_packed_seq_params_key,
        image_grid_thw_key=image_grid_thw_key,
        video_grid_thw_key=video_grid_thw_key,
        mm_token_type_ids_key=mm_token_type_ids_key,
    )
    if not media_spans:
        return prepare_cp_causal_batch(
            batch,
            device_mesh=device_mesh,
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
            input_ids_key=input_ids_key,
            label_key=label_key,
            attention_mask_key=attention_mask_key,
            segment_ids_key=segment_ids_key,
            position_ids_key=position_ids_key,
            token_aligned_keys=token_aligned_keys,
            split_strategy="text",
        )

    packed_seq_params = batch[global_packed_seq_params_key]
    if temporal_patch_size is None:
        temporal_patch_size = _get_optional_int(packed_seq_params, "temporal_patch_size")
    split_units = _expand_video_tubelet_units(media_spans, temporal_patch_size=temporal_patch_size)
    rank_ranges = _build_multimodal_rank_ranges(
        split_units,
        global_seq_len=unpadded_global_seq_len,
        context_size=context_size,
    )
    local_seq_len = max(1, max(end - start for start, end in rank_ranges))
    total_padded_seq_len = local_seq_len * context_size

    global_batch = dict(batch)
    if attention_mask_key not in global_batch or global_batch[attention_mask_key] is None:
        global_batch[attention_mask_key] = torch.ones_like(input_ids)

    global_batch = _pad_global_batch_to_length(
        global_batch,
        global_seq_len=unpadded_global_seq_len,
        target_seq_len=total_padded_seq_len,
        pad_values={
            input_ids_key: int(pad_token_id),
            label_key: int(ignore_index),
            attention_mask_key: 0,
            segment_ids_key: 0,
            position_ids_key: 0,
            mm_token_type_ids_key: 0,
        },
        token_aligned_keys=set(token_aligned_keys)
        | {input_ids_key, label_key, attention_mask_key, segment_ids_key, position_ids_key, mm_token_type_ids_key},
        batch_size=int(input_ids.shape[0]),
        position_ids_key=position_ids_key,
    )

    rank_position_indices = _build_rank_position_indices(
        rank_ranges,
        local_seq_len=local_seq_len,
        unpadded_global_seq_len=unpadded_global_seq_len,
        device=input_ids.device,
    )
    context_rank = int(device_mesh.get_local_rank("context"))
    local_position_indices = rank_position_indices[context_rank]
    rank_order_position_indices = torch.cat(rank_position_indices, dim=0)

    shifted_labels = _build_global_next_token_labels(
        global_batch[label_key],
        segment_ids=global_batch.get(segment_ids_key),
        ignore_index=ignore_index,
    )

    local_batch = dict(global_batch)
    global_seq_len = int(global_batch[input_ids_key].shape[1])
    for key in token_aligned_keys:
        value = global_batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and int(value.shape[1]) == global_seq_len:
            local_batch[key] = _index_local_sequence(value, local_position_indices)

    local_batch[label_key] = _index_local_sequence(shifted_labels, local_position_indices)
    position_ids = global_batch.get(position_ids_key)
    if isinstance(position_ids, torch.Tensor):
        local_batch[position_ids_key] = _index_position_ids(
            position_ids,
            local_position_indices,
            batch_size=int(input_ids.shape[0]),
            global_seq_len=global_seq_len,
            position_ids_key=position_ids_key,
        )
    else:
        local_batch[position_ids_key] = local_position_indices.unsqueeze(0)
    local_batch[attention_mask_key] = _index_local_sequence(global_batch[attention_mask_key], local_position_indices)

    layout = CPBatchLayout(
        global_seq_len=global_seq_len,
        unpadded_global_seq_len=unpadded_global_seq_len,
        local_seq_len=local_seq_len,
        context_size=context_size,
        local_position_indices=local_position_indices,
        local_loss_tokens=int(local_batch[label_key].ne(ignore_index).sum().item()),
        split_strategy="multimodal",
        rank_order_position_indices=rank_order_position_indices,
        media_spans=media_spans,
    )
    return CPCausalBatch(local_batch=local_batch, global_batch=global_batch, layout=layout)


def _normalize_split_strategy(split_strategy: str) -> str:
    strategy = str(split_strategy).strip().lower().replace("_", "-")
    if strategy in {"text", "text-only"}:
        return "text"
    if strategy in {"multimodal", "multi-modal"}:
        return "multimodal"
    return strategy


def _has_multimodal_cp_metadata(
    batch: Mapping[str, Any],
    *,
    global_packed_seq_params_key: str,
    image_grid_thw_key: str,
    video_grid_thw_key: str,
    mm_token_type_ids_key: str,
) -> bool:
    if batch.get(global_packed_seq_params_key) is not None:
        return True
    if _has_nonempty_metadata(batch.get(image_grid_thw_key)) or _has_nonempty_metadata(batch.get(video_grid_thw_key)):
        return True

    token_type_ids = batch.get(mm_token_type_ids_key)
    if isinstance(token_type_ids, torch.Tensor) and token_type_ids.numel() > 0:
        return bool(torch.any(token_type_ids != 0).item())
    return token_type_ids is not None


def _has_nonempty_metadata(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    try:
        return len(value) > 0
    except TypeError:
        return True


def _build_media_spans(
    batch: Mapping[str, Any],
    *,
    global_seq_len: int,
    global_packed_seq_params_key: str,
    image_grid_thw_key: str,
    video_grid_thw_key: str,
    mm_token_type_ids_key: str,
) -> tuple[CPMediaSpan, ...]:
    if global_packed_seq_params_key not in batch:
        raise KeyError(f"Multimodal CP split requires {global_packed_seq_params_key!r}.")

    packed_seq_params = batch[global_packed_seq_params_key]
    cu_seqlens = _get_required_cu_seqlens(packed_seq_params)
    media_lengths = _cu_seqlens_to_lengths(cu_seqlens)
    if not media_lengths:
        return ()

    starts, ends, token_type_values = _infer_media_positions(
        batch,
        packed_seq_params=packed_seq_params,
        media_lengths=media_lengths,
        global_seq_len=global_seq_len,
        mm_token_type_ids_key=mm_token_type_ids_key,
    )
    media_types = _infer_media_types(
        packed_seq_params,
        token_type_values=token_type_values,
        media_count=len(media_lengths),
        image_grid_thw=batch.get(image_grid_thw_key),
        video_grid_thw=batch.get(video_grid_thw_key),
    )
    media_num_frames = _infer_media_num_frames(packed_seq_params, media_types)

    image_grid_rows = _grid_rows(batch.get(image_grid_thw_key))
    video_grid_rows = _grid_rows(batch.get(video_grid_thw_key))
    media_grid_rows = _grid_rows(_get_param_value(packed_seq_params, "grid_thw", "media_grid_thw"))
    if media_grid_rows is not None and len(media_grid_rows) != len(media_lengths):
        raise ValueError("grid_thw must have one row per media span.")
    image_cursor = 0
    video_cursor = 0
    spans: list[CPMediaSpan] = []
    for media_id, (start, end, seq_len, media_type) in enumerate(zip(starts, ends, media_lengths, media_types)):
        if end - start != seq_len:
            raise ValueError("Media span length must match global_packed_seq_params.cu_seqlens_q.")
        if media_grid_rows is not None:
            grid_thw = media_grid_rows[media_id]
        elif media_type == "video":
            if video_grid_rows is None:
                raise ValueError("Multimodal CP video spans require video_grid_thw metadata.")
            if video_cursor >= len(video_grid_rows):
                raise ValueError("video_grid_thw row count must match video media spans.")
            grid_thw = video_grid_rows[video_cursor]
            video_cursor += 1
        else:
            if image_grid_rows is None:
                raise ValueError("Multimodal CP image spans require image_grid_thw metadata.")
            if image_cursor >= len(image_grid_rows):
                raise ValueError("image_grid_thw row count must match image media spans.")
            grid_thw = image_grid_rows[image_cursor]
            image_cursor += 1

        spans.append(
            CPMediaSpan(
                media_id=media_id,
                start=start,
                end=end,
                seq_len=seq_len,
                grid_thw=grid_thw,
                num_frames=media_num_frames[media_id],
                media_type=media_type,
            )
        )

    if media_grid_rows is None:
        if image_grid_rows is not None and image_cursor != len(image_grid_rows):
            raise ValueError("image_grid_thw row count must match image media spans.")
        if video_grid_rows is not None and video_cursor != len(video_grid_rows):
            raise ValueError("video_grid_thw row count must match video media spans.")

    return tuple(spans)


def _get_required_cu_seqlens(packed_seq_params: Any) -> torch.Tensor:
    cu_seqlens = _get_param_value(
        packed_seq_params,
        "cu_seqlens_q",
        "cu_seq_lens_q",
        "cu_seqlens",
        "cu_seq_lens",
    )
    if cu_seqlens is None:
        raise KeyError("Multimodal CP split requires global_packed_seq_params.cu_seqlens_q.")
    if not isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.long)
    if cu_seqlens.ndim != 1:
        raise ValueError("global_packed_seq_params.cu_seqlens_q must be a 1D tensor.")
    return cu_seqlens.detach().to(device="cpu", dtype=torch.long)


def _cu_seqlens_to_lengths(cu_seqlens: torch.Tensor) -> list[int]:
    if cu_seqlens.numel() < 2:
        raise ValueError("global_packed_seq_params.cu_seqlens_q must contain at least two entries.")
    if int(cu_seqlens[0].item()) != 0:
        raise ValueError("global_packed_seq_params.cu_seqlens_q must start with 0.")
    if bool(torch.any(cu_seqlens[1:] < cu_seqlens[:-1]).item()):
        raise ValueError("global_packed_seq_params.cu_seqlens_q must be monotonically non-decreasing.")

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    if any(int(length) <= 0 for length in lengths):
        raise ValueError("Multimodal CP media sequence lengths must be positive.")
    return [int(length) for length in lengths]


def _infer_media_positions(
    batch: Mapping[str, Any],
    *,
    packed_seq_params: Any,
    media_lengths: list[int],
    global_seq_len: int,
    mm_token_type_ids_key: str,
) -> tuple[list[int], list[int], list[int | str] | None]:
    starts = _get_optional_int_list(
        packed_seq_params,
        "media_starts",
        "media_start_indices",
        "visual_token_starts",
    )
    if starts is not None:
        if len(starts) != len(media_lengths):
            raise ValueError("media_starts must have one entry per media span.")
        ends = [start + length for start, length in zip(starts, media_lengths)]
        _validate_media_positions(starts, ends, global_seq_len=global_seq_len)
        return starts, ends, None

    regions = _media_token_regions(batch.get(mm_token_type_ids_key), global_seq_len=global_seq_len)
    if regions is not None:
        if len(regions) != len(media_lengths):
            raise ValueError("mm_token_type_ids media regions must align with cu_seqlens_q media lengths.")
        starts = [start for start, _, _ in regions]
        ends = [end for _, end, _ in regions]
        region_lengths = [end - start for start, end, _ in regions]
        if region_lengths != media_lengths:
            raise ValueError("mm_token_type_ids media region lengths must match cu_seqlens_q diffs.")
        return starts, ends, [token_type for _, _, token_type in regions]

    total_media_len = sum(media_lengths)
    if total_media_len != global_seq_len:
        raise ValueError(
            "Multimodal CP split requires mm_token_type_ids or media_starts when visual tokens do not cover "
            "the whole sequence."
        )
    starts = [0]
    for length in media_lengths[:-1]:
        starts.append(starts[-1] + length)
    ends = [start + length for start, length in zip(starts, media_lengths)]
    return starts, ends, None


def _validate_media_positions(starts: list[int], ends: list[int], *, global_seq_len: int) -> None:
    previous_end = -1
    for start, end in zip(starts, ends):
        if start < 0 or end > global_seq_len or start >= end:
            raise ValueError("Media spans must be non-empty and within the global sequence.")
        if start < previous_end:
            raise ValueError("Media spans must be sorted and non-overlapping.")
        previous_end = end


def _media_token_regions(value: Any, *, global_seq_len: int) -> list[tuple[int, int, int]] | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise ValueError("mm_token_type_ids must be a tensor when provided.")
    if value.ndim == 1:
        token_types = value
    elif value.ndim == 2 and int(value.shape[0]) == 1:
        token_types = value[0]
    else:
        raise ValueError("Multimodal CP split requires mm_token_type_ids with shape [seq] or [1, seq].")
    if int(token_types.shape[0]) != global_seq_len:
        raise ValueError("mm_token_type_ids length must match input_ids sequence length.")

    row = token_types.detach().to(device="cpu", dtype=torch.long).tolist()
    regions: list[tuple[int, int, int]] = []
    start: int | None = None
    token_type = 0
    for index, current in enumerate(row):
        current = int(current)
        if current == 0:
            if start is not None:
                regions.append((start, index, token_type))
                start = None
            continue
        if start is None:
            start = index
            token_type = current
        elif current != token_type:
            regions.append((start, index, token_type))
            start = index
            token_type = current
    if start is not None:
        regions.append((start, len(row), token_type))
    return regions


def _infer_media_types(
    packed_seq_params: Any,
    *,
    token_type_values: list[int | str] | None,
    media_count: int,
    image_grid_thw: Any,
    video_grid_thw: Any,
) -> list[str]:
    media_types = _get_optional_list(packed_seq_params, "media_types", "media_type")
    if media_types is not None:
        if len(media_types) != media_count:
            raise ValueError("media_types must have one entry per media span.")
        return [_normalize_media_type(media_type) for media_type in media_types]

    if token_type_values is not None:
        return [_normalize_media_type(token_type) for token_type in token_type_values]

    image_count = _grid_count(image_grid_thw)
    video_count = _grid_count(video_grid_thw)
    if video_count == media_count and image_count == 0:
        return ["video"] * media_count
    if image_count == media_count and video_count == 0:
        return ["image"] * media_count
    if video_count == 0:
        return ["image"] * media_count
    raise ValueError("Mixed image/video multimodal CP split requires explicit media_types metadata.")


def _infer_media_num_frames(packed_seq_params: Any, media_types: list[str]) -> list[int | None]:
    aligned_num_frames = _get_optional_int_list(packed_seq_params, "num_frames", "media_num_frames")
    if aligned_num_frames is not None:
        if len(aligned_num_frames) != len(media_types):
            raise ValueError("num_frames must have one entry per media span.")
        return [int(num_frames) for num_frames in aligned_num_frames]

    video_num_frames = _get_optional_int_list(packed_seq_params, "video_num_frames")
    if video_num_frames is None:
        return [None] * len(media_types)

    video_cursor = 0
    result: list[int | None] = []
    for media_type in media_types:
        if media_type != "video":
            result.append(None)
            continue
        if video_cursor >= len(video_num_frames):
            raise ValueError("video_num_frames must have one entry per video span.")
        result.append(int(video_num_frames[video_cursor]))
        video_cursor += 1
    if video_cursor != len(video_num_frames):
        raise ValueError("video_num_frames has entries that do not match video spans.")
    return result


def _normalize_media_type(media_type: int | str) -> str:
    if isinstance(media_type, bytes):
        media_type = media_type.decode()
    if isinstance(media_type, int):
        return "video" if media_type == 2 else "image"

    normalized = str(media_type).strip().lower()
    if normalized in {"video", "videos", "vid"}:
        return "video"
    if normalized in {"image", "images", "img"}:
        return "image"
    if normalized.isdigit():
        return "video" if int(normalized) == 2 else "image"
    raise ValueError(f"Unsupported media type {media_type!r}.")


def _expand_video_tubelet_units(
    media_spans: tuple[CPMediaSpan, ...],
    *,
    temporal_patch_size: int | None,
) -> tuple[CPMediaSpan, ...]:
    if temporal_patch_size is None or int(temporal_patch_size) <= 1:
        return media_spans
    temporal_patch_size = int(temporal_patch_size)

    split_units: list[CPMediaSpan] = []
    for span in media_spans:
        if span.media_type != "video":
            split_units.append(span)
            continue
        if span.num_frames is None:
            raise ValueError("num_frames is required for video CP split when temporal_patch_size > 1.")

        tubelet_count = (int(span.num_frames) + temporal_patch_size - 1) // temporal_patch_size
        if tubelet_count <= 0:
            raise ValueError("Video num_frames must be positive.")
        if span.seq_len % tubelet_count != 0:
            raise ValueError("Video visual sequence length must be divisible by its tubelet count.")

        tubelet_seq_len = span.seq_len // tubelet_count
        for tubelet_index in range(tubelet_count):
            start = span.start + tubelet_index * tubelet_seq_len
            split_units.append(
                CPMediaSpan(
                    media_id=span.media_id,
                    start=start,
                    end=start + tubelet_seq_len,
                    seq_len=tubelet_seq_len,
                    grid_thw=span.grid_thw,
                    num_frames=span.num_frames,
                    media_type=span.media_type,
                )
            )
    return tuple(split_units)


def _build_multimodal_rank_ranges(
    split_units: tuple[CPMediaSpan, ...],
    *,
    global_seq_len: int,
    context_size: int,
) -> list[tuple[int, int]]:
    if context_size <= 0:
        raise ValueError("context_size must be positive.")
    if not split_units:
        return [(0, 0)] * context_size

    _validate_split_units(split_units, global_seq_len=global_seq_len)
    active_ranks = min(context_size, len(split_units))
    unit_groups = _partition_media_units(split_units, active_ranks, context_size=context_size)

    rank_ranges: list[tuple[int, int]] = []
    for rank, group in enumerate(unit_groups):
        if not group:
            rank_ranges.append((0, 0))
            continue
        next_group = next((candidate for candidate in unit_groups[rank + 1 :] if candidate), None)
        start = 0 if rank == 0 else group[0].start
        end = next_group[0].start if next_group is not None else global_seq_len
        rank_ranges.append((start, end))

    return rank_ranges


def _validate_split_units(split_units: tuple[CPMediaSpan, ...], *, global_seq_len: int) -> None:
    previous_end = -1
    for unit in split_units:
        if unit.start < 0 or unit.end > global_seq_len or unit.start >= unit.end:
            raise ValueError("Multimodal CP split units must be non-empty and within the global sequence.")
        if unit.start < previous_end:
            raise ValueError("Multimodal CP split units must be sorted and non-overlapping.")
        if unit.seq_len != unit.end - unit.start:
            raise ValueError("Multimodal CP split unit seq_len must match end - start.")
        previous_end = unit.end


def _partition_media_units(
    split_units: tuple[CPMediaSpan, ...],
    active_ranks: int,
    *,
    context_size: int,
) -> list[list[CPMediaSpan]]:
    lengths = [unit.seq_len for unit in split_units]
    target = sum(lengths) / float(context_size)
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)

    count = len(split_units)
    best: list[list[tuple[float, int] | None]] = [[None] * (count + 1) for _ in range(active_ranks + 1)]
    choice = [[0] * (count + 1) for _ in range(active_ranks + 1)]
    best[0][0] = (0.0, 0)

    for groups in range(1, active_ranks + 1):
        for end in range(groups, count + 1):
            for start in range(groups - 1, end):
                previous = best[groups - 1][start]
                if previous is None:
                    continue
                group_sum = prefix[end] - prefix[start]
                score = (max(previous[0], abs(group_sum - target)), max(previous[1], group_sum))
                if best[groups][end] is None or score < best[groups][end]:
                    best[groups][end] = score
                    choice[groups][end] = start

    ranges: list[tuple[int, int]] = []
    end = count
    for groups in range(active_ranks, 0, -1):
        start = choice[groups][end]
        ranges.append((start, end))
        end = start
    ranges.reverse()

    unit_groups = [[split_units[start:end] for start, end in ranges][rank] for rank in range(active_ranks)]
    unit_groups.extend([] for _ in range(context_size - active_ranks))
    return [list(group) for group in unit_groups]


def _build_rank_position_indices(
    rank_ranges: list[tuple[int, int]],
    *,
    local_seq_len: int,
    unpadded_global_seq_len: int,
    device: torch.device,
) -> list[torch.Tensor]:
    rank_indices: list[torch.Tensor] = []
    pad_cursor = unpadded_global_seq_len
    for start, end in rank_ranges:
        real_length = end - start
        if real_length < 0 or real_length > local_seq_len:
            raise ValueError("Invalid multimodal CP rank range.")
        real_positions = torch.arange(start, end, device=device, dtype=torch.long)
        pad_length = local_seq_len - real_length
        if pad_length > 0:
            pad_positions = torch.arange(pad_cursor, pad_cursor + pad_length, device=device, dtype=torch.long)
            pad_cursor += pad_length
            real_positions = torch.cat([real_positions, pad_positions], dim=0)
        rank_indices.append(real_positions)

    expected_total = local_seq_len * len(rank_ranges)
    if pad_cursor != expected_total:
        raise ValueError("Multimodal CP rank ranges must cover the full global sequence exactly once.")
    return rank_indices


def _pad_global_batch_to_length(
    batch: Mapping[str, Any],
    *,
    global_seq_len: int,
    target_seq_len: int,
    pad_values: Mapping[str, int],
    token_aligned_keys: set[str],
    batch_size: int,
    position_ids_key: str,
) -> dict[str, Any]:
    if target_seq_len < global_seq_len:
        raise ValueError("target_seq_len must be greater than or equal to global_seq_len.")
    pad_len = target_seq_len - global_seq_len
    if pad_len == 0:
        return dict(batch)

    padded = dict(batch)
    for key in token_aligned_keys:
        value = padded.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        sequence_dim = _get_token_sequence_dim(
            key,
            value,
            batch_size=batch_size,
            global_seq_len=global_seq_len,
            position_ids_key=position_ids_key,
        )
        if sequence_dim is not None:
            padded[key] = _pad_sequence_dim(
                value,
                dim=sequence_dim,
                pad_len=pad_len,
                pad_value=int(pad_values.get(key, 0)),
            )
    return padded


def _get_optional_int(source: Any, name: str) -> int | None:
    value = _get_param_value(source, name)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar.")
        return int(value.item())
    return int(value)


def _get_param_value(source: Any, *names: str) -> Any:
    if source is None:
        return None
    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    return None


def _get_optional_list(source: Any, *names: str) -> list[Any] | None:
    value = _get_param_value(source, *names)
    if value is None:
        return None
    if isinstance(value, str | bytes):
        return [value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().flatten().tolist()
    return list(value)


def _get_optional_int_list(source: Any, *names: str) -> list[int] | None:
    values = _get_optional_list(source, *names)
    if values is None:
        return None
    return [int(value) for value in values]


def _grid_rows(value: Any) -> list[tuple[int, int, int]] | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=torch.long)
    if value.numel() == 0:
        return []
    if value.numel() % 3 != 0:
        raise ValueError("grid_thw metadata must contain rows of three values.")
    value = value.detach().to(device="cpu", dtype=torch.long).reshape(-1, 3)
    rows = [tuple(int(part) for part in row.tolist()) for row in value]
    if any(any(part <= 0 for part in row) for row in rows):
        raise ValueError("grid_thw entries must be positive.")
    return rows


def _grid_count(value: Any) -> int:
    rows = _grid_rows(value)
    return 0 if rows is None else len(rows)


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


def _pad_global_batch(
    batch: Mapping[str, Any],
    *,
    global_seq_len: int,
    target_multiple: int,
    pad_values: Mapping[str, int],
    token_aligned_keys: set[str],
    batch_size: int,
    position_ids_key: str,
) -> dict[str, Any]:
    if target_multiple <= 0:
        raise ValueError("target_multiple must be positive.")
    pad_len = (target_multiple - global_seq_len % target_multiple) % target_multiple
    if pad_len == 0:
        return dict(batch)

    padded = dict(batch)
    for key in token_aligned_keys:
        value = padded.get(key)
        if not isinstance(value, torch.Tensor):
            continue

        sequence_dim = _get_token_sequence_dim(
            key,
            value,
            batch_size=batch_size,
            global_seq_len=global_seq_len,
            position_ids_key=position_ids_key,
        )
        if sequence_dim is not None:
            padded[key] = _pad_sequence_dim(
                value,
                dim=sequence_dim,
                pad_len=pad_len,
                pad_value=int(pad_values.get(key, 0)),
            )
    return padded


def _pad_sequence_dim(value: torch.Tensor, *, dim: int, pad_len: int, pad_value: int) -> torch.Tensor:
    pad_shape = list(value.shape)
    pad_shape[dim] = int(pad_len)
    pad = value.new_full(pad_shape, pad_value)
    return torch.cat([value, pad], dim=dim)


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


def _build_packed_cu_seqlens(
    pack_segment_ids: torch.Tensor,
    *,
    segment_ids_key: str,
) -> tuple[torch.Tensor, int]:
    if not isinstance(pack_segment_ids, torch.Tensor) or pack_segment_ids.ndim != 2:
        raise ValueError(
            f"Expected 2D tensor {segment_ids_key}, got "
            f"{type(pack_segment_ids)} {getattr(pack_segment_ids, 'shape', None)}."
        )

    segment_lengths = []
    for row in pack_segment_ids:
        if row.numel() == 0:
            continue

        valid_length = int(row.ne(0).sum().item())
        if valid_length <= 0:
            raise ValueError(f"Each packed CP row in {segment_ids_key} must contain at least one token.")
        if bool(row[:valid_length].eq(0).any().item()) or bool(row[valid_length:].ne(0).any().item()):
            raise ValueError(f"{segment_ids_key} padding must be a single zero-valued suffix.")

        starts = torch.cat(
            [
                torch.zeros(1, device=row.device, dtype=torch.long),
                torch.nonzero(row[1:] != row[:-1], as_tuple=False).flatten() + 1,
            ]
        )
        ends = torch.cat(
            [
                starts[1:],
                torch.tensor([row.numel()], device=row.device, dtype=torch.long),
            ]
        )
        segment_lengths.append(ends - starts)

    if not segment_lengths:
        raise ValueError(f"{segment_ids_key} must contain at least one packed token.")

    seqlens = torch.cat(segment_lengths).to(dtype=torch.int32)
    cu_seq_lens = torch.zeros(seqlens.numel() + 1, device=pack_segment_ids.device, dtype=torch.int32)
    cu_seq_lens[1:] = torch.cumsum(seqlens, dim=0)
    if int(cu_seq_lens[-1].item()) != int(pack_segment_ids.numel()):
        raise ValueError("Packed CP cu_seq_lens must cover the full padded batch.")

    return cu_seq_lens, int(seqlens.max().item())


def _index_local_sequence(value: torch.Tensor, local_position_indices: torch.Tensor) -> torch.Tensor:
    return value.index_select(1, local_position_indices.to(device=value.device)).contiguous()


def _index_position_ids(
    value: torch.Tensor,
    local_position_indices: torch.Tensor,
    *,
    batch_size: int,
    global_seq_len: int,
    position_ids_key: str,
) -> torch.Tensor:
    sequence_dim = _get_position_ids_sequence_dim(
        value,
        batch_size=batch_size,
        global_seq_len=global_seq_len,
        position_ids_key=position_ids_key,
    )
    return value.index_select(sequence_dim, local_position_indices.to(device=value.device)).contiguous()


def _get_token_sequence_dim(
    key: str,
    value: torch.Tensor,
    *,
    batch_size: int,
    global_seq_len: int,
    position_ids_key: str,
) -> int | None:
    if key == position_ids_key:
        return _get_position_ids_sequence_dim(
            value,
            batch_size=batch_size,
            global_seq_len=global_seq_len,
            position_ids_key=position_ids_key,
        )
    if value.ndim >= 2 and int(value.shape[1]) == global_seq_len:
        return 1
    return None


def _get_position_ids_sequence_dim(
    value: torch.Tensor,
    *,
    batch_size: int,
    global_seq_len: int,
    position_ids_key: str,
) -> int:
    if value.ndim == 2 and int(value.shape[0]) == batch_size and int(value.shape[1]) == global_seq_len:
        return 1
    if value.ndim == 3 and int(value.shape[1]) == batch_size and int(value.shape[2]) == global_seq_len:
        return 2
    raise ValueError(
        f"{position_ids_key} must have shape [batch, seq] or [dims, batch, seq], got {tuple(value.shape)}."
    )
