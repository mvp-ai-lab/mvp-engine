"""Context attention helpers."""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_world_size

try:
    from torch.distributed.tensor import DTensor
except Exception:  # pragma: no cover - runtime-dependent
    DTensor = ()


class SeqAllToAll4D(torch.autograd.Function):
    """Autograd-aware all-to-all for ``[batch, sequence, heads, head_dim]`` tensors."""

    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup | None,
        value: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        use_sync: bool = False,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.use_sync = use_sync
        return all_to_all_4d(
            value,
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
            group=group,
            use_sync=use_sync,
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[None, torch.Tensor, None, None, None]:
        grad_input = SeqAllToAll4D.apply(
            ctx.group,
            grad_output,
            ctx.gather_dim,
            ctx.scatter_dim,
            ctx.use_sync,
        )
        return None, grad_input, None, None, None


def _maybe_sync(value: torch.Tensor, use_sync: bool) -> None:
    if use_sync and value.device.type == "cuda":
        torch.cuda.synchronize(value.device)
    elif use_sync and value.device.type == "npu":
        torch.npu.synchronize(value.device)


def _scatter_heads_gather_sequence(
    value: torch.Tensor,
    world_size: int,
    *,
    group: dist.ProcessGroup | None,
    use_sync: bool,
) -> torch.Tensor:
    batch, local_seq_len, heads, head_dim = value.shape
    if heads % world_size != 0:
        raise ValueError(f"Ulysses requires heads ({heads}) to be divisible by context size ({world_size}).")

    local_heads = heads // world_size
    send = value.reshape(batch, local_seq_len, world_size, local_heads, head_dim)
    send = send.permute(2, 1, 0, 3, 4).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    _maybe_sync(value, use_sync)

    global_seq_len = local_seq_len * world_size
    return recv.reshape(global_seq_len, batch, local_heads, head_dim).permute(1, 0, 2, 3).contiguous()


def _scatter_sequence_gather_heads(
    value: torch.Tensor,
    world_size: int,
    *,
    group: dist.ProcessGroup | None,
    use_sync: bool,
) -> torch.Tensor:
    batch, global_seq_len, local_heads, head_dim = value.shape
    if global_seq_len % world_size != 0:
        raise ValueError(
            f"Ulysses requires sequence length ({global_seq_len}) to be divisible by context size ({world_size})."
        )

    local_seq_len = global_seq_len // world_size
    heads = local_heads * world_size
    send = value.reshape(batch, world_size, local_seq_len, local_heads, head_dim)
    send = send.permute(1, 3, 2, 0, 4).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    _maybe_sync(value, use_sync)

    return recv.reshape(heads, local_seq_len, batch, head_dim).permute(2, 1, 0, 3).contiguous()


def all_to_all_4d(
    value: torch.Tensor,
    *,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup | None,
    use_sync: bool = False,
) -> torch.Tensor:
    """Reshard a 4D sequence/head tensor between Ulysses sequence and head layouts."""
    if value.ndim != 4:
        raise ValueError(f"Ulysses all-to-all expects a 4D tensor, got shape {tuple(value.shape)}.")

    world_size = get_world_size(group)
    if world_size == 1:
        return value.contiguous()

    if scatter_dim == 2 and gather_dim == 1:
        return _scatter_heads_gather_sequence(value, world_size, group=group, use_sync=use_sync)
    if scatter_dim == 1 and gather_dim == 2:
        return _scatter_sequence_gather_heads(value, world_size, group=group, use_sync=use_sync)
    raise ValueError("Ulysses all-to-all supports only scatter/gather dim pairs (2, 1) and (1, 2).")


_FLASH_ATTENTION_IMPL = "flash_attention_2"
_NPU_ATTENTION_IMPL = "npu_fusion_attention_v2"
_TORCH_ATTENTION_IMPLS = (
    "sdpa",
    "torch_math",
    "torch_flash",
    "torch_efficient",
    "torch_cudnn",
)
_SUPPORTED_ATTENTION_IMPLEMENTATIONS = (
    _FLASH_ATTENTION_IMPL,
    *_TORCH_ATTENTION_IMPLS,
    _NPU_ATTENTION_IMPL,
)
_SUPPORTED_ATTENTION_IMPLEMENTATIONS_MESSAGE = ", ".join(sorted(_SUPPORTED_ATTENTION_IMPLEMENTATIONS))
_NATIVE_QKV_LAYOUT = "BSHD"
_HF_QKV_LAYOUT = "BHSD"
_SUPPORTED_QKV_LAYOUTS = (_NATIVE_QKV_LAYOUT, _HF_QKV_LAYOUT)
_SUPPORTED_QKV_LAYOUTS_MESSAGE = ", ".join(_SUPPORTED_QKV_LAYOUTS)


class UlyssesSPAttention(nn.Module):
    """Ulysses attention over context-sharded sequence chunks."""

    def __init__(
        self,
        cp_mesh,
        attn_implementation: str = "flash_attention_2",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.sequence_process_group = cp_mesh.get_group()
        self.attn_implementation = str(attn_implementation)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        cu_seq_lens_q: torch.Tensor | None = None,
        cu_seq_lens_k: torch.Tensor | None = None,
        max_length_q: int | None = None,
        max_length_k: int | None = None,
        dropout_p: float = 0.0,
        scaling: float | None = None,
        is_causal: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        softcap: float = 0.0,
        alibi_slopes: torch.Tensor | None = None,
        deterministic: bool = False,
        return_attn_probs: bool = False,
    ) -> torch.Tensor:
        """Run Ulysses all-to-all, local attention, and inverse all-to-all."""
        q = SeqAllToAll4D.apply(self.sequence_process_group, query, 2, 1, False)
        k = SeqAllToAll4D.apply(self.sequence_process_group, key, 2, 1, False)
        v = SeqAllToAll4D.apply(self.sequence_process_group, value, 2, 1, False)

        if scaling is None:
            scaling = q.shape[-1] ** -0.5

        attention_mask = _prepare_attention_mask(
            attention_mask,
            batch_size=q.shape[0],
            local_seq_len=query.shape[1],
            global_seq_len=q.shape[1],
            group=self.sequence_process_group,
        )
        context = run_attention(
            q,
            k,
            v,
            attn_implementation=self.attn_implementation,
            attention_mask=attention_mask,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
            dropout_p=dropout_p,
            scaling=scaling,
            is_causal=is_causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )
        if isinstance(context, tuple):
            context = context[0]

        return SeqAllToAll4D.apply(self.sequence_process_group, context, 1, 2, False)


def run_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_implementation: str | None,
    attention_mask: torch.Tensor | None = None,
    cu_seq_lens_q: torch.Tensor | None = None,
    cu_seq_lens_k: torch.Tensor | None = None,
    max_length_q: int | None = None,
    max_length_k: int | None = None,
    dropout_p: float,
    scaling: float | None,
    is_causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    deterministic: bool,
    return_attn_probs: bool,
) -> torch.Tensor:
    """Dispatch local attention over full sequence and local head shards."""
    normalized = _FLASH_ATTENTION_IMPL if attn_implementation is None else str(attn_implementation).strip()
    if normalized not in _SUPPORTED_ATTENTION_IMPLEMENTATIONS:
        raise ValueError(
            f"Unsupported Ulysses attention implementation: {attn_implementation!r}. "
            f"Supported implementations: {_SUPPORTED_ATTENTION_IMPLEMENTATIONS_MESSAGE}."
        )
    if normalized == _FLASH_ATTENTION_IMPL and query.device.type == "npu":
        normalized = _NPU_ATTENTION_IMPL

    if normalized == _FLASH_ATTENTION_IMPL:
        return _flash_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
            dropout_p=dropout_p,
            scaling=scaling,
            is_causal=is_causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )
    if normalized == _NPU_ATTENTION_IMPL:
        return _npu_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
            dropout_p=dropout_p,
            scaling=scaling,
            is_causal=is_causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )
    return _torch_attention(
        query,
        key,
        value,
        attn_implementation=normalized,
        attention_mask=attention_mask,
        cu_seq_lens_q=cu_seq_lens_q,
        cu_seq_lens_k=cu_seq_lens_k,
        max_length_q=max_length_q,
        max_length_k=max_length_k,
        dropout_p=dropout_p,
        scaling=scaling,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        return_attn_probs=return_attn_probs,
    )


def _has_cu_seqlens(
    cu_seq_lens_q: torch.Tensor | None,
    cu_seq_lens_k: torch.Tensor | None,
    max_length_q: int | None,
    max_length_k: int | None,
) -> bool:
    values = (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
    if all(value is None for value in values):
        return False
    if any(value is None for value in values):
        raise ValueError("cu_seq_lens_q, cu_seq_lens_k, max_length_q, and max_length_k must be provided together.")
    return True


def _prepare_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    local_seq_len: int,
    global_seq_len: int,
    group: dist.ProcessGroup | None,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim != 2:
        raise ValueError(
            f"Ulysses SP attention only supports 2D 0/1 attention_mask, got shape {tuple(attention_mask.shape)}."
        )
    if attention_mask.shape[0] != batch_size:
        raise ValueError(
            f"attention_mask batch size ({attention_mask.shape[0]}) must match query batch size ({batch_size})."
        )

    mask = attention_mask
    if mask.shape[1] == local_seq_len and local_seq_len != global_seq_len:
        gathered = [torch.empty_like(mask) for _ in range(get_world_size(group))]
        dist.all_gather(gathered, mask.contiguous(), group=group)
        mask = torch.cat(gathered, dim=1)
    elif mask.shape[1] != global_seq_len:
        raise ValueError(
            "attention_mask sequence length must be either local or global sequence length, "
            f"got {mask.shape[1]}, expected {local_seq_len} or {global_seq_len}."
        )

    return _normalize_2d_attention_mask(mask)


def _validate_cu_seqlens(
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    *,
    total_q: int,
    total_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cu_seq_lens_q = cu_seq_lens_q.to(device=device, dtype=torch.int32)
    cu_seq_lens_k = cu_seq_lens_k.to(device=device, dtype=torch.int32)
    if cu_seq_lens_q.ndim != 1 or cu_seq_lens_k.ndim != 1:
        raise ValueError("cu_seq_lens_q and cu_seq_lens_k must be 1D tensors.")
    if cu_seq_lens_q.numel() < 2 or cu_seq_lens_k.numel() < 2:
        raise ValueError("cu_seq_lens_q and cu_seq_lens_k must contain at least two entries.")
    if int(cu_seq_lens_q[0].item()) != 0 or int(cu_seq_lens_k[0].item()) != 0:
        raise ValueError("cu_seq_lens_q and cu_seq_lens_k must start with 0.")
    if int(cu_seq_lens_q[-1].item()) != total_q or int(cu_seq_lens_k[-1].item()) != total_k:
        raise ValueError(
            "CP cu_seq_lens must cover the full global flattened sequence: "
            f"got q={int(cu_seq_lens_q[-1].item())}, k={int(cu_seq_lens_k[-1].item())}, "
            f"expected q={total_q}, k={total_k}."
        )
    if torch.any(cu_seq_lens_q[1:] < cu_seq_lens_q[:-1]) or torch.any(cu_seq_lens_k[1:] < cu_seq_lens_k[:-1]):
        raise ValueError("cu_seq_lens_q and cu_seq_lens_k must be monotonically non-decreasing.")
    return cu_seq_lens_q, cu_seq_lens_k


def _validate_right_padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1, dtype=torch.int32)
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    expected = positions.unsqueeze(0) < lengths.to(device=attention_mask.device).unsqueeze(1)
    if not torch.equal(attention_mask, expected):
        raise ValueError("flash_attention_2 only supports right-padding 0/1 attention_mask.")
    return lengths


def _normalize_2d_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be a 2D 0/1 mask, got shape {tuple(attention_mask.shape)}.")
    if attention_mask.numel() > 0 and (torch.any(attention_mask < 0) or torch.any(attention_mask > 1)):
        raise ValueError("attention_mask must be a 0/1 mask.")
    return attention_mask.to(dtype=torch.bool)


def _flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    cu_seq_lens_q: torch.Tensor | None,
    cu_seq_lens_k: torch.Tensor | None,
    max_length_q: int | None,
    max_length_k: int | None,
    dropout_p: float,
    scaling: float | None,
    is_causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    deterministic: bool,
    return_attn_probs: bool,
) -> torch.Tensor:
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
    except ImportError as exc:  # pragma: no cover - optional CUDA dependency
        raise ImportError("Ulysses attn_implementation='flash_attention_2' requires flash-attn.") from exc

    total_q = query.shape[0] * query.shape[1]
    total_k = key.shape[0] * key.shape[1]
    has_cu_seqlens = _has_cu_seqlens(cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
    if has_cu_seqlens and attention_mask is not None:
        raise ValueError("Pass either attention_mask or cu_seq_lens, not both.")

    if has_cu_seqlens:
        if return_attn_probs:
            raise ValueError("flash_attention_2 cu_seq_lens path does not support return_attn_probs.")
        cu_seq_lens_q, cu_seq_lens_k = _validate_cu_seqlens(
            cu_seq_lens_q,
            cu_seq_lens_k,
            total_q=total_q,
            total_k=total_k,
            device=query.device,
        )
        return flash_attn_varlen_func(
            query.reshape(total_q, query.shape[2], query.shape[3]),
            key.reshape(total_k, key.shape[2], key.shape[3]),
            value.reshape(total_k, value.shape[2], value.shape[3]),
            cu_seq_lens_q,
            cu_seq_lens_k,
            int(max_length_q),
            int(max_length_k),
            dropout_p=dropout_p,
            softmax_scale=scaling,
            causal=is_causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        ).reshape_as(query)

    if attention_mask is not None:
        if return_attn_probs:
            raise ValueError("flash_attention_2 attention_mask path does not support return_attn_probs.")

        attention_mask = _normalize_2d_attention_mask(attention_mask)
        lengths = _validate_right_padding_mask(attention_mask)
        if torch.any(lengths == 0):
            raise ValueError("attention_mask rows must contain at least one valid token.")

        flat_mask = attention_mask.reshape(-1)
        indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
        cu_seqlens = torch.zeros(lengths.numel() + 1, device=lengths.device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
        max_seqlen = int(lengths.max().item())
        output = flash_attn_varlen_func(
            query.reshape(total_q, query.shape[2], query.shape[3]).index_select(0, indices),
            key.reshape(total_k, key.shape[2], key.shape[3]).index_select(0, indices),
            value.reshape(total_k, value.shape[2], value.shape[3]).index_select(0, indices),
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=scaling,
            causal=is_causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=False,
        )
        padded_output = torch.zeros_like(query.reshape(total_q, query.shape[2], query.shape[3]))
        padded_output.index_copy_(0, indices, output)
        return padded_output.reshape_as(query)

    return flash_attn_func(
        query,
        key,
        value,
        dropout_p=dropout_p,
        softmax_scale=scaling,
        causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )


def _npu_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    cu_seq_lens_q: torch.Tensor | None,
    cu_seq_lens_k: torch.Tensor | None,
    max_length_q: int | None,
    max_length_k: int | None,
    dropout_p: float,
    scaling: float | None,
    is_causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    deterministic: bool,
    return_attn_probs: bool,
) -> torch.Tensor:
    has_cu_seqlens = _has_cu_seqlens(cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
    if attention_mask is not None and has_cu_seqlens:
        raise ValueError("Pass either attention_mask or cu_seq_lens, not both.")
    if dropout_p != 0.0:
        raise ValueError("NPU Ulysses attention does not support dropout_p.")
    if window_size != (-1, -1):
        raise ValueError("NPU Ulysses attention does not support local attention window_size.")
    if softcap != 0.0:
        raise ValueError("NPU Ulysses attention does not support softcap.")
    if alibi_slopes is not None:
        raise ValueError("NPU Ulysses attention does not support alibi_slopes.")
    if deterministic:
        raise ValueError("NPU Ulysses attention does not support deterministic mode.")
    if return_attn_probs:
        raise ValueError("NPU Ulysses attention does not support return_attn_probs.")

    try:
        import torch_npu
    except ImportError as exc:  # pragma: no cover - optional NPU dependency
        raise ImportError("Ulysses attn_implementation='npu_fusion_attention_v2' requires torch_npu.") from exc

    if scaling is None:
        scaling = query.shape[-1] ** -0.5

    pre_tokens = 65535
    next_tokens = 0 if is_causal else 65535
    atten_mask = None
    actual_seq_qlen = None
    actual_seq_kvlen = None
    if attention_mask is not None:
        attention_mask = _normalize_2d_attention_mask(attention_mask)
        atten_mask = _build_npu_attention_mask(attention_mask, is_causal=is_causal)
        next_tokens = 65535
    elif has_cu_seqlens:
        total_q = query.shape[0] * query.shape[1]
        total_k = key.shape[0] * key.shape[1]
        cu_seq_lens_q, cu_seq_lens_k = _validate_cu_seqlens(
            cu_seq_lens_q,
            cu_seq_lens_k,
            total_q=total_q,
            total_k=total_k,
            device=query.device,
        )
        actual_seq_qlen = [int(length) for length in cu_seq_lens_q[1:].detach().cpu().tolist()]
        actual_seq_kvlen = [int(length) for length in cu_seq_lens_k[1:].detach().cpu().tolist()]

    output = torch_npu.npu_fusion_attention_v2(
        query,
        key,
        value,
        head_num=query.shape[-2],
        input_layout="BSND",
        atten_mask=atten_mask,
        scale=scaling,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )
    return output[0] if isinstance(output, tuple) else output


def _build_npu_attention_mask(attention_mask: torch.Tensor, *, is_causal: bool) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    blocked = ~attention_mask[:, None, None, :]
    if is_causal:
        positions = torch.arange(seq_len, device=attention_mask.device)
        causal_blocked = positions[None, :] > positions[:, None]
        blocked = blocked | causal_blocked[None, None, :, :]
    return blocked.expand(batch_size, 1, seq_len, seq_len).contiguous()


def _torch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_implementation: str,
    attention_mask: torch.Tensor | None,
    cu_seq_lens_q: torch.Tensor | None,
    cu_seq_lens_k: torch.Tensor | None,
    max_length_q: int | None,
    max_length_k: int | None,
    dropout_p: float,
    scaling: float | None,
    is_causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    return_attn_probs: bool,
) -> torch.Tensor:
    if window_size != (-1, -1):
        raise ValueError("Torch Ulysses attention does not support local attention window_size.")
    if softcap != 0.0:
        raise ValueError("Torch Ulysses attention does not support softcap.")
    if alibi_slopes is not None:
        raise ValueError("Torch Ulysses attention does not support alibi_slopes.")
    if return_attn_probs:
        raise ValueError("Torch Ulysses attention does not support return_attn_probs.")
    if attention_mask is not None and _has_cu_seqlens(cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k):
        raise ValueError("Pass either attention_mask or cu_seq_lens, not both.")

    if _has_cu_seqlens(cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k):
        return _torch_varlen_attention(
            query,
            key,
            value,
            attn_implementation=attn_implementation,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            dropout_p=dropout_p,
            scaling=scaling,
            is_causal=is_causal,
        )

    attn_mask = None
    if attention_mask is not None:
        attention_mask = _normalize_2d_attention_mask(attention_mask)
        query_len = query.shape[1]
        key_len = key.shape[1]
        if attention_mask.shape != (query.shape[0], key_len):
            raise ValueError(
                "attention_mask must have shape [batch, key_seq_len], "
                f"got {tuple(attention_mask.shape)}, expected {(query.shape[0], key_len)}."
            )
        attn_mask = attention_mask[:, None, None, :]
        if is_causal:
            query_positions = torch.arange(query_len, device=query.device)
            key_positions = torch.arange(key_len, device=query.device)
            causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            attn_mask = attn_mask & causal_mask[None, None, :, :]
            is_causal = False

    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    enable_gqa = q.shape[1] != k.shape[1]
    backend = _sdpa_backend(attn_implementation)
    if backend is None:
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scaling,
            enable_gqa=enable_gqa,
        )
    else:
        from torch.nn.attention import sdpa_kernel

        with sdpa_kernel(backend):
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scaling,
                enable_gqa=enable_gqa,
            )
    return output.transpose(1, 2).contiguous()


def _torch_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_implementation: str,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    dropout_p: float,
    scaling: float | None,
    is_causal: bool,
) -> torch.Tensor:
    total_q = query.shape[0] * query.shape[1]
    total_k = key.shape[0] * key.shape[1]
    cu_seq_lens_q, cu_seq_lens_k = _validate_cu_seqlens(
        cu_seq_lens_q,
        cu_seq_lens_k,
        total_q=total_q,
        total_k=total_k,
        device=query.device,
    )
    segment_q = torch.arange(cu_seq_lens_q.numel() - 1, device=query.device)
    segment_k = torch.arange(cu_seq_lens_k.numel() - 1, device=query.device)
    q_lengths = (cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]).to(device=query.device)
    k_lengths = (cu_seq_lens_k[1:] - cu_seq_lens_k[:-1]).to(device=query.device)
    q_segment_ids = torch.repeat_interleave(segment_q, q_lengths)
    k_segment_ids = torch.repeat_interleave(segment_k, k_lengths)
    attn_mask = q_segment_ids[:, None] == k_segment_ids[None, :]

    if is_causal:
        q_positions = torch.arange(total_q, device=query.device) - torch.repeat_interleave(
            cu_seq_lens_q[:-1].to(device=query.device),
            q_lengths,
        )
        k_positions = torch.arange(total_k, device=query.device) - torch.repeat_interleave(
            cu_seq_lens_k[:-1].to(device=query.device),
            k_lengths,
        )
        attn_mask = attn_mask & (k_positions[None, :] <= q_positions[:, None])

    q = query.reshape(1, total_q, query.shape[2], query.shape[3]).transpose(1, 2)
    k = key.reshape(1, total_k, key.shape[2], key.shape[3]).transpose(1, 2)
    v = value.reshape(1, total_k, value.shape[2], value.shape[3]).transpose(1, 2)
    backend = _sdpa_backend(attn_implementation)
    if backend is None:
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask[None, None, :, :],
            dropout_p=dropout_p,
            is_causal=False,
            scale=scaling,
            enable_gqa=q.shape[1] != k.shape[1],
        )
    else:
        from torch.nn.attention import sdpa_kernel

        with sdpa_kernel(backend):
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask[None, None, :, :],
                dropout_p=dropout_p,
                is_causal=False,
                scale=scaling,
                enable_gqa=q.shape[1] != k.shape[1],
            )
    return output.transpose(1, 2).reshape_as(query).contiguous()


def _sdpa_backend(attn_implementation: str):
    if attn_implementation == "sdpa":
        return None

    from torch.nn.attention import SDPBackend

    backends = {
        "torch_math": SDPBackend.MATH,
        "torch_flash": SDPBackend.FLASH_ATTENTION,
        "torch_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "torch_cudnn": SDPBackend.CUDNN_ATTENTION,
    }
    return backends[attn_implementation]


@dataclass(frozen=True)
class CPGradSyncStats:
    """Summary of one CP gradient sync."""

    buckets: int
    tensors: int
    bytes: int


class CPGradSync:
    """Bucketed context-parallel gradient synchronizer."""

    def __init__(
        self,
        model: nn.Module,
        cp_mesh,
        *,
        bucket_mb: int = 128,
        reduce_dtype: str = "float32",
        exclude: Sequence[str] = (),
    ) -> None:
        if bucket_mb <= 0:
            raise ValueError(f"bucket_mb must be positive, got {bucket_mb}.")
        if reduce_dtype not in {"same", "float32"}:
            raise ValueError(f"reduce_dtype must be 'same' or 'float32', got {reduce_dtype!r}.")

        self.model = model
        self.group = cp_mesh.get_group() if hasattr(cp_mesh, "get_group") else cp_mesh
        self.bucket_bytes = int(bucket_mb) * 1024 * 1024
        self.reduce_dtype = reduce_dtype
        self.exclude = tuple(str(pattern) for pattern in exclude)

    @torch.no_grad()
    def sync(self) -> CPGradSyncStats:
        """Sum trainable gradients across the context-parallel group."""
        if get_world_size(self.group) == 1:
            return CPGradSyncStats(buckets=0, tensors=0, bytes=0)

        buckets = 0
        tensors = 0
        bytes_count = 0
        grouped_items: dict[
            tuple[torch.device, torch.dtype, torch.dtype],
            list[tuple[torch.Tensor, torch.Tensor]],
        ] = {}

        for name, parameter in self.model.named_parameters(remove_duplicate=True):
            grad = parameter.grad
            if not parameter.requires_grad or grad is None:
                continue
            if any(fnmatch.fnmatch(name, pattern) for pattern in self.exclude):
                continue

            local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
            wait = getattr(local_grad, "wait", None)
            if callable(wait):
                local_grad = wait()
            if local_grad.is_sparse:
                raise ValueError(f"CP grad sync does not support sparse gradients: {name}.")

            flat_grad = local_grad.detach().reshape(-1)
            grad_dtype = flat_grad.dtype
            reduce_dtype = grad_dtype
            if self.reduce_dtype == "float32" and grad_dtype in (torch.float16, torch.bfloat16):
                reduce_dtype = torch.float32
            grouped_items.setdefault((flat_grad.device, grad_dtype, reduce_dtype), []).append((local_grad, flat_grad))

        for (device, _grad_dtype, reduce_dtype), items in grouped_items.items():
            bucket_items: list[tuple[torch.Tensor, torch.Tensor]] = []
            bucket_numel = 0
            reduce_dtype_size = torch.empty((), device=device, dtype=reduce_dtype).element_size()

            for local_grad, flat_grad in items:
                item_bytes = flat_grad.numel() * reduce_dtype_size
                if bucket_items and (bucket_numel + flat_grad.numel()) * reduce_dtype_size > self.bucket_bytes:
                    self._sync_bucket(bucket_items, bucket_numel, device, reduce_dtype)
                    buckets += 1
                    bucket_items = []
                    bucket_numel = 0

                bucket_items.append((local_grad, flat_grad))
                bucket_numel += flat_grad.numel()
                tensors += 1
                bytes_count += item_bytes

            if bucket_items:
                self._sync_bucket(bucket_items, bucket_numel, device, reduce_dtype)
                buckets += 1

        return CPGradSyncStats(buckets=buckets, tensors=tensors, bytes=bytes_count)

    def _sync_bucket(
        self,
        bucket_items: list[tuple[torch.Tensor, torch.Tensor]],
        bucket_numel: int,
        device: torch.device,
        reduce_dtype: torch.dtype,
    ) -> None:
        bucket = torch.empty(bucket_numel, device=device, dtype=reduce_dtype)
        offset = 0
        for _, flat_grad in bucket_items:
            next_offset = offset + flat_grad.numel()
            bucket[offset:next_offset].copy_(flat_grad)
            offset = next_offset

        dist.all_reduce(bucket, op=dist.ReduceOp.SUM, group=self.group)

        offset = 0
        for local_grad, flat_grad in bucket_items:
            next_offset = offset + flat_grad.numel()
            local_grad.copy_(bucket[offset:next_offset].view_as(local_grad))
            offset = next_offset


def attach_cp_grad_sync(
    model: nn.Module,
    cp_mesh,
    *,
    bucket_mb: int = 128,
    reduce_dtype: str = "float32",
    exclude: Sequence[str] = (),
) -> nn.Module:
    """Attach a CP gradient synchronizer to a model."""
    model._cp_grad_sync = CPGradSync(
        model,
        cp_mesh,
        bucket_mb=bucket_mb,
        reduce_dtype=reduce_dtype,
        exclude=exclude,
    )
    return model


def sync_cp_grads(model: nn.Module) -> CPGradSyncStats | None:
    """Run CP gradient sync when the model has a synchronizer attached."""
    sync = getattr(model, "_cp_grad_sync", None)
    if sync is None:
        return None
    return sync.sync()


def parallelize_model_with_context_parallel(model: nn.Module, cp_mesh, backend_kwargs):
    """Patch model attention modules to use Ulysses context parallel attention.

    ``model.CP_MODULE_CONFIG`` must map attention module class names to config
    dicts, for example ``{"Qwen3VLTextAttention": {"qkv_layout": "BHSD"}}``.
    Supported QKV layouts are ``BSHD`` and ``BHSD``.

    Attention masks may be either 2D 0/1 masks with local or global sequence
    length, or global FlashAttention-style ``cu_seq_lens_*`` metadata covering
    the flattened ``[batch, global_seq]`` sequence.
    """
    from transformers.modeling_utils import AttentionInterface

    attention = UlyssesSPAttention(
        cp_mesh=cp_mesh,
        **(backend_kwargs or {}),
    )
    module_qkv_layouts: dict[int, str] = {}

    def ulysses_sp_attention(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        scaling: float | None = None,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "Ulysses SP attention expects 4D query/key/value tensors, "
                f"got {tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}."
            )

        qkv_layout = module_qkv_layouts.get(id(module), _NATIVE_QKV_LAYOUT)
        if qkv_layout == _HF_QKV_LAYOUT:
            query = query.transpose(1, 2).contiguous()
            key = key.transpose(1, 2).contiguous()
            value = value.transpose(1, 2).contiguous()

        return (
            attention(
                query,
                key,
                value,
                attention_mask=attention_mask,
                cu_seq_lens_q=kwargs.get("cu_seq_lens_q"),
                cu_seq_lens_k=kwargs.get("cu_seq_lens_k"),
                max_length_q=kwargs.get("max_length_q"),
                max_length_k=kwargs.get("max_length_k"),
                dropout_p=dropout,
                scaling=scaling,
                is_causal=bool(kwargs.get("is_causal", getattr(module, "is_causal", True))),
                window_size=kwargs.get("window_size", (-1, -1)),
                softcap=float(kwargs.get("softcap", 0.0)),
                alibi_slopes=kwargs.get("alibi_slopes"),
                deterministic=bool(kwargs.get("deterministic", False)),
                return_attn_probs=bool(kwargs.get("return_attn_probs", False)),
            ),
            None,
        )

    AttentionInterface._global_mapping["ulysses_sp"] = ulysses_sp_attention
    cp_module_config = getattr(model, "CP_MODULE_CONFIG", {})
    if not isinstance(cp_module_config, dict):
        raise TypeError("CP_MODULE_CONFIG must be a dict mapping module class names to config dicts.")

    for module_name, module in model.named_modules():
        module_class_name = module.__class__.__name__
        module_config = cp_module_config.get(module_class_name)
        if module_config is None:
            continue
        if not isinstance(module_config, dict):
            raise TypeError(f"CP_MODULE_CONFIG[{module_class_name!r}] must be a config dict.")

        qkv_layout = str(module_config.get("qkv_layout", _NATIVE_QKV_LAYOUT)).strip().upper()
        if qkv_layout not in _SUPPORTED_QKV_LAYOUTS:
            raise ValueError(
                f"Unsupported CP QKV layout for {module_class_name!r}: {qkv_layout!r}. "
                f"Supported layouts: {_SUPPORTED_QKV_LAYOUTS_MESSAGE}."
            )

        module_qkv_layouts[id(module)] = qkv_layout
        module.config._attn_implementation = "ulysses_sp"

    return model
