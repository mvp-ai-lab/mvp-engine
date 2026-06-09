"""Local Ulysses sequence-parallel attention."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


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
        return all_to_all_4d(value, scatter_dim=scatter_dim, gather_dim=gather_dim, group=group, use_sync=use_sync)

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


class UlyssesAttention(nn.Module):
    """Ulysses attention over context-sharded sequence chunks."""

    def __init__(
        self,
        *,
        sequence_process_group: dist.ProcessGroup | None,
        attn_impl: str = "fa",
    ) -> None:
        super().__init__()
        self.sequence_process_group = sequence_process_group
        self.attn_impl = str(attn_impl)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = False,
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

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5

        context = run_attention(
            q,
            k,
            v,
            attn_impl=self.attn_impl,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )
        if isinstance(context, tuple):
            context = context[0]

        return SeqAllToAll4D.apply(self.sequence_process_group, context, 1, 2, False)


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

    world_size = _get_group_world_size(group)
    if world_size == 1:
        return value.contiguous()

    if scatter_dim == 2 and gather_dim == 1:
        return _scatter_heads_gather_sequence(value, world_size, group=group, use_sync=use_sync)
    if scatter_dim == 1 and gather_dim == 2:
        return _scatter_sequence_gather_heads(value, world_size, group=group, use_sync=use_sync)
    raise ValueError("Ulysses all-to-all supports only scatter/gather dim pairs (2, 1) and (1, 2).")


def run_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_impl: str,
    dropout_p: float,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    deterministic: bool,
    return_attn_probs: bool,
) -> torch.Tensor:
    """Dispatch local attention over full sequence and local head shards."""
    normalized = _normalize_attn_impl(attn_impl)
    if normalized == "fa":
        return _flash_attention(
            query,
            key,
            value,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )
    if normalized == "npu":
        return _npu_attention(
            query,
            key,
            value,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
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
        attn_impl=normalized,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        return_attn_probs=return_attn_probs,
    )


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


def _flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    dropout_p: float,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    deterministic: bool,
    return_attn_probs: bool,
) -> torch.Tensor:
    try:
        from flash_attn import flash_attn_func
    except ImportError as exc:  # pragma: no cover - optional CUDA dependency
        raise ImportError("Ulysses attn_impl='fa' requires flash-attn.") from exc

    return flash_attn_func(
        query,
        key,
        value,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
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
    dropout_p: float,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    deterministic: bool,
    return_attn_probs: bool,
) -> torch.Tensor:
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
        raise ImportError("Ulysses attn_impl='npu' requires torch_npu.") from exc

    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5

    output = torch_npu.npu_fusion_attention_v2(
        query,
        key,
        value,
        head_num=query.shape[-2],
        input_layout="BSND",
        scale=softmax_scale,
        pre_tokens=65535,
        next_tokens=0 if causal else 65535,
    )
    return output[0] if isinstance(output, tuple) else output


def _torch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_impl: str,
    dropout_p: float,
    softmax_scale: float | None,
    causal: bool,
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

    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    enable_gqa = q.shape[1] != k.shape[1]
    backend = _sdpa_backend(attn_impl)
    if backend is None:
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
            enable_gqa=enable_gqa,
        )
    else:
        from torch.nn.attention import sdpa_kernel

        with sdpa_kernel(backend):
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=softmax_scale,
                enable_gqa=enable_gqa,
            )
    return output.transpose(1, 2).contiguous()


def _normalize_attn_impl(attn_impl: str) -> str:
    aliases = {
        "fa": "fa",
        "flash_attn": "fa",
        "flash_attention_2": "fa",
        "sdpa": "torch",
        "torch": "torch",
        "torch_math": "torch_math",
        "torch_flash": "torch_flash",
        "torch_efficient": "torch_efficient",
        "torch_cudnn": "torch_cudnn",
        "npu": "npu",
        "torch_npu": "npu",
        "npu_fa": "npu",
    }
    try:
        return aliases[attn_impl]
    except KeyError as exc:
        raise ValueError(f"Unsupported Ulysses attention implementation: {attn_impl!r}.") from exc


def _sdpa_backend(attn_impl: str):
    if attn_impl == "torch":
        return None

    from torch.nn.attention import SDPBackend

    backends = {
        "torch_math": SDPBackend.MATH,
        "torch_flash": SDPBackend.FLASH_ATTENTION,
        "torch_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "torch_cudnn": SDPBackend.CUDNN_ATTENTION,
    }
    return backends[attn_impl]


def _get_group_world_size(group: dist.ProcessGroup | None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return int(dist.get_world_size(group))


def _maybe_sync(value: torch.Tensor, use_sync: bool) -> None:
    if use_sync and value.device.type == "cuda":
        torch.cuda.synchronize(value.device)
    elif use_sync and value.device.type == "npu":
        torch.npu.synchronize(value.device)
