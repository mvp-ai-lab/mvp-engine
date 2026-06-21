#!/usr/bin/env python3
"""Minimal Ulysses CP equivalence demo.

Run CP with:
    torchrun --nproc_per_node=2 playground/cp_equivalence_demo.py --device cpu

Run CP + TP with:
    torchrun --nproc_per_node=4 playground/cp_equivalence_demo.py --device cuda --tp-size 2
"""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh

from mvp_engine.distributed.cp import (
    UlyssesSPAttention,
    attach_cp_grad_sync,
    run_attention,
    sync_cp_grads,
)
from mvp_engine.distributed.tp import parallelize_model_with_tensor_parallel
from mvp_engine.utils.log import get_logger, init_logger

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class ProcessGroupMesh:
    """Tiny adapter for APIs that expect a mesh-like object with get_group()."""

    def __init__(self, group: dist.ProcessGroup) -> None:
        self.group = group

    def get_group(self) -> dist.ProcessGroup:
        return self.group


class TinyAttentionBlock(nn.Module):
    """Tiny self-attention block that can run full-sequence or Ulysses CP attention."""

    TP_MODULE_CONFIG = {
        "TinyAttentionBlock": {
            "q_proj": "col",
            "k_proj": "col",
            "v_proj": "col",
            "out": "row",
        },
    }

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        cp_group: dist.ProcessGroup | None = None,
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.embed_dim = embed_dim
        self.local_embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_implementation = attn_implementation
        self.cp_attention = (
            UlyssesSPAttention(cp_group, attn_implementation=attn_implementation) if cp_group is not None else None
        )
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        cu_seq_lens_q: torch.Tensor | None = None,
        cu_seq_lens_k: torch.Tensor | None = None,
        max_length_q: int | None = None,
        max_length_k: int | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        scaling = self.head_dim**-0.5

        if self.cp_attention is None:
            context = run_attention(
                query,
                key,
                value,
                attn_implementation=self.attn_implementation,
                attention_mask=attention_mask,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                max_length_q=max_length_q,
                max_length_k=max_length_k,
                dropout_p=0.0,
                scaling=scaling,
                is_causal=False,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=False,
            )
        else:
            context = self.cp_attention(
                query,
                key,
                value,
                attention_mask=attention_mask,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                max_length_q=max_length_q,
                max_length_k=max_length_k,
                dropout_p=0.0,
                scaling=scaling,
                is_causal=False,
            )

        return self.out(context.reshape(batch_size, seq_len, self.local_embed_dim))


def update_tiny_attention_block_for_tp(module: nn.Module, tp_mesh) -> None:
    module.num_heads //= tp_mesh.size()
    module.local_embed_dim = module.num_heads * module.head_dim


TinyAttentionBlock.TP_MODULE_POSTPROCESSORS = {
    "TinyAttentionBlock": update_tiny_attention_block_for_tp,
}


def run_demo(rank: int, world_size: int, args: argparse.Namespace, init_file: str | None = None) -> None:
    local_rank = rank if init_file is not None else int(os.getenv("LOCAL_RANK", "0"))
    device_type = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_type == "auto":
        device_type = "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dtype = DTYPES[args.dtype]
    if args.attn_implementation == "flash_attention_2" and dtype == torch.float32:
        raise ValueError("flash_attention_2 requires --dtype bfloat16 or --dtype float16.")

    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        if backend == "gloo":
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        init_method = f"file://{init_file}" if init_file is not None else "env://"
        dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=world_size)

    if world_size % args.tp_size != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by tp_size ({args.tp_size}).")
    cp_size = world_size // args.tp_size
    cp_rank = rank // args.tp_size
    cp_group = dist.group.WORLD if dist.is_initialized() else None
    cp_mesh = ProcessGroupMesh(cp_group) if cp_group is not None else None
    tp_mesh = None
    if dist.is_initialized():
        device_mesh = init_device_mesh(
            device.type,
            mesh_shape=(cp_size, args.tp_size),
            mesh_dim_names=("context", "tensor"),
        )
        cp_mesh = device_mesh["context"]
        cp_group = cp_mesh.get_group()
        tp_mesh = device_mesh["tensor"] if args.tp_size > 1 else None

    if args.seq_len % cp_size != 0:
        raise ValueError(f"seq_len ({args.seq_len}) must be divisible by cp_size ({cp_size}).")
    if args.num_heads % world_size != 0:
        raise ValueError(f"num_heads ({args.num_heads}) must be divisible by world_size ({world_size}).")

    torch.manual_seed(1234)
    full_model = TinyAttentionBlock(
        args.embed_dim,
        args.num_heads,
        attn_implementation=args.attn_implementation,
    ).to(device=device, dtype=dtype)
    cp_model = TinyAttentionBlock(
        args.embed_dim,
        args.num_heads,
        cp_group=cp_mesh,
        attn_implementation=args.attn_implementation,
    ).to(device=device, dtype=dtype)
    cp_model.load_state_dict(full_model.state_dict())
    if tp_mesh is not None:
        if get_logger() is None:
            init_logger([])
        parallelize_model_with_tensor_parallel(cp_model, tp_mesh)
    attach_cp_grad_sync(cp_model, cp_mesh)

    torch.manual_seed(5678)
    full_input = torch.randn(
        args.batch_size,
        args.seq_len,
        args.embed_dim,
        device=device,
        dtype=dtype,
    ).requires_grad_(True)
    target = torch.randn_like(full_input)
    local_seq_len = args.seq_len // cp_size
    local_start = cp_rank * local_seq_len
    local_end = local_start + local_seq_len
    cp_input = full_input.detach()[:, local_start:local_end].clone().requires_grad_(True)
    full_attention_mask = None
    local_attention_mask = None
    cu_seq_lens = None
    max_length = None
    if args.mask_mode == "padding":
        lengths = torch.full((args.batch_size,), args.seq_len, device=device, dtype=torch.long)
        if args.batch_size > 1:
            lengths[-1] = max(1, args.seq_len - 2)
        positions = torch.arange(args.seq_len, device=device)
        full_attention_mask = (positions.unsqueeze(0) < lengths.unsqueeze(1)).to(dtype=torch.long)
        local_attention_mask = full_attention_mask[:, local_start:local_end].contiguous()
    elif args.mask_mode == "cu":
        segment_lengths = []
        for batch_index in range(args.batch_size):
            split = max(1, min(args.seq_len - 1, args.seq_len // 2 + batch_index % 2))
            segment_lengths.extend([split, args.seq_len - split])
        segment_lengths = torch.tensor(segment_lengths, device=device, dtype=torch.int32)
        cu_seq_lens = torch.zeros(segment_lengths.numel() + 1, device=device, dtype=torch.int32)
        cu_seq_lens[1:] = torch.cumsum(segment_lengths, dim=0)
        max_length = int(segment_lengths.max().item())

    full_output = full_model(
        full_input,
        attention_mask=full_attention_mask,
        cu_seq_lens_q=cu_seq_lens,
        cu_seq_lens_k=cu_seq_lens,
        max_length_q=max_length,
        max_length_k=max_length,
    )
    cp_output = cp_model(
        cp_input,
        attention_mask=local_attention_mask,
        cu_seq_lens_q=cu_seq_lens,
        cu_seq_lens_k=cu_seq_lens,
        max_length_q=max_length,
        max_length_k=max_length,
    )
    if dist.is_initialized() and cp_size > 1:
        gathered_outputs = [torch.empty_like(cp_output) for _ in range(cp_size)]
        dist.all_gather(gathered_outputs, cp_output.detach(), group=cp_group)
        cp_full_output = torch.cat(gathered_outputs, dim=1)
    else:
        cp_full_output = cp_output.detach()

    full_loss = (full_output.float() * target.float()).sum()
    cp_loss = (cp_output.float() * target[:, local_start:local_end].float()).sum()
    full_loss.backward()
    cp_loss.backward()
    sync_cp_grads(cp_model)

    cp_loss_sum = cp_loss.detach().clone()
    if dist.is_initialized() and cp_size > 1:
        dist.all_reduce(cp_loss_sum, op=dist.ReduceOp.SUM, group=cp_group)

    output_diff = (full_output.detach() - cp_full_output).abs().max()
    loss_diff = (full_loss.detach() - cp_loss_sum).abs()
    input_grad_diff = (full_input.grad[:, local_start:local_end] - cp_input.grad).abs().max()
    cp_param_grads = {}
    for name, parameter in cp_model.named_parameters():
        grad = parameter.grad
        if hasattr(grad, "full_tensor"):
            grad = grad.full_tensor()
        cp_param_grads[name] = grad
    grad_diffs = {
        name: (parameter.grad - cp_param_grads[name]).abs().max() for name, parameter in full_model.named_parameters()
    }
    grad_diff = torch.stack([*grad_diffs.values(), input_grad_diff, output_diff, loss_diff]).max()
    passed = bool(torch.allclose(full_output.detach(), cp_full_output, rtol=args.rtol, atol=args.atol))
    passed = passed and bool(torch.allclose(full_loss.detach(), cp_loss_sum, rtol=args.rtol, atol=args.atol))
    passed = passed and bool(
        torch.allclose(
            full_input.grad[:, local_start:local_end],
            cp_input.grad,
            rtol=args.rtol,
            atol=args.atol,
        )
    )
    passed = passed and all(
        torch.allclose(parameter.grad, cp_param_grads[name], rtol=args.rtol, atol=args.atol)
        for name, parameter in full_model.named_parameters()
    )

    if rank == 0:
        print(
            f"world_size={world_size}, cp_size={cp_size}, tp_size={args.tp_size}, "
            f"device={device.type}, dtype={args.dtype}, attn={args.attn_implementation}, mask={args.mask_mode}"
        )
        print(f"max output diff: {output_diff.item():.6e}")
        print(f"max loss diff:   {loss_diff.item():.6e}")
        print(f"max input grad diff on local shard: {input_grad_diff.item():.6e}")
        for name, diff in grad_diffs.items():
            print(f"max param grad diff [{name}]: {diff.item():.6e}")
        print(f"max diff overall: {grad_diff.item():.6e}")
        print("PASS" if passed else "FAIL")

    if dist.is_initialized():
        status = torch.tensor([1 if passed else 0], device=device)
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
        dist.destroy_process_group()
        if not bool(status.item()):
            raise SystemExit(1)
    elif not passed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--mask-mode", choices=("none", "padding", "cu"), default="none")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--spawn", type=int, default=0)
    parser.add_argument("--atol", type=float, default=5e-5)
    parser.add_argument("--rtol", type=float, default=5e-4)
    args = parser.parse_args()

    if args.spawn > 0:
        with tempfile.TemporaryDirectory() as tmpdir:
            mp.spawn(
                run_demo,
                args=(args.spawn, args, os.path.join(tmpdir, "dist_init")),
                nprocs=args.spawn,
                join=True,
            )
        return

    run_demo(
        rank=int(os.getenv("RANK", "0")),
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        args=args,
    )


if __name__ == "__main__":
    main()
