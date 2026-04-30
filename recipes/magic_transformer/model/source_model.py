import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    max_seq_len: int = 2048
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2  # GQA; must divide tp_size when TP shards columns.
    n_layers: int = 8  # Even counts keep the dual-stream layout symmetric.
    dropout: float = 0.1
    rope_base: float = 10000.0

    # [1] MoD
    mod_top_k_ratio: float = 0.5  # Fraction of tokens kept by each layer.
    # NOTE: top-k must stay identical across all TP ranks; do not sample per rank.

    # [3] Dual-stream residuals
    dual_stream: bool = True  # Set False to fall back to a single stream for ablations.

    # [4] Router Feedback EMA
    router_ema_decay: float = 0.9
    # NOTE: ema_state is a buffer. FSDP2 does not sync it; multi-node runs need manual all-reduce.

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        assert self.n_layers % 2 == 0, "n_layers should be even to keep the dual-stream layout symmetric"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_hidden(self) -> int:
        raw = int(2 / 3 * 4 * self.d_model)
        return ((raw + 63) // 64) * 64


# =============================================================================
# Building blocks
# =============================================================================


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        scale: optional multiplicative injection with shape (B, T, 1) from Router Feedback [4].
        NOTE: under TP, scale comes from the router. If the router is column-sharded,
        it must be all-reduced first.
        """
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        normed = (x / rms) * self.weight
        if scale is not None:
            normed = normed * scale
        return normed


def build_rope_cache(
    seq_len: int, head_dim: int, base: float = 10000.0, device=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# -----------------------------------------------------------------------------
# GQA (supports external KV inputs for cross-layer KV tying [2])
# -----------------------------------------------------------------------------


class GQAttention(nn.Module):
    """
    NOTE: with TP column sharding, q/k/v_proj shard on output dim and out_proj shards
    on input dim, so out_proj needs an all-reduce afterward.
    NOTE: during cross-layer KV tying, ext_kv carries the previous layer's graph.
    FSDP2 activation-checkpoint recompute reruns the even-layer forward pass, while
    the odd layer can still hold a stale ext_kv reference and trigger use-after-free.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ext_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns (output, (K_raw, V_raw)).
        K_raw is the original K before RoPE so the next odd layer can reuse it and apply RoPE there.
        When ext_kv is provided, k/v projection is skipped and the external tensors are used directly.

        NOTE: this intentionally keeps a known bug for testing: when ext_kv is passed in,
        cos/sin is applied to K again, so odd layers receive RoPE twice. The correct fix is to
        cache already-rotated K and skip the second rotation when reusing it.
        """
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if ext_kv is None:
            k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        else:
            k, v = ext_kv  # Reuse (K_raw, V) from the previous even layer.

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)  # NOTE: ext_kv receives RoPE a second time here.

        repeat = self.n_heads // self.n_kv_heads
        if repeat > 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.drop.p if self.training else 0.0,
                is_causal=(mask is None),
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is None:
                causal = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
                attn = attn.masked_fill(~causal, float("-inf"))
            else:
                attn = attn + mask
            attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
            attn = self.drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        # In the ext_kv case, return the original non-expanded K so later layers can still reuse it.
        k_raw = k if ext_kv is None else ext_kv[0]
        return self.out_proj(out), (k_raw, v if ext_kv is None else ext_kv[1])


# -----------------------------------------------------------------------------
# SwiGLU FFN
# -----------------------------------------------------------------------------


class SwiGLUFFN(nn.Module):
    """NOTE: under TP, gate_proj/up_proj shard by columns and down_proj shards by rows."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        hidden = cfg.ffn_hidden
        self.gate_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.drop(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# =============================================================================
# [1] MoD router + scatter/gather
# =============================================================================


class MoDRouter(nn.Module):
    """
    Scores each token and returns the top-k subset, the remainder, and continuous scores
    that can be reused by the feedback path.

    NOTE: router_linear consumes the full x tensor in d_model space.
    Under TP sequence parallelism, x only contains a token shard so top-k is not global.
    Under TP tensor parallelism, d_model is sharded so router_linear needs an all-gather first.
    NOTE: top-k indices must be identical across TP ranks inside the same batch item, or
    scatter_back will scramble sequence order and gradient communication shapes will diverge.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.top_k_ratio = cfg.mod_top_k_ratio
        self.router_linear = nn.Linear(cfg.d_model, 1, bias=False)

    def forward(self, x: torch.Tensor):
        bsz, seq_len, dim = x.shape
        top_k = max(1, int(seq_len * self.top_k_ratio))

        router_score = self.router_linear(x)  # (B, T, 1) continuous score
        score_flat = router_score.squeeze(-1)  # (B, T)

        _, top_indices = score_flat.topk(top_k, dim=1, sorted=False)  # (B, k)
        top_indices, _ = top_indices.sort(dim=1)  # Preserve original token order.

        # rest indices: complement set
        mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=x.device)
        mask.scatter_(1, top_indices, True)
        rest_indices = torch.stack(
            [(~mask[batch_idx]).nonzero(as_tuple=False).squeeze(-1) for batch_idx in range(bsz)],
            dim=0,
        )  # (B, T-k)

        idx_expanded = top_indices.unsqueeze(-1).expand(-1, -1, dim)
        rest_expanded = rest_indices.unsqueeze(-1).expand(-1, -1, dim)

        selected = torch.gather(x, 1, idx_expanded)  # (B, k, D)
        rest = torch.gather(x, 1, rest_expanded)  # (B, T-k, D)

        return selected, rest, top_indices, rest_indices, router_score


def scatter_back(
    selected_out: torch.Tensor,  # (B, k, D)
    rest_x: torch.Tensor,  # (B, T-k, D)
    top_indices: torch.Tensor,  # (B, k)
    rest_indices: torch.Tensor,  # (B, T-k)
    seq_len: int,
) -> torch.Tensor:
    """
    Scatter processed tokens and skipped tokens back into the original sequence order.

    NOTE: scatter_ creates sparse gradients (IndexPutBackward).
    FSDP2 reduce-scatter hooks assume dense gradients, and sparse gradients can trip
    contiguous asserts in those hooks. out.scatter_(...).contiguous() or index_put can help.
    NOTE: if selected_out and rest_x come from different TP shards, the scatter result is wrong.
    """
    bsz, _, dim = selected_out.shape
    out = torch.empty(bsz, seq_len, dim, dtype=selected_out.dtype, device=selected_out.device)
    out.scatter_(1, top_indices.unsqueeze(-1).expand_as(selected_out), selected_out)
    out.scatter_(1, rest_indices.unsqueeze(-1).expand_as(rest_x), rest_x)
    return out


# =============================================================================
# [3] Dual-stream fusion gate
# =============================================================================


class DualStreamFusion(nn.Module):
    """
    g = sigmoid(W · cat(x_c, x_m))
    out_c =      g * x_c + (1-g) * x_m
    out_m = (1-g)* x_c +      g * x_m

    NOTE: gate_proj has shape (d_model, 2 * d_model).
    Under TP, x_c and x_m each hold only d_model / tp_size features, so concatenating them
    produces 2 * d_model / tp_size inputs and no longer matches gate_proj. Both streams must
    be all-gathered before concatenation, which is the most expensive TP communication in this block.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, x_c: torch.Tensor, x_m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.gate_proj(torch.cat([x_c, x_m], dim=-1)))
        out_c = gate * x_c + (1 - gate) * x_m
        out_m = (1 - gate) * x_c + gate * x_m
        return out_c, out_m


# =============================================================================
# Full transformer block
# =============================================================================


class TransformerBlock(nn.Module):
    """
    Integrates all four computation-graph changes.

    Even-layer role: x_c -> attn (MoD), x_m -> ffn (MoD), then DualStreamFusion.
    Odd-layer role: x_m -> attn (MoD, ext_kv from previous even layer),
    x_c -> ffn (MoD), then fusion.

    NOTE: FSDP2 treats TransformerBlock as one flat unit.
    kv_output is an activation that spans two flat units, and FSDP2 does not track activation
    lifetime across unit boundaries.
    NOTE: inject_scale (next_scale) carries gradients from layer i to layer i+1, which can
    conflict with FSDP2 reduce-scatter timing because layer i+1 gradients may be consumed before
    layer i finishes its own reduce-scatter hooks.
    """

    def __init__(self, cfg: TransformerConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.is_even = layer_idx % 2 == 0

        # Each stream gets two norms: pre-norm before the sub-layer and post-norm after residual add.
        self.norm_c1 = RMSNorm(cfg.d_model)
        self.norm_m1 = RMSNorm(cfg.d_model)
        self.norm_c2 = RMSNorm(cfg.d_model)
        self.norm_m2 = RMSNorm(cfg.d_model)

        self.attn = GQAttention(cfg)
        self.ffn = SwiGLUFFN(cfg)
        self.router = MoDRouter(cfg)

        if cfg.dual_stream:
            self.fusion = DualStreamFusion(cfg.d_model)

        # [4] EMA state stored as a non-parameter buffer.
        self.register_buffer("ema_score", torch.zeros(1))

    def _mod_wrap_attn(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor],
        ext_kv: Optional[Tuple],
    ) -> Tuple[torch.Tensor, Optional[Tuple], torch.Tensor]:
        """Wrap attention with MoD gather/attend/scatter and return (out, kv, router_score)."""
        _, seq_len, _ = x.shape
        sel, rest, top_idx, rest_idx, router_score = self.router(x)

        # NOTE: sel has sequence length k < T, so this uses the first k RoPE positions instead of
        # the original positions from top_idx. That mismatch is intentionally left here as a second test bug.
        top_k = sel.shape[1]
        sel_out, kv_out = self.attn(sel, cos[:top_k], sin[:top_k], mask=None, ext_kv=ext_kv)

        # EMA update is detached: it only affects next-layer scale injection and should not backpropagate.
        # NOTE: router_score.mean() is local under TP sequence parallelism, so ranks can diverge.
        with torch.no_grad():
            self.ema_score.mul_(self.cfg.router_ema_decay).add_(
                router_score.detach().mean() * (1 - self.cfg.router_ema_decay)
            )

        out = scatter_back(sel_out, rest, top_idx, rest_idx, seq_len)
        return out, kv_out, router_score

    def _mod_wrap_ffn(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wrap the FFN with MoD gather/compute/scatter and return (out, router_score)."""
        _, seq_len, _ = x.shape
        sel, rest, top_idx, rest_idx, router_score = self.router(x)
        sel_out = self.ffn(sel)
        out = scatter_back(sel_out, rest, top_idx, rest_idx, seq_len)
        return out, router_score

    def forward(
        self,
        x_c: torch.Tensor,
        x_m: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ext_kv: Optional[Tuple] = None,
        inject_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple], torch.Tensor]:
        """
        Returns (x_c, x_m, kv_output, next_inject_scale).

        inject_scale has shape (B, T, 1) and broadcasts the previous layer's EMA score
        into this layer's RMSNorm.
        next_inject_scale is the current layer's EMA score forwarded to the next layer.

        NOTE: inject_scale connects adjacent layers in the graph. With activation checkpointing,
        recompute can rebuild inject_scale with values that differ from the original forward
        pass, for example because of dropout, which breaks gradient equivalence.
        """
        if self.is_even:
            # x_c -> attention, x_m -> FFN
            xc_norm = self.norm_c1(x_c, scale=inject_scale)
            xm_norm = self.norm_m1(x_m, scale=inject_scale)

            attn_out, kv_out, _ = self._mod_wrap_attn(xc_norm, cos, sin, mask, ext_kv)
            ffn_out, _ = self._mod_wrap_ffn(xm_norm)

            x_c = x_c + attn_out
            x_m = x_m + ffn_out
        else:
            # x_m -> attention using even-layer KV, x_c -> FFN
            xc_norm = self.norm_c1(x_c, scale=inject_scale)
            xm_norm = self.norm_m1(x_m, scale=inject_scale)

            attn_out, kv_out, _ = self._mod_wrap_attn(xm_norm, cos, sin, mask, ext_kv)
            ffn_out, _ = self._mod_wrap_ffn(xc_norm)

            x_m = x_m + attn_out
            x_c = x_c + ffn_out

        # Post-norm after the residual addition.
        x_c = self.norm_c2(x_c)
        x_m = self.norm_m2(x_m)

        # [3] Dual-stream fusion
        if self.cfg.dual_stream and hasattr(self, "fusion"):
            x_c, x_m = self.fusion(x_c, x_m)

        # [4] Build the injected scale for the next layer.
        # NOTE: expand would alias the ema_score buffer. clone() avoids in-place races when
        # multiple micro-batches overlap and later updates overwrite the shared buffer.
        next_scale = self.ema_score.view(1, 1, 1).expand(x_c.shape[0], x_c.shape[1], 1).clone()

        return x_c, x_m, kv_out, next_scale


# =============================================================================
# Full model
# =============================================================================


class MagicTransformer(nn.Module):
    """
    Computation graph summary for one forward pass:

      embed(ids) --+--> x_c ------------------------------------------------------------>
                   +--> meta_proj --> x_m ----------------------------------------------->
                                                                                         |
      +-------------------------------------------------------------------------+        |
      | layer[0] even                                                           |        |
      |  inject_scale=None                                                      |        |
      |  x_c -> norm -> router -> gather(k tokens) -> GQAttn -> scatter -> +x_c|        |
      |  x_m -> norm -> router -> gather(k tokens) -> SwiGLU -> scatter -> +x_m|        |
      |  DualStreamFusion(x_c, x_m) -> x_c', x_m'                               |        |
      |  ema_score -> next_scale                                                |        |
      |  returns: kv_out (K_raw, V)                                             |        |
      +-------------------------------------------------------------------------+        |
                kv_out ------------------------------------------------------------------+
                next_scale --------------------------------------------------------------+
      +-------------------------------------------------------------------------+        |
      | layer[1] odd                                                           |<-------+
      |  inject_scale = next_scale[0]  (cross-layer gradient path)             |
      |  x_m -> norm -> router -> gather -> GQAttn(ext_kv=kv_out[0]) -> ...    | <- KV tying
      |  x_c -> norm -> router -> gather -> SwiGLU -> scatter -> +x_c          |
      |  DualStreamFusion -> x_c', x_m'                                         |
      +-------------------------------------------------------------------------+
        ... alternating even/odd layers ...

      norm_c(x_c) + norm_m(x_m) -> cat -> out_fusion -> lm_head -> logits
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Dual-stream split: this projection seeds the meta stream.
        # NOTE: x_m must follow the same TP sharding strategy as x_c.
        self.meta_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

        self.layers = nn.ModuleList([TransformerBlock(cfg, idx) for idx in range(cfg.n_layers)])
        self.norm_c = RMSNorm(cfg.d_model)
        self.norm_m = RMSNorm(cfg.d_model)

        # Fuse the two streams back to d_model before the LM head.
        # NOTE: the 2 * d_model input also needs TP-specific handling, same as DualStreamFusion.
        self.out_fusion = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # Weight tying.

        # RoPE cache
        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_base)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        scale = std / math.sqrt(2 * self.cfg.n_layers)
        for name, param in self.named_parameters():
            if param.dim() < 2:
                continue
            if any(key in name for key in ("out_proj", "down_proj", "out_fusion")):
                nn.init.normal_(param, std=scale)
            else:
                nn.init.normal_(param, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        assert seq_len <= self.cfg.max_seq_len, f"seq_len {seq_len} > max_seq_len {self.cfg.max_seq_len}"

        emb = self.drop(self.embed(input_ids))  # (B, T, D)
        x_c = emb
        x_m = self.meta_proj(emb)  # Dual-stream split [3]

        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]

        prev_kv: Optional[Tuple] = None
        prev_scale: Optional[torch.Tensor] = None

        for layer_idx, layer in enumerate(self.layers):
            # [2] Odd layers reuse the KV produced by the previous even layer.
            use_ext_kv = prev_kv if layer_idx % 2 == 1 else None

            x_c, x_m, kv_out, next_scale = layer(
                x_c,
                x_m,
                cos,
                sin,
                mask=mask,
                ext_kv=use_ext_kv,
                inject_scale=prev_scale,  # [4] Cross-layer gradient path
            )

            if layer_idx % 2 == 0:
                prev_kv = kv_out  # [2] Cache even-layer KV.

            prev_scale = next_scale  # [4] Propagate EMA scale.

        x_c = self.norm_c(x_c)
        x_m = self.norm_m(x_m)

        # Concatenate the two streams before the final projection.
        # NOTE: TP must all-gather on d_model before cat, same as DualStreamFusion.
        x = self.out_fusion(torch.cat([x_c, x_m], dim=-1))
        return self.lm_head(x)  # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
    ) -> torch.Tensor:
        """
        Simple top-k sampling without a KV cache.
        NOTE: incremental inference for MoD + dual-stream mode needs extra state management,
        so this path is only meant for functional verification.
        NOTE: TP inference in production needs an all-gather across vocab shards before softmax.
        """
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.cfg.max_seq_len :]
            logits = self(ctx)[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                values, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids


# =============================================================================
# Quick verification
# =============================================================================


if __name__ == "__main__":
    cfg = TransformerConfig(
        vocab_size=1000,
        max_seq_len=32,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        n_layers=4,
        dropout=0.0,
        mod_top_k_ratio=0.5,
        dual_stream=True,
    )
    model = MagicTransformer(cfg)
    total = sum(param.numel() for param in model.parameters())
    print(f"Parameter count: {total:,}")

    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(input_ids)
    print(f"Input: {input_ids.shape} -> logits: {logits.shape}")
    assert logits.shape == (2, 16, cfg.vocab_size), "Unexpected logits shape"

    loss = logits.mean()
    loss.backward()
    print("forward + backward passed")

    out = model.generate(input_ids[:1], max_new_tokens=4)
    print(f"Generated length: {out.shape[1]}")
    print("generate passed")
