"""Recipe-local adapter around the Magic Transformer source model."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from . import source_model

TransformerConfig = source_model.TransformerConfig
apply_rope = source_model.apply_rope


class GQAttention(source_model.GQAttention):
    """Patch KV reuse so the source model can train correctly inside the recipe."""

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ext_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if ext_kv is None:
            k_raw = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v_raw = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        else:
            k_raw, v_raw = ext_kv

        q = apply_rope(q, cos, sin)
        k = apply_rope(k_raw, cos, sin)
        v = v_raw

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
        return self.out_proj(out), (k_raw, v_raw)


class MagicTransformer(source_model.MagicTransformer):
    """Source model with the recipe-local attention fix applied."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__(cfg)
        for layer in self.layers:
            patched_attention = GQAttention(cfg)
            patched_attention.load_state_dict(layer.attn.state_dict())
            layer.attn = patched_attention


__all__ = [
    "MagicTransformer",
    "TransformerConfig",
]
