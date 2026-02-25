# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Cheese ViT model."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.siglip.modeling_siglip import SiglipMLP
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

logger = logging.get_logger(__name__)

try:
    from flash_attn import flash_attn_func

    _flash_attn_available = True
except ImportError:
    _flash_attn_available = False


class CheeseViTConfig(PretrainedConfig):
    r"""
    Configuration class for [`CheeseViTModel`].

    This keeps the same core ViT fields as TomatoViT while adding depth-bin RoPE controls.
    New depth-related parameters are initialized so the model can load RGB-only checkpoints
    and initially behave like the original 3D (T,H,W) RoPE.
    """

    model_type = "cheese_vit"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=448,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        layer_norm_type="layer_norm",
        attention_dropout=0.0,
        initializer_range=0.02,
        rope_theta=10000.0,
        use_head=True,
        depth_num_bins=64,
        depth_min=0.0,
        depth_max=1.0,
        depth_invalid_fill_value=None,
        depth_inject_axes="hw",
        depth_inject_ratio=0.25,
        depth_gate_max=0.2,
        depth_use_separate_freq=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.layer_norm_type = layer_norm_type
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.use_head = use_head

        # Depth-to-RoPE controls.
        self.depth_num_bins = depth_num_bins
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_invalid_fill_value = depth_invalid_fill_value
        self.depth_inject_axes = depth_inject_axes
        self.depth_inject_ratio = depth_inject_ratio
        self.depth_gate_max = depth_gate_max
        self.depth_use_separate_freq = depth_use_separate_freq


@dataclass
class CheeseViTModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    mask: Optional[torch.FloatTensor] = None


@dataclass
class CheeseViTModelOutputWithPooling(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    mask: Optional[torch.FloatTensor] = None


CHEESE_VIT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for generic methods.
"""

CHEESE_VIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            RGB pixel values.
        pixel_values_depth (`torch.FloatTensor` of shape `(batch_size, 1, height, width)`, *optional*):
            Depth map used to produce depth bins and inject a 4th coordinate into RoPE.
        depth_valid_masks (`torch.FloatTensor`, *optional*):
            Optional valid mask for depth input. Invalid regions are filled with a stable fallback depth.
        mask_ratio (`float`, *optional*, defaults to `0.0`):
            Random patch mask ratio.
        output_attentions (`bool`, *optional*):
            Whether to return attention tensors.
        output_hidden_states (`bool`, *optional*):
            Whether to return hidden states.
        return_dict (`bool`, *optional*):
            Whether to return a [`ModelOutput`] instead of a tuple.
"""


def get_norm_layer(config):
    if config.layer_norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


def rotate_half(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, freqs):
    # q, k: (B, H, L, D)
    # freqs: (B, L, D)
    cos = freqs.cos().unsqueeze(1).to(q.dtype)
    sin = freqs.sin().unsqueeze(1).to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DepthAwareVideoRotaryEmbedding(nn.Module):
    """
    4D RoPE built on top of 3D (T,H,W) 4:6:6 split.

    Depth is injected additively into selected axes with gated magnitude:
    alpha = gate_max * tanh(raw_alpha), raw_alpha initialized to 0.
    This makes resumed training from depth-free checkpoints stable by default.
    """

    def __init__(self, config: CheeseViTConfig):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary.")
        if head_dim % 16 != 0:
            raise ValueError("head_dim must be divisible by 16 for 4:6:6 split.")
        half = head_dim // 2
        if half % 16 != 0:
            raise ValueError("head_dim//2 must be divisible by 16 for 4:6:6 split.")

        unit = half // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit

        self.register_buffer(
            "inv_freq_t",
            1.0 / (base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_h",
            1.0 / (base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_w",
            1.0 / (base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)),
            persistent=False,
        )

        self.depth_use_separate_freq = config.depth_use_separate_freq
        depth_axes = (config.depth_inject_axes or "hw").lower()
        invalid_axes = set(depth_axes) - {"t", "h", "w"}
        if invalid_axes:
            raise ValueError(f"depth_inject_axes has invalid entries: {sorted(invalid_axes)}")
        self.depth_inject_axes = depth_axes

        ratio = float(config.depth_inject_ratio)
        self.depth_inject_ratio = min(max(ratio, 0.0), 1.0)
        self.depth_gate_max = float(config.depth_gate_max)

        self.depth_scale = nn.Parameter(torch.ones(()))
        self.raw_alpha_t = nn.Parameter(torch.zeros(()))
        self.raw_alpha_h = nn.Parameter(torch.zeros(()))
        self.raw_alpha_w = nn.Parameter(torch.zeros(()))

        if self.depth_use_separate_freq:
            self.register_buffer(
                "inv_freq_d_t",
                1.0 / (base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)),
                persistent=False,
            )
            self.register_buffer(
                "inv_freq_d_h",
                1.0 / (base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)),
                persistent=False,
            )
            self.register_buffer(
                "inv_freq_d_w",
                1.0 / (base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)),
                persistent=False,
            )

        self.t_inject_dims = self._resolve_inject_dims(self.t_size)
        self.h_inject_dims = self._resolve_inject_dims(self.h_size)
        self.w_inject_dims = self._resolve_inject_dims(self.w_size)

    def _resolve_inject_dims(self, axis_size: int) -> int:
        if self.depth_inject_ratio <= 0.0:
            return 0
        inject_dims = int(round(axis_size * self.depth_inject_ratio))
        inject_dims = max(inject_dims, 1)
        return min(inject_dims, axis_size)

    def _depth_alpha(self, raw_alpha: torch.Tensor) -> torch.Tensor:
        return self.depth_gate_max * torch.tanh(raw_alpha)

    def _axis_inv_freq(self, axis: str, device: torch.device) -> torch.Tensor:
        if self.depth_use_separate_freq:
            if axis == "t":
                return self.inv_freq_d_t.to(device=device)
            if axis == "h":
                return self.inv_freq_d_h.to(device=device)
            return self.inv_freq_d_w.to(device=device)
        if axis == "t":
            return self.inv_freq_t.to(device=device)
        if axis == "h":
            return self.inv_freq_h.to(device=device)
        return self.inv_freq_w.to(device=device)

    @staticmethod
    def _inject_depth(base_phase, depth_phase, alpha, inject_dims):
        if inject_dims <= 0:
            return base_phase
        if inject_dims >= base_phase.shape[-1]:
            return base_phase + alpha * depth_phase
        return torch.cat(
            [
                base_phase[..., :inject_dims] + alpha * depth_phase[..., :inject_dims],
                base_phase[..., inject_dims:],
            ],
            dim=-1,
        )

    def forward(self, t: int, h: int, w: int, device=None) -> torch.Tensor:
        if device is None:
            device = self.inv_freq_t.device

        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        ft = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv_t)
        fh = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)
        fw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)

        t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
        h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
        w_ids = torch.arange(w, device=device).repeat(h).repeat(t)

        return torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)

    def forward_from_positions(self, patch_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_positions: [B, L, 3] -> (t,h,w) or [B, L, 4] -> (t,h,w,depth_bin_position)
        Returns:
            [B, L, head_dim//2]
        """
        if patch_positions.size(-1) not in (3, 4):
            raise ValueError("patch_positions last dim must be 3 or 4.")

        device = patch_positions.device

        t_pos = patch_positions[..., 0].float()
        h_pos = patch_positions[..., 1].float()
        w_pos = patch_positions[..., 2].float()

        ft = torch.einsum("bl,d->bld", t_pos, self.inv_freq_t.to(device=device))
        fh = torch.einsum("bl,d->bld", h_pos, self.inv_freq_h.to(device=device))
        fw = torch.einsum("bl,d->bld", w_pos, self.inv_freq_w.to(device=device))

        if patch_positions.size(-1) == 4:
            d_pos = patch_positions[..., 3].float() * self.depth_scale

            if "t" in self.depth_inject_axes:
                fd_t = torch.einsum("bl,d->bld", d_pos, self._axis_inv_freq("t", device))
                alpha_t = self._depth_alpha(self.raw_alpha_t)
                ft = self._inject_depth(ft, fd_t, alpha_t, self.t_inject_dims)

            if "h" in self.depth_inject_axes:
                fd_h = torch.einsum("bl,d->bld", d_pos, self._axis_inv_freq("h", device))
                alpha_h = self._depth_alpha(self.raw_alpha_h)
                fh = self._inject_depth(fh, fd_h, alpha_h, self.h_inject_dims)

            if "w" in self.depth_inject_axes:
                fd_w = torch.einsum("bl,d->bld", d_pos, self._axis_inv_freq("w", device))
                alpha_w = self._depth_alpha(self.raw_alpha_w)
                fw = self._inject_depth(fw, fd_w, alpha_w, self.w_inject_dims)

        return torch.cat([ft, fh, fw], dim=-1)


class MultiheadAttentionPoolingHead(nn.Module):
    """
    Multi-Head Attention Pooling with a learned probe (PMA-style).
    """

    def __init__(self, config: CheeseViTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)
        attn_output, _ = self.attention(probe, hidden_states, hidden_states)
        residual = attn_output
        attn_output = self.norm(attn_output)
        attn_output = residual + self.mlp(attn_output)
        return attn_output[:, 0]


class CheeseViTEmbeddings(nn.Module):
    def __init__(self, config: CheeseViTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(2)  # (B, C, 1, H, W)

        batch_size, channels, t_frames, height, width = pixel_values.shape
        x_2d = pixel_values.permute(0, 2, 1, 3, 4).reshape(batch_size * t_frames, channels, height, width)
        embeddings = self.patch_embedding(x_2d)  # (B*T, C, Hp, Wp)
        embeddings = embeddings.flatten(2).transpose(1, 2)  # (B*T, L_frame, C)

        total_patches = t_frames * (height // self.patch_size) * (width // self.patch_size)
        embeddings = embeddings.reshape(batch_size, total_patches, self.embed_dim)
        return embeddings


class CheeseViTAttention(nn.Module):
    def __init__(self, config: CheeseViTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim={self.embed_dim}, num_heads={self.num_heads})."
            )

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if rotary_pos_emb is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, rotary_pos_emb)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, q_len):
                if attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, q_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights if output_attentions else None


class CheeseViTFlashAttention2(nn.Module):
    def __init__(self, config: CheeseViTConfig):
        super().__init__()
        if not _flash_attn_available:
            raise ImportError("flash_attn is not installed. Please install it to use flash_attention_2.")

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim={self.embed_dim}, num_heads={self.num_heads})."
            )

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del attention_mask, output_attentions

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim)

        if rotary_pos_emb is not None:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, rotary_pos_emb)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        )
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None


CHEESEVIT_ATTENTION_CLASSES = {
    "eager": CheeseViTAttention,
    "flash_attention_2": CheeseViTFlashAttention2,
}


class CheeseViTEncoderLayer(nn.Module):
    def __init__(self, config: CheeseViTConfig):
        super().__init__()
        attn_implementation = getattr(config, "_attn_implementation", "flash_attention_2")
        if attn_implementation not in CHEESEVIT_ATTENTION_CLASSES:
            raise ValueError(
                f"Unknown attention implementation: {attn_implementation}. "
                f"Available implementations: {list(CHEESEVIT_ATTENTION_CLASSES.keys())}"
            )
        if attn_implementation == "flash_attention_2" and not _flash_attn_available:
            logger.warning("flash_attn is not available; falling back to eager attention.")
            attn_implementation = "eager"

        self.self_attn = CHEESEVIT_ATTENTION_CLASSES[attn_implementation](config)
        self.layer_norm1 = get_norm_layer(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = get_norm_layer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if output_attentions:
            return hidden_states, attn_weights
        return (hidden_states,)


class CheeseViTEncoder(nn.Module):
    def __init__(self, config: CheeseViTConfig):
        super().__init__()
        self.layers = nn.ModuleList([CheeseViTEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return CheeseViTModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@add_start_docstrings("The bare Cheese ViT Model outputting raw hidden-states.", CHEESE_VIT_START_DOCSTRING)
class CheeseViTPreTrainedModel(PreTrainedModel):
    config_class = CheeseViTConfig
    base_model_prefix = "cheese_vit"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CheeseViTEncoderLayer", "MultiheadAttentionPoolingHead"]
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()


@add_start_docstrings("Cheese ViT Model with depth-bin 4D RoPE.", CHEESE_VIT_START_DOCSTRING)
class CheeseViTModel(CheeseViTPreTrainedModel):
    def __init__(self, config: CheeseViTConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = CheeseViTEmbeddings(config)
        self.mask_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.layernorm_pre = get_norm_layer(config)
        self.encoder = CheeseViTEncoder(config)
        self.video_rope = DepthAwareVideoRotaryEmbedding(config)

        if config.use_head:
            self.layernorm_post = get_norm_layer(config)
            self.head = MultiheadAttentionPoolingHead(config)
        else:
            self.layernorm_post = None
            self.head = None

        self.post_init()

    def _normalize_depth_to_bins(
        self,
        pixel_values_depth: Optional[torch.Tensor],
        depth_valid_masks: Optional[torch.Tensor],
        t_frames: int,
        height: int,
        width: int,
    ) -> Optional[torch.Tensor]:
        if pixel_values_depth is None:
            return None

        if pixel_values_depth.dim() == 4:
            depth = pixel_values_depth.unsqueeze(2)  # (B, C, 1, H, W)
        elif pixel_values_depth.dim() == 5:
            depth = pixel_values_depth
        else:
            raise ValueError("pixel_values_depth must be 4D or 5D.")

        batch_size = depth.shape[0]
        if depth.shape[1] > 1:
            depth = depth[:, :1]

        if depth.shape[2] != t_frames:
            if depth.shape[2] == 1:
                depth = depth.repeat(1, 1, t_frames, 1, 1)
            else:
                raise ValueError("Depth temporal dimension does not match RGB temporal dimension.")

        if depth.shape[-2:] != (height, width):
            depth_2d = depth.permute(0, 2, 1, 3, 4).reshape(batch_size * t_frames, 1, depth.shape[-2], depth.shape[-1])
            depth_2d = F.interpolate(depth_2d, size=(height, width), mode="bilinear", align_corners=False)
            depth = depth_2d.reshape(batch_size, t_frames, 1, height, width).permute(0, 2, 1, 3, 4)

        valid_mask = torch.isfinite(depth)
        if depth_valid_masks is not None:
            if depth_valid_masks.dim() == 4:
                depth_valid_masks = depth_valid_masks.unsqueeze(2)
            elif depth_valid_masks.dim() != 5:
                raise ValueError("depth_valid_masks must be 4D or 5D when provided.")

            if depth_valid_masks.shape[1] > 1:
                depth_valid_masks = depth_valid_masks[:, :1]

            if depth_valid_masks.shape[2] != t_frames:
                if depth_valid_masks.shape[2] == 1:
                    depth_valid_masks = depth_valid_masks.repeat(1, 1, t_frames, 1, 1)
                else:
                    raise ValueError("Depth valid mask temporal dimension does not match RGB temporal dimension.")

            if depth_valid_masks.shape[-2:] != (height, width):
                mask_2d = depth_valid_masks.permute(0, 2, 1, 3, 4).reshape(
                    batch_size * t_frames, 1, depth_valid_masks.shape[-2], depth_valid_masks.shape[-1]
                )
                mask_2d = F.interpolate(mask_2d, size=(height, width), mode="nearest")
                depth_valid_masks = mask_2d.reshape(batch_size, t_frames, 1, height, width).permute(0, 2, 1, 3, 4)

            valid_mask = valid_mask & (depth_valid_masks > 0.5)

        depth_min = float(self.config.depth_min)
        depth_max = float(self.config.depth_max)
        if depth_max <= depth_min:
            raise ValueError("depth_max must be greater than depth_min.")

        fill_value = self.config.depth_invalid_fill_value
        if fill_value is None:
            fill_value = depth_max
        fill_value = min(max(float(fill_value), depth_min), depth_max)

        depth = torch.nan_to_num(depth, nan=fill_value, posinf=fill_value, neginf=depth_min)
        depth = torch.clamp(depth, min=depth_min, max=depth_max)
        depth = torch.where(valid_mask, depth, torch.full_like(depth, fill_value))

        depth = (depth - depth_min) / (depth_max - depth_min + 1e-6)
        depth = torch.clamp(depth, 0.0, 1.0)

        depth_2d = depth.permute(0, 2, 1, 3, 4).reshape(batch_size * t_frames, 1, height, width)
        depth_pooled = F.avg_pool2d(depth_2d, kernel_size=self.config.patch_size, stride=self.config.patch_size)
        patch_h, patch_w = depth_pooled.shape[-2], depth_pooled.shape[-1]
        depth_pooled = depth_pooled.reshape(batch_size, t_frames, patch_h, patch_w)

        num_bins = max(int(self.config.depth_num_bins), 2)
        depth_bins = torch.round(depth_pooled * (num_bins - 1)).clamp(0, num_bins - 1).to(torch.long)
        return depth_bins

    def _build_patch_positions(
        self,
        batch_size: int,
        t_frames: int,
        patch_h: int,
        patch_w: int,
        depth_bins: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        t_ids = torch.arange(t_frames, device=device).repeat_interleave(patch_h * patch_w)
        h_ids = torch.arange(patch_h, device=device).repeat_interleave(patch_w).repeat(t_frames)
        w_ids = torch.arange(patch_w, device=device).repeat(patch_h).repeat(t_frames)

        t_pos = t_ids.unsqueeze(0).expand(batch_size, -1).float()
        h_pos = h_ids.unsqueeze(0).expand(batch_size, -1).float()
        w_pos = w_ids.unsqueeze(0).expand(batch_size, -1).float()

        if depth_bins is None:
            return torch.stack([t_pos, h_pos, w_pos], dim=-1)

        depth_tokens = depth_bins.reshape(batch_size, -1).to(device=device, dtype=torch.float32)
        num_bins = max(int(self.config.depth_num_bins), 2)
        depth_tokens = depth_tokens / float(num_bins - 1)
        spatial_scale = float(max(max(patch_h, patch_w) - 1, 1))
        depth_tokens = depth_tokens * spatial_scale
        return torch.stack([t_pos, h_pos, w_pos, depth_tokens], dim=-1)

    @add_start_docstrings_to_model_forward(CHEESE_VIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CheeseViTConfig)
    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_values_depth: Optional[torch.Tensor] = None,
        depth_valid_masks: Optional[torch.Tensor] = None,
        mask_ratio: Optional[float] = 0.0,
        mask_rgb: Optional[bool] = True,
        mask_depth: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
            [`BaseModelOutputWithPooling`] or `tuple`: Model outputs with optional pooled representation.
        """
        del mask_depth

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values.dim() == 4:
            t_frames = 1
            height, width = pixel_values.shape[2], pixel_values.shape[3]
        elif pixel_values.dim() == 5:
            t_frames = pixel_values.shape[2]
            height, width = pixel_values.shape[3], pixel_values.shape[4]
        else:
            raise ValueError("pixel_values must be 4D (B,C,H,W) or 5D (B,C,T,H,W).")

        hidden_states = self.embeddings(pixel_values)
        batch_size, total_patches, _ = hidden_states.shape

        if mask_ratio > 0.0:
            num_masked = int(total_patches * mask_ratio)
            mask_indices = torch.rand(batch_size, total_patches, device=hidden_states.device).argsort(dim=-1)[
                :, :num_masked
            ]
            if mask_rgb and self.mask_embedding is not None:
                hidden_states[torch.arange(batch_size).unsqueeze(1), mask_indices] = self.mask_embedding.to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
        else:
            mask_indices = None

        patch_h = height // self.config.patch_size
        patch_w = width // self.config.patch_size

        depth_bins = self._normalize_depth_to_bins(
            pixel_values_depth=pixel_values_depth,
            depth_valid_masks=depth_valid_masks,
            t_frames=t_frames,
            height=height,
            width=width,
        )
        patch_positions = self._build_patch_positions(
            batch_size=batch_size,
            t_frames=t_frames,
            patch_h=patch_h,
            patch_w=patch_w,
            depth_bins=depth_bins,
            device=pixel_values.device,
        )

        if patch_positions.shape[1] != total_patches:
            raise ValueError(
                f"Patch position length mismatch: got {patch_positions.shape[1]}, expected {total_patches}. "
                "Check image size and patch size consistency."
            )

        freqs_visible = self.video_rope.forward_from_positions(patch_positions)
        freqs_visible = torch.cat([freqs_visible, freqs_visible], dim=-1)

        hidden_states = self.layernorm_pre(hidden_states)
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=freqs_visible,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state

        if self.layernorm_post is not None:
            sequence_output = self.layernorm_post(sequence_output)

        pooled_output = self.head(sequence_output) if self.head is not None else None

        if not return_dict:
            return (sequence_output, pooled_output, mask_indices) + encoder_outputs[1:]

        return CheeseViTModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            mask=mask_indices,
        )


def cheese_vit_l(pretrained: bool = False, ckpt_path=None, **kwargs):
    del pretrained, ckpt_path

    config_kwargs = {
        "patch_size": 14,
        "hidden_size": 1024,
        "num_attention_heads": 1024 // 64,
        "num_hidden_layers": 24,
        "intermediate_size": 4096,
        "hidden_act": "gelu",
        "layer_norm_type": "layer_norm",
        "use_head": True,
        "depth_num_bins": 64,
        "depth_inject_axes": "hw",
    }
    config_kwargs.update(kwargs)
    config = CheeseViTConfig(**config_kwargs)
    config._attn_implementation = "flash_attention_2"
    return CheeseViTModel(config)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = cheese_vit_l().eval()
    test_rgb = torch.randn(2, 3, 224, 224)
    test_depth = torch.rand(2, 1, 224, 224)
    output = model(test_rgb, pixel_values_depth=test_depth, mask_ratio=0.25)
    print(output.last_hidden_state.shape, output.pooler_output.shape if output.pooler_output is not None else None)
