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
"""PyTorch Tomato ViT model."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from flash_attn import flash_attn_func
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
    is_flash_attn_2_available,
    logging,
    replace_return_docstrings,
)

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration Class
# ---------------------------------------------------------------------------


class TomatoViTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TomatoViTModel`]. It is used to instantiate a
    Tomato ViT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Tomato ViT architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        layer_norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The type of layer normalization to use. Supported values: `"layer_norm"`, `"rms_norm"`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        use_head (`bool`, *optional*, defaults to `True`):
            Whether to use the pooling head.
        mot_layers (`list[int]`, *optional*, defaults to `None`):
            List of layer indices for the mixture of transformer layers. If None, no layers are treated as MoT layers.


    Example:

    ```python
    >>> from transformers import TomatoViTConfig, TomatoViTModel

    >>> # Initializing a TomatoViT configuration
    >>> configuration = TomatoViTConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = TomatoViTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "tomato_vit"

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
        mot_layers=None,
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
        self.mot_layers = mot_layers if mot_layers is not None else []


@dataclass
class TomatoViTModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    last_hidden_state_depth: Optional[torch.FloatTensor] = None

    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    hidden_states_depth: Optional[Tuple[torch.FloatTensor]] = None

    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions_depth: Optional[tuple[torch.FloatTensor]] = None

    mask: Optional[torch.FloatTensor] = None


@dataclass
class TomatoViTModelOutputWithPooling(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    pooler_output: Optional[torch.FloatTensor] = None
    last_hidden_state_depth: Optional[torch.FloatTensor] = None
    pooler_output_depth: Optional[torch.FloatTensor] = None

    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    hidden_states_depth: Optional[tuple[torch.FloatTensor]] = None

    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions_depth: Optional[tuple[torch.FloatTensor]] = None

    mask: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# Model Docstrings
# ---------------------------------------------------------------------------

TOMATO_VIT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TomatoViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TOMATO_VIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`].
        pixel_values_depth (`torch.FloatTensor` of shape `(batch_size, 1, height, width)`, *optional*):
            Depth pixel values. Depth pixel values can be obtained using [`AutoImageProcessor`].
        visible_indices (`torch.Tensor`, *optional*):
            Indices of visible patches for masking. Used in MAE-style pretraining or inference.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def get_norm_layer(config):
    if config.layer_norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    else:
        return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


def rotate_half(x):
    """
    Interleaved rotation to match Source model's implementation.
    (x1, x2, x3, x4) -> (-x2, x1, -x4, x3)
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, freqs):
    # q, k: (B, H, L, D)
    # freqs: (B, L, D)

    # We need to broadcast freqs to match heads
    # (B, L, D) -> (B, 1, L, D)

    # !!! CRITICAL FIX: Cast cos/sin to q.dtype (bf16/fp16) immediately
    # freqs are typically float32, so cos() returns float32.
    # Without this cast, (q * cos) upcasts q to float32, causing FlashAttention to fail.
    cos = freqs.cos().unsqueeze(1).to(q.dtype)
    sin = freqs.sin().unsqueeze(1).to(q.dtype)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Modeling Components
# ---------------------------------------------------------------------------


class VideoRotaryEmbedding(nn.Module):
    """
    3D (T,H,W) Rotary frequency constructor with 4:6:6 split.
    """

    def __init__(self, config: TomatoViTConfig):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta

        assert head_dim % 2 == 0, "head_dim must be even for rotary."
        assert head_dim % 16 == 0, "head_dim must be divisible by 16."
        half = head_dim // 2
        assert half % 16 == 0, "head_dim//2 must also be divisible by 16 to split into 4:6:6."

        self.head_dim = head_dim
        self.half = half

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

    def forward(self, t: int, h: int, w: int, device=None):
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

        freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)
        return freqs


class MultiheadAttentionPoolingHead(nn.Module):
    """
    Multi-Head Attention Pooling with a learned probe (PMA-style).
    """

    def __init__(self, config: TomatoViTConfig):
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


class TomatoViTEmbeddings(nn.Module):
    def __init__(self, config: TomatoViTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # Handle 4D (B, C, H, W) or 5D (B, C, T, H, W) inputs
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(2)  # (B, C, 1, H, W)

        batch_size, channels, t_frames, height, width = pixel_values.shape

        # Merge time into batch for Conv2d
        x_2d = pixel_values.permute(0, 2, 1, 3, 4).reshape(batch_size * t_frames, channels, height, width)

        # Patch Embed
        embeddings = self.patch_embedding(x_2d)  # (B*T, C, Hp, Wp)
        embeddings = embeddings.flatten(2).transpose(1, 2)  # (B*T, L_frame, C)

        # Flatten all patches
        total_patches = t_frames * (height // self.patch_size) * (width // self.patch_size)
        embeddings = embeddings.reshape(batch_size, total_patches, self.embed_dim)

        return embeddings


class TomatoViTAttention(nn.Module):
    """Multi-headed attention with RoPE support"""

    def __init__(self, config: TomatoViTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
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

        local_embed_dim = query_states.size(-1)
        if local_embed_dim % self.head_dim != 0:
            raise ValueError(
                "Local embed dim must be divisible by head_dim "
                f"(got local_embed_dim={local_embed_dim}, head_dim={self.head_dim})."
            )
        num_heads = local_embed_dim // self.head_dim

        # (B, L, H, D) -> Transpose to (B, H, L, D)
        query_states = query_states.view(batch_size, q_len, num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, num_heads, self.head_dim).transpose(1, 2)

        if rotary_pos_emb is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, rotary_pos_emb)

        # Calculate attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, q_len):
                if attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
            attn_weights = attn_weights + attention_mask

        # FIX: Remove dtype=torch.float32 to stay in original dtype (bf16/fp16)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, local_embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None


class TomatoViTFlashAttention2(nn.Module):
    """
    Multi-headed attention with RoPE support using Flash Attention 2.
    This module implements the same attention mechanism as TomatoViTAttention but uses
    Flash Attention for improved performance and memory efficiency.
    """

    def __init__(self, config: TomatoViTConfig):
        super().__init__()

        if not is_flash_attn_2_available():
            raise ImportError("Flash Attention 2 is not available. Please install it to use TomatoViTFlashAttention2.")

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
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
        """
        Forward pass using Flash Attention 2.
        """
        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash Attention requires (B, L, H, D) format
        local_embed_dim = query_states.size(-1)
        if local_embed_dim % self.head_dim != 0:
            raise ValueError(
                "Local embed dim must be divisible by head_dim "
                f"(got local_embed_dim={local_embed_dim}, head_dim={self.head_dim})."
            )
        num_heads = local_embed_dim // self.head_dim
        query_states = query_states.view(batch_size, q_len, num_heads, self.head_dim)
        key_states = key_states.view(batch_size, q_len, num_heads, self.head_dim)
        value_states = value_states.view(batch_size, q_len, num_heads, self.head_dim)

        # Apply RoPE if provided
        if rotary_pos_emb is not None:
            # Transpose for RoPE application: (B, L, H, D) -> (B, H, L, D)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            # NOTE: apply_rotary_pos_emb now ensures NO float32 cast happens
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, rotary_pos_emb)
            # Transpose back: (B, H, L, D) -> (B, L, H, D)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

        # FIX: Removed the explicit float32 check and downcast.
        # We assume input is already correct (bf16/fp16) thanks to RoPE fix.

        # Flash Attention forward pass
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        )

        # Reshape to (B, L, embed_dim)
        attn_output = attn_output.reshape(batch_size, q_len, local_embed_dim)

        # No extra casting here.
        attn_output = self.out_proj(attn_output)

        return attn_output, None


class TomatoViTMoTFlashAttention2(nn.Module):
    """
    Mixture of Transformers (MoT) Flash Attention for joint RGB+Depth attention.

    In MoT, tokens from both modalities (RGB and Depth) attend to ALL tokens
    from both modalities jointly, enabling cross-modal information exchange.
    Each modality has its own Q/K/V projections and output projections.
    """

    def __init__(self, config: TomatoViTConfig):
        super().__init__()

        if not is_flash_attn_2_available():
            raise ImportError(
                "Flash Attention 2 is not available. Please install it to use TomatoViTMoTFlashAttention2."
            )

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        # Branch A modality projections
        self.q_proj_a = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj_a = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj_a = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj_a = nn.Linear(self.embed_dim, self.embed_dim)

        # Branch B modality projections
        self.q_proj_b = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj_b = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj_b = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj_b = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states_a: torch.Tensor,
        hidden_states_b: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb_a: Optional[torch.Tensor] = None,
        rotary_pos_emb_b: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for MoT attention.

        Args:
            hidden_states_a: Branch A hidden states (B, L_a, D)
            hidden_states_b: Branch B hidden states (B, L_b, D)
            attention_mask: Optional attention mask
            rotary_pos_emb_a: RoPE for branch A tokens
            rotary_pos_emb_b: RoPE for branch B tokens
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output_a, output_b, optional_attention_weights)
        """
        batch_size, seq_len_a, _ = hidden_states_a.size()
        _, seq_len_b, _ = hidden_states_b.size()

        # Compute Q, K, V for branch A
        q_a = self.q_proj_a(hidden_states_a)
        k_a = self.k_proj_a(hidden_states_a)
        v_a = self.v_proj_a(hidden_states_a)

        # Compute Q, K, V for branch B
        q_b = self.q_proj_b(hidden_states_b)
        k_b = self.k_proj_b(hidden_states_b)
        v_b = self.v_proj_b(hidden_states_b)

        # Reshape for attention: (B, L, H, D)
        local_embed_dim_a = q_a.size(-1)
        local_embed_dim_b = q_b.size(-1)
        if local_embed_dim_a != local_embed_dim_b:
            raise ValueError(
                f"Local embed dims for branches must match (got a={local_embed_dim_a}, b={local_embed_dim_b})."
            )
        if local_embed_dim_a % self.head_dim != 0:
            raise ValueError(
                "Local embed dim must be divisible by head_dim "
                f"(got local_embed_dim={local_embed_dim_a}, head_dim={self.head_dim})."
            )
        num_heads = local_embed_dim_a // self.head_dim

        q_a = q_a.view(batch_size, seq_len_a, num_heads, self.head_dim)
        k_a = k_a.view(batch_size, seq_len_a, num_heads, self.head_dim)
        v_a = v_a.view(batch_size, seq_len_a, num_heads, self.head_dim)

        q_b = q_b.view(batch_size, seq_len_b, num_heads, self.head_dim)
        k_b = k_b.view(batch_size, seq_len_b, num_heads, self.head_dim)
        v_b = v_b.view(batch_size, seq_len_b, num_heads, self.head_dim)

        # Apply RoPE if provided
        if rotary_pos_emb_a is not None:
            # Transpose for RoPE: (B, L, H, D) -> (B, H, L, D)
            q_a = q_a.transpose(1, 2)
            k_a = k_a.transpose(1, 2)
            q_a, k_a = apply_rotary_pos_emb(q_a, k_a, rotary_pos_emb_a)
            q_a = q_a.transpose(1, 2)
            k_a = k_a.transpose(1, 2)

        if rotary_pos_emb_b is not None:
            q_b = q_b.transpose(1, 2)
            k_b = k_b.transpose(1, 2)
            q_b, k_b = apply_rotary_pos_emb(q_b, k_b, rotary_pos_emb_b)
            q_b = q_b.transpose(1, 2)
            k_b = k_b.transpose(1, 2)

        # Concatenate branches for joint attention
        # Q: each branch queries all tokens
        # K, V: concatenate both branches
        k_joint = torch.cat([k_a, k_b], dim=1)  # (B, L_a + L_b, H, D)
        v_joint = torch.cat([v_a, v_b], dim=1)  # (B, L_a + L_b, H, D)

        # Branch A tokens attend to all tokens
        attn_output_a = flash_attn_func(
            q_a,
            k_joint,
            v_joint,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        )

        # Branch B tokens attend to all tokens
        attn_output_b = flash_attn_func(
            q_b,
            k_joint,
            v_joint,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        )

        # Reshape outputs: (B, L, H, D) -> (B, L, embed_dim)
        attn_output_a = attn_output_a.reshape(batch_size, seq_len_a, local_embed_dim_a)
        attn_output_b = attn_output_b.reshape(batch_size, seq_len_b, local_embed_dim_b)

        # Apply output projections (branch-specific)
        attn_output_a = self.out_proj_a(attn_output_a)
        attn_output_b = self.out_proj_b(attn_output_b)

        return attn_output_a, None, attn_output_b, None


TOMATOVIT_ATTENTION_CLASSES = {
    "flash_attention_2": TomatoViTFlashAttention2,
}

TOMATOVIT_MOT_ATTENTION_CLASSES = {
    "flash_attention_2": TomatoViTMoTFlashAttention2,
}


class TomatoViTEncoderLayer(nn.Module):
    def __init__(self, config: TomatoViTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        # Get attention implementation from config, default to "flash_attention_2"
        attn_implementation = getattr(config, "_attn_implementation", "flash_attention_2")
        if attn_implementation not in TOMATOVIT_ATTENTION_CLASSES:
            raise ValueError(
                f"Unknown attention implementation: {attn_implementation}. "
                f"Available implementations: {list(TOMATOVIT_ATTENTION_CLASSES.keys())}"
            )
        self.self_attn = TOMATOVIT_ATTENTION_CLASSES[attn_implementation](config)
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

        outputs = (hidden_states, attn_weights) if output_attentions else (hidden_states,)
        return outputs


class TomatoViTIdentityEncoderLayer(nn.Module):
    def __init__(self, config: TomatoViTConfig):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs = (hidden_states, None) if output_attentions else (hidden_states,)
        return outputs


class TomatoViTMixtureEncoderLayer(nn.Module):
    """
    Mixture of Transformers (MoT) Encoder Layer.

    In MoT, two branches (A and B) share attention computation (attending to
    all tokens from both branches), but have separate layer norms and FFN/MLP
    for branch-specific processing.
    """

    def __init__(self, config: TomatoViTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size

        # Get attention implementation from config, default to "flash_attention_2"
        attn_implementation = getattr(config, "_attn_implementation", "flash_attention_2")
        if attn_implementation not in TOMATOVIT_MOT_ATTENTION_CLASSES:
            raise ValueError(
                f"Unknown MoT attention implementation: {attn_implementation}. "
                f"Available implementations: {list(TOMATOVIT_MOT_ATTENTION_CLASSES.keys())}"
            )

        # Joint attention for both branches
        self.self_attn = TOMATOVIT_MOT_ATTENTION_CLASSES[attn_implementation](config)

        # Separate pre-attention layer norms for each branch
        self.layer_norm1_a = get_norm_layer(config)
        self.layer_norm1_b = get_norm_layer(config)

        # Separate FFN/MLP for each branch
        self.mlp_a = SiglipMLP(config)
        self.mlp_b = SiglipMLP(config)

        # Separate post-attention layer norms for each branch
        self.layer_norm2_a = get_norm_layer(config)
        self.layer_norm2_b = get_norm_layer(config)

    def forward(
        self,
        hidden_states_a: torch.Tensor,
        hidden_states_b: torch.Tensor,
        attention_mask_a: Optional[torch.Tensor] = None,
        attention_mask_b: Optional[torch.Tensor] = None,
        rotary_pos_emb_a: Optional[torch.Tensor] = None,
        rotary_pos_emb_b: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for MoT encoder layer.

        Args:
            hidden_states_a: Branch A hidden states (B, L_a, D)
            hidden_states_b: Branch B hidden states (B, L_b, D)
            attention_mask_a: Optional attention mask for branch A
            attention_mask_b: Optional attention mask for branch B
            rotary_pos_emb_a: RoPE for branch A tokens
            rotary_pos_emb_b: RoPE for branch B tokens
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output_a, output_b, attn_weights_a, attn_weights_b)
        """
        # Store residuals
        residual_a = hidden_states_a
        residual_b = hidden_states_b

        # Pre-attention layer norm (branch-specific)
        hidden_states_a = self.layer_norm1_a(hidden_states_a)
        hidden_states_b = self.layer_norm1_b(hidden_states_b)

        # Joint MoT attention - both branches attend to all tokens
        attn_output_a, attn_weights_a, attn_output_b, attn_weights_b = self.self_attn(
            hidden_states_a=hidden_states_a,
            hidden_states_b=hidden_states_b,
            attention_mask=None,  # TODO: handle joint attention mask if needed
            rotary_pos_emb_a=rotary_pos_emb_a,
            rotary_pos_emb_b=rotary_pos_emb_b,
            output_attentions=output_attentions,
        )

        # Residual connection
        hidden_states_a = residual_a + attn_output_a
        hidden_states_b = residual_b + attn_output_b

        # Store residuals for FFN
        residual_a = hidden_states_a
        residual_b = hidden_states_b

        # Post-attention layer norm (branch-specific)
        hidden_states_a = self.layer_norm2_a(hidden_states_a)
        hidden_states_b = self.layer_norm2_b(hidden_states_b)

        # FFN/MLP (branch-specific)
        hidden_states_a = self.mlp_a(hidden_states_a)
        hidden_states_b = self.mlp_b(hidden_states_b)

        # Residual connection
        hidden_states_a = residual_a + hidden_states_a
        hidden_states_b = residual_b + hidden_states_b

        if output_attentions:
            return (hidden_states_a, hidden_states_b, attn_weights_a, attn_weights_b)
        return (hidden_states_a, hidden_states_b, None, None)


class TomatoViTEncoder(nn.Module):
    def __init__(self, config: TomatoViTConfig):
        super().__init__()
        self.config = config

        self.mixture_layers = nn.ModuleList([TomatoViTMixtureEncoderLayer(config) for _ in config.mot_layers])
        # With the help of TomatoViTIdentityEncoderLayer, we can get the correct layers with the layer indices.
        self.layers = nn.ModuleList(
            [
                TomatoViTEncoderLayer(config) if i not in config.mot_layers else TomatoViTIdentityEncoderLayer(config)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.layers_depth = nn.ModuleList(
            [
                TomatoViTEncoderLayer(config) if i not in config.mot_layers else TomatoViTIdentityEncoderLayer(config)
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_depth: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_depth: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_emb_depth: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states_depth = () if output_hidden_states else None
        all_self_attentions_depth = () if output_attentions else None

        for layer_i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                if hidden_states_depth is not None:
                    all_hidden_states_depth = all_hidden_states_depth + (hidden_states_depth,)

            if layer_i in self.config.mot_layers:
                # MoT layer: joint attention between RGB and Depth
                mixture_layer = self.mixture_layers[self.config.mot_layers.index(layer_i)]
                mixture_layer_outputs = mixture_layer(
                    hidden_states,
                    hidden_states_depth,
                    attention_mask,
                    attention_mask_depth,
                    rotary_pos_emb,
                    rotary_pos_emb_depth,
                    output_attentions,
                )
                hidden_states = mixture_layer_outputs[0]
                hidden_states_depth = mixture_layer_outputs[1]

                if output_attentions:
                    all_self_attentions = all_self_attentions + (mixture_layer_outputs[2],)
                    if hidden_states_depth is not None:
                        all_self_attentions_depth = all_self_attentions_depth + (mixture_layer_outputs[3],)
            else:
                # Regular layer: separate attention for RGB and Depth
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]

                if hidden_states_depth is not None:
                    # Process depth with the corresponding depth layer
                    layer_depth = self.layers_depth[layer_i]
                    layer_outputs_depth = layer_depth(
                        hidden_states_depth,
                        attention_mask=attention_mask_depth,
                        rotary_pos_emb=rotary_pos_emb_depth,
                        output_attentions=output_attentions,
                    )
                    hidden_states_depth = layer_outputs_depth[0]

                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    if hidden_states_depth is not None:
                        all_self_attentions_depth = all_self_attentions_depth + (layer_outputs_depth[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            if hidden_states_depth is not None:
                all_hidden_states_depth = all_hidden_states_depth + (hidden_states_depth,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_hidden_states_depth,
                    all_self_attentions_depth,
                ]
                if v is not None
            )

        return TomatoViTModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            last_hidden_state_depth=hidden_states_depth,
            hidden_states_depth=all_hidden_states_depth,
            attentions_depth=all_self_attentions_depth,
        )


# ---------------------------------------------------------------------------
# Main Models
# ---------------------------------------------------------------------------


@add_start_docstrings(
    "The bare Tomato ViT Model outputting raw hidden-states without any specific head on top.",
    TOMATO_VIT_START_DOCSTRING,
)
class TomatoViTPreTrainedModel(PreTrainedModel):
    config_class = TomatoViTConfig
    base_model_prefix = "tomato_vit"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "TomatoViTEncoderLayer",
        "TomatoViTMixtureEncoderLayer",
        "TomatoViTIdentityEncoderLayer",
        "MultiheadAttentionPoolingHead",
    ]
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        """Initialize the weights"""
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


@add_start_docstrings(
    "Tomato ViT Model with a vision transformer encoder.",
    TOMATO_VIT_START_DOCSTRING,
)
class TomatoViTModel(TomatoViTPreTrainedModel):
    def __init__(self, config: TomatoViTConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = TomatoViTEmbeddings(config)
        self.embeddings_depth = TomatoViTEmbeddings(config)

        self.mask_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))

        self.layernorm_pre = get_norm_layer(config)
        self.layernorm_pre_depth = get_norm_layer(config)

        self.encoder = TomatoViTEncoder(config)
        self.video_rope = VideoRotaryEmbedding(config)

        if config.use_head:
            self.layernorm_post = get_norm_layer(config)
            self.head = MultiheadAttentionPoolingHead(config)

            self.layernorm_post_depth = get_norm_layer(config)
            self.head_depth = MultiheadAttentionPoolingHead(config)
        else:
            self.layernorm_post = None
            self.head = None
            self.layernorm_post_depth = None
            self.head_depth = None

        self.post_init()

    @add_start_docstrings_to_model_forward(TOMATO_VIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=TomatoViTConfig)
    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_values_depth: Optional[torch.Tensor] = None,
        depth_valid_masks: Optional[torch.Tensor] = None,
        mask_ratio: Optional[float] = 0.0,
        mask_rgb: Optional[bool] = False,
        mask_depth: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
            [`BaseModelOutputWithPooling`] or `tuple`: A [`BaseModelOutputWithPooling`] object if
            `return_dict=True`. Otherwise, a tuple of tensors comprising various elements depending on the configuration
            and inputs:

            - **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
              Sequence of hidden-states at the output of the last layer of the model.
            - **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
              The output of the pooling head. Only present if the model is configured with a pooling head.

        Examples:

        ```python
        >>> from transformers import TomatoViTModel
        >>> import torch

        >>> model = TomatoViTModel.from_pretrained("mvp-ai-lab/tomato-vit")
        >>> pixel_values = torch.randn(1, 3, 224, 224)
        >>> pixel_values_depth = torch.randn(1, 1, 224, 224)
        >>> depth_valid_masks = torch.randn(1, 1, 224, 224)
        >>> outputs = model(pixel_values, pixel_values_depth=pixel_values_depth, depth_valid_masks=depth_valid_masks)
        >>> last_hidden_states = outputs.last_hidden_state
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Determine video dimensions for RoPE
        # Note: pixel_values passed to embeddings can be 4D only for now
        if pixel_values.dim() != 4:
            raise NotImplementedError("TomatoViTModel currently expects 4D pixel_values (B,C,H,W).")
        else:
            t_frames = 1
            height = pixel_values.shape[2]
            width = pixel_values.shape[3]

        if pixel_values_depth is not None and pixel_values_depth.shape[1] == 1:
            if depth_valid_masks is not None:
                pixel_values_depth = torch.cat(
                    [
                        pixel_values_depth,
                        pixel_values_depth,
                        depth_valid_masks.to(device=pixel_values_depth.device, dtype=pixel_values_depth.dtype),
                    ],
                    dim=1,
                )
            else:
                pixel_values_depth = pixel_values_depth.repeat(1, 3, 1, 1)

        # 1. Embeddings
        hidden_states = self.embeddings(pixel_values)
        batch_size, total_patches, _ = hidden_states.shape
        hidden_states_depth = self.embeddings_depth(pixel_values_depth)

        # 2. Mask Handling
        if mask_ratio > 0.0:
            num_masked = int(total_patches * mask_ratio)

            # Generate random mask indices for each sample in the batch
            mask_indices = torch.rand(batch_size, total_patches, device=hidden_states.device).argsort(dim=-1)[
                :, :num_masked
            ]

            # Apply mask to hidden states
            if mask_rgb:
                hidden_states[torch.arange(batch_size).unsqueeze(1), mask_indices] = self.mask_embedding.to(
                    hidden_states_depth.device, dtype=hidden_states_depth.dtype
                )

            if mask_depth:
                hidden_states_depth[torch.arange(batch_size).unsqueeze(1), mask_indices] = self.mask_embedding.to(
                    hidden_states_depth.device, dtype=hidden_states_depth.dtype
                )
        else:
            mask_indices = None

        # 3. RoPE Construction
        freqs_full = self.video_rope(
            t=t_frames,
            h=height // self.config.patch_size,
            w=width // self.config.patch_size,
            device=pixel_values.device,
        )
        freqs_visible = freqs_full.unsqueeze(0).expand(batch_size, -1, -1)
        # Concatenate D/2 + D/2 -> D for applying rope
        freqs_visible = torch.cat([freqs_visible, freqs_visible], dim=-1)

        freqs_visible_depth = freqs_full.unsqueeze(0).expand(batch_size, -1, -1)
        freqs_visible_depth = torch.cat([freqs_visible_depth, freqs_visible_depth], dim=-1)

        # 4. Pre-Norm & Encoder
        hidden_states = self.layernorm_pre(hidden_states)
        hidden_states_depth = self.layernorm_pre_depth(hidden_states_depth)

        encoder_outputs: TomatoViTModelOutput = self.encoder(
            hidden_states,
            hidden_states_depth,
            attention_mask=None,
            attention_mask_depth=None,
            rotary_pos_emb=freqs_visible,
            rotary_pos_emb_depth=freqs_visible_depth,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs.last_hidden_state
        sequence_output_depth = encoder_outputs.last_hidden_state_depth

        if self.layernorm_post is not None:
            sequence_output = self.layernorm_post(sequence_output)

        if self.layernorm_post_depth is not None:
            sequence_output_depth = self.layernorm_post_depth(sequence_output_depth)

        # 5. Pooling Head
        pooled_output = None
        pooled_output_depth = None
        if self.head is not None:
            pooled_output = self.head(sequence_output)

        if self.head_depth is not None:
            pooled_output_depth = self.head_depth(sequence_output_depth)

        if not return_dict:
            return (sequence_output, pooled_output) + (sequence_output_depth, pooled_output_depth) + encoder_outputs[2:]

        return TomatoViTModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            last_hidden_state_depth=sequence_output_depth,
            pooler_output_depth=pooled_output_depth,
            hidden_states_depth=encoder_outputs.hidden_states_depth,
            attentions_depth=encoder_outputs.attentions_depth,
            mask=mask_indices,
        )


def tomato_vit_l(pretrained: bool = False, ckpt_path=None, **kwargs):
    config = TomatoViTConfig(
        patch_size=14,
        hidden_size=1024,
        num_attention_heads=1024 // 64,
        num_hidden_layers=24,
        intermediate_size=4096,
        hidden_act="gelu",
        layer_norm_type="layer_norm",
        use_head=True,
        mot_layers=[5, 11, 17, 23],
    )
    config._attn_implementation = "flash_attention_2"
    model = TomatoViTModel(config)
    return model


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    model = tomato_vit_l(pretrained=False).cuda().eval()
    print(model)

    test_rgb = torch.randn(4, 3, 224, 224, device="cuda")
    test_depth = torch.randn(4, 3, 224, 224, device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(test_rgb, test_depth, mask_ratio=0.75)

    print(output)
