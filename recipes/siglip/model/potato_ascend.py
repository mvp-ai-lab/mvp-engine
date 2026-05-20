# $$$!!Warning: Huawei key information asset. No spread without permission.$$$
# !CODEMARK:RKeR1B8WMAfemkt1tTDGp4eOEddgxKn4NOPmdw0w+6Q3n1pxgDEX+kGBiRV20e1NKuLwOh60qWwx
# 7DOUvTqsDhlXzSmTU10bmKROYG5QSBFCWYwf86o8mK04Er8uGzTBg5d382PpwtM5nwkqZq8hMwFX
# h8Y19DRWNCr1BSvLSxDkRny/s6flKhpH3eCZmhzclRexVeP8FHktzbepcO6Qs4RPFvCrWb0AoC9z
# KhxXbJnOB5eg5r7y5W/szwO0/P/sKVJcscgQaJaqcUVNfZfNgFW5Ul5iuoO/Y5fkcJPN5wRYkd47
# ZQHm6xmY25eP5X8s#!
# $$$!!Warning: Deleting or modifying the preceding information is prohibited.$$$
import math
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    ACT2FN,
    GradientCheckpointingLayer,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLVisionConfig,
    logger,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
    Qwen2VLImageProcessorFast as _Qwen2VLImageProcessorFast,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
    SizeDict,
    TensorType,
    group_images_by_shape,
    reorder_images,
    smart_resize,
)

from ..model.rot_attn_pooling import RotAttentionPool2d

# def apply_rotary_pos_emb_vision(
#     q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     cos, sin = cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2)
#     q = q.unsqueeze(0)
#     k = k.unsqueeze(0)

#     return torch_npu.npu_rotary_mul(q, cos, sin)[0], torch_npu.npu_rotary_mul(k, cos, sin)[0]


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        PotatoRMSNorm is equivalent to Qwen2RMSNorm and T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Conv3D2BMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, inputs):
        ctx.save_for_backward(inputs, weight)
        result = torch.matmul(weight, inputs)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        grad_weight = torch.matmul(
            grad_output.transpose(0, 1).squeeze(2),
            inputs.squeeze(2).to(grad_output.dtype),
        )  # [h_weight, s]
        grad_inputs = torch.matmul(grad_output.squeeze(2), weight.to(grad_output.dtype))  # [h_input, s]

        return grad_weight, grad_inputs, None


class BMMModule(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
        dtype=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(
            torch.empty(
                (embed_dim, in_channels, temporal_patch_size, patch_size, patch_size),
                dtype=dtype,
            )
        )
        nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        batch_mm_x = self.weight.view(self.embed_dim, -1)
        batch_mm_y = hidden_states.view(hidden_states.size(0), -1, 1)

        hidden_states = Conv3D2BMM.apply(batch_mm_x, batch_mm_y)
        hidden_states = hidden_states.view(-1, self.embed_dim)
        return hidden_states


class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = BMMModule(patch_size, temporal_patch_size, in_channels, embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )

        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen2VLImageProcessorFast(_Qwen2VLImageProcessorFast):
    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ):
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            # Fused rescale and normalize
            patches = self.rescale_and_normalize(
                stacked_images,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            if patches.ndim == 4:
                # add a temporal dimension if we have images
                patches = patches.unsqueeze(1)
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            # Step 1: Reshape into a 5D tensor (B, T, C, H, W)
            # This groups temporal components and spatial components. Max dimensions: 5
            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size * channel,
                grid_h * patch_size,
                grid_w * patch_size,
            )

            # Step 2: Factorize spatial dimensions to separate grid from patches.
            # This results in a 7D tensor. Max dimensions: 7
            patches = patches.view(
                batch_size * grid_t * temporal_patch_size * channel,  # 0
                grid_h // merge_size,  # 1
                merge_size,  # 2
                patch_size,  # 3
                grid_w // merge_size,  # 4
                merge_size,  # 5
                patch_size,  # 6
            )
            patches = patches.permute(0, 1, 4, 2, 5, 3, 6)

            patches = patches.reshape(
                batch_size * grid_t,  # 0
                temporal_patch_size,  # 1
                channel,  # 2
                grid_h // merge_size,  # 3
                grid_w // merge_size,  # 4
                merge_size * merge_size,  # 5
                patch_size * patch_size,  # 6
            )

            patches = patches.permute(0, 3, 4, 5, 2, 1, 6)

            # Step 4: Reshape into the final 3D tensor by flattening grid and patch dimensions.
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                -1,  # PyTorch automatically calculates the feature dimension
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids = reorder_images(processed_grids, grouped_images_index)
        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )


class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.up_proj(hidden_state)))


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen2_5_VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen2_5_VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5_VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen2_5_VLVisionAttention(config=config)
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VLPreTrainedModel):
    config: Qwen2_5_VLVisionConfig
    _no_split_modules = ["Qwen2_5_VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen2_5_VLVisionBlock(config) for _ in range(config.depth)])
        self.gradient_checkpointing = False

        self.final_pooling = RotAttentionPool2d(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            num_heads=4,
            qkv_bias=True,
        )
        # self.final_pooling = NPURotAttentionPool2d(
        #     in_features=config.hidden_size,
        #     out_features=config.hidden_size,
        #     num_heads=4,
        #     qkv_bias=True,
        # )

        self.processor = Qwen2VLImageProcessorFast(
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            do_convert_rgb=False,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
        )

        self.prepooling_norm = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.post_norm = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.proj = nn.Parameter((config.hidden_size**-0.5) * torch.randn(config.hidden_size, config.hidden_size))

        self.anyres = kwargs.get("anyres", False)

        if not self.anyres:
            self.rot_pos_emb_cache = None
            self.window_index_cache = None
            self.cu_window_seqlens_cache = None

    def rot_pos_emb(self, grid_thw):
        if self.rot_pos_emb_cache is not None and not self.anyres:
            return self.rot_pos_emb_cache

        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)

        if not self.anyres:
            self.rot_pos_emb_cache = rotary_pos_emb
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        if self.window_index_cache is not None and not self.anyres:
            return self.window_index_cache, self.cu_window_seqlens_cache

        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        if not self.anyres:
            self.window_index_cache = window_index
            self.cu_window_seqlens_cache = cu_window_seqlens

        return window_index, cu_window_seqlens

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.Tensor`): Input pixel values of shape `(batch_size, num_channels, height, width)`.
        Returns:
            `torch.Tensor`: hidden_states.
        """

        if self.anyres:
            raise NotImplementedError("Anyres is not supported yet.")
        else:
            processed_pixel_values = self.processor(pixel_values)
            hidden_states = processed_pixel_values["pixel_values"].to(
                device=pixel_values.device, dtype=pixel_values.dtype
            )
            grid_thw = processed_pixel_values["image_grid_thw"].to(device=pixel_values.device)

            hidden_states = self.patch_embed(hidden_states)
            rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.prepooling_norm(hidden_states)
        if self.anyres:
            raise NotImplementedError("Anyres is not supported yet.")
        else:
            # We don't do spatial merging during vit training
            b = grid_thw.shape[0]
            h = grid_thw[0, 1]
            w = grid_thw[0, 2]
            c = h * w
            hidden_states = rearrange(hidden_states, "(b c) d -> b c d", b=b, c=c).permute(0, 2, 1)
            hidden_states = hidden_states.view(b, -1, h, w)
            hidden_states = self.final_pooling(hidden_states)
            hidden_states = self.post_norm(hidden_states)

        if self.proj is not None:
            hidden_states = hidden_states @ self.proj

        return hidden_states


class POTATO_26_ASCEND(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, *inputs, **kwargs) -> None:
        config = Qwen2_5_VLVisionConfig(
            depth=26,
            hidden_size=1536,
            num_heads=16,
            intermediate_size=4608,
            temporal_patch_size=1,
            hidden_act="gelu",
            fullatt_block_indexes=[5, 12, 19, 25],
        )
        super().__init__(config, *inputs, **kwargs)
        self.config = config


class POTATO_26_ASCEND_ANYRES(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, *inputs, **kwargs) -> None:
        config = Qwen2_5_VLVisionConfig(
            depth=26,
            hidden_size=1536,
            num_heads=16,
            intermediate_size=4608,
            temporal_patch_size=1,
            hidden_act="gelu",
            fullatt_block_indexes=[5, 12, 19, 25],
        )
        super().__init__(config, anyres=True, *inputs, **kwargs)
        self.config = config


def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    import random

    random.seed(0)
    import numpy as np

    np.random.seed(0)

    model = POTATO_26_ASCEND()
    print(model)

    model = model.npu().to(torch.bfloat16)

    torch.manual_seed(42)
    import random

    random.seed(0)
    import numpy as np

    np.random.seed(0)

    pixel_values = torch.randn(16, 3, 224, 224).to(torch.bfloat16).npu()

    # torch.save(pixel_values.cpu(), "pixel_values.pth")
    output = model(pixel_values)

    print(output.shape)
    print(torch.norm(output, dim=-1).mean())
    print(output.max())


if __name__ == "__main__":
    main()
