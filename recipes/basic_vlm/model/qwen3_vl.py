"""Qwen3-VL model helpers for the Basic VLM recipe."""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F


def calculate_model_flops(
    self,
    *,
    batch_size: int,
    seq_len: int,
    attention_mask: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    is_training: bool = True,
    freeze_vit: bool = False,
    freeze_projector: bool = False,
    freeze_llm: bool = False,
) -> float:
    """Estimate local-rank logical Qwen3-VL FLOPs for one prepared batch."""
    batch = int(batch_size)
    tokens = int(seq_len)
    if batch <= 0 or tokens <= 0:
        raise ValueError("batch_size and seq_len must be > 0")

    text_cfg = self.config.text_config
    vision_cfg = self.config.vision_config

    text_layers = int(text_cfg.num_hidden_layers)
    text_hidden = int(text_cfg.hidden_size)
    text_intermediate = int(text_cfg.intermediate_size)
    vocab = int(text_cfg.vocab_size)
    attention_token_pairs = _count_attention_token_pairs(
        batch=batch,
        tokens=tokens,
        attention_mask=attention_mask,
        attn_implementation=getattr(self.config, "_attn_implementation", None),
    )

    language_per_layer = (
        8 * batch * tokens * text_hidden * text_hidden
        + 4 * attention_token_pairs * text_hidden
        + 6 * batch * tokens * text_hidden * text_intermediate
    )
    language_flops = float(text_layers * language_per_layer + 2 * batch * tokens * text_hidden * vocab)

    vit_flops, merger_flops = _calculate_vision_flops(
        vision_cfg=vision_cfg,
        image_grid_thw=image_grid_thw,
    )

    if not is_training:
        return language_flops + vit_flops + merger_flops

    upstream_of_llm_is_trained = (not freeze_projector) or (not freeze_vit)
    upstream_of_merger_is_trained = not freeze_vit

    llm_mult = 3.0 if not freeze_llm else (2.0 if upstream_of_llm_is_trained else 1.0)
    merger_mult = 3.0 if not freeze_projector else (2.0 if upstream_of_merger_is_trained else 1.0)
    vit_mult = 3.0 if not freeze_vit else 1.0

    return language_flops * llm_mult + merger_flops * merger_mult + vit_flops * vit_mult


def patch_qwen3vl_model_flops(model):
    """Inject Qwen3-VL FLOPs estimation into the runtime model instance."""
    model.calculate_model_flops = MethodType(calculate_model_flops, model)
    return model


def _count_attention_token_pairs(
    *,
    batch: int,
    tokens: int,
    attention_mask: torch.Tensor | None,
    attn_implementation: str | None,
) -> int:
    """Count logical attention token pairs for padded or packed text tokens."""
    attention_token_pairs = batch * tokens * tokens
    if attention_mask is None or attention_mask.ndim != 2:
        return int(attention_token_pairs)

    mask = attention_mask.detach().to(device="cpu", dtype=torch.long)
    max_mask_value = int(mask.max().item()) if mask.numel() > 0 else 0
    if max_mask_value > 1:
        attention_token_pairs = 0
        for row in mask:
            segment_ids = row[row > 0]
            if segment_ids.numel() == 0:
                continue
            segment_lengths = torch.bincount(segment_ids, minlength=max_mask_value + 1)[1:]
            attention_token_pairs += int(torch.square(segment_lengths).sum().item())
    elif attn_implementation == "flash_attention_2":
        valid_lengths = mask.ne(0).sum(dim=-1)
        attention_token_pairs = int(torch.square(valid_lengths).sum().item())

    if attention_token_pairs <= 0:
        raise ValueError("attention_mask must contain at least one valid token")
    return int(attention_token_pairs)


def _calculate_vision_flops(
    *,
    vision_cfg,
    image_grid_thw: torch.Tensor | None,
) -> tuple[float, float]:
    """Estimate Qwen3-VL vision encoder and merger FLOPs."""
    if image_grid_thw is None or image_grid_thw.numel() == 0:
        return 0.0, 0.0

    grid = image_grid_thw.detach().to(device="cpu", dtype=torch.long).reshape(-1, 3)
    if torch.any(grid <= 0):
        raise ValueError("image_grid_thw must contain positive temporal/height/width values")

    temporal_tokens = grid[:, 0]
    height_tokens = grid[:, 1]
    width_tokens = grid[:, 2]
    spatial_merge_size = int(vision_cfg.spatial_merge_size)

    if torch.any(height_tokens % spatial_merge_size != 0) or torch.any(width_tokens % spatial_merge_size != 0):
        raise ValueError("image_grid_thw height/width must be divisible by spatial_merge_size")

    visual_seq_lens = temporal_tokens * height_tokens * width_tokens
    merged_seq_lens = temporal_tokens * (height_tokens // spatial_merge_size) * (width_tokens // spatial_merge_size)

    vision_hidden = int(vision_cfg.hidden_size)
    vision_layers = int(vision_cfg.depth)
    vision_intermediate = int(vision_cfg.intermediate_size)
    vision_out_hidden = int(vision_cfg.out_hidden_size)
    channels = int(vision_cfg.in_channels)
    patch_size = int(vision_cfg.patch_size)
    temporal_patch_size = int(vision_cfg.temporal_patch_size)

    visual_tokens = int(visual_seq_lens.sum().item())
    patch_dim = channels * temporal_patch_size * patch_size * patch_size
    patch_embed_flops = 2 * visual_tokens * patch_dim * vision_hidden
    attention_projection_flops = 8 * vision_hidden * vision_hidden * visual_tokens
    attention_scores_flops = 4 * vision_hidden * int(torch.square(visual_seq_lens).sum().item())
    mlp_flops = 4 * vision_hidden * vision_intermediate * visual_tokens
    vision_encoder_flops = vision_layers * (attention_projection_flops + attention_scores_flops + mlp_flops)

    merger_input_hidden = vision_hidden * (spatial_merge_size**2)
    vit_flops = float(patch_embed_flops + vision_encoder_flops)
    merger_flops = float(2 * int(merged_seq_lens.sum().item()) * merger_input_hidden * vision_out_hidden)
    return vit_flops, merger_flops


def patch_qwen3vl_conv3d(model):
    """Apply recipe-local Qwen3-VL vision/runtime compatibility patches.

    These are behavior patches, not optimizations:
    - run vision patch embedding as fp32 linear math instead of Conv3D
    """

    def patch_embed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run Qwen3-VL vision patch projection in fp32 and cast back."""
        target_dtype = self.proj.weight.dtype
        proj_weight = self.proj.weight
        proj_bias = self.proj.bias
        device_type = hidden_states.device.type

        with torch.amp.autocast(device_type=device_type, enabled=False):
            hidden_states_fp32 = hidden_states.float()
            weight_fp32 = proj_weight.view(self.embed_dim, -1).float()
            bias_fp32 = proj_bias.float() if proj_bias is not None else None
            hidden_states = F.linear(hidden_states_fp32, weight_fp32, bias_fp32)

        return hidden_states.to(dtype=target_dtype)

    model.model.visual.patch_embed.forward = MethodType(patch_embed_forward, model.model.visual.patch_embed)

    return model
