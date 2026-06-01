"""Qwen2.5-VL model patches used by the recipe."""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F


def patch_qwen2_5vl_conv3d(model: torch.nn.Module) -> torch.nn.Module:
    """Run Qwen2.5-VL patch embedding projection in fp32 for stability."""
    patch_embed = getattr(getattr(getattr(model, "model", None), "visual", None), "patch_embed", None)
    projection = getattr(patch_embed, "proj", None)
    if patch_embed is None or projection is None or not isinstance(projection, torch.nn.Conv3d):
        raise AttributeError("Expected Qwen2.5-VL patch embedding at `model.model.visual.patch_embed.proj`.")

    def patch_embed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        device_type = hidden_states.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            hidden_states = hidden_states.float().view(
                -1,
                int(self.in_channels),
                int(self.temporal_patch_size),
                int(self.patch_size),
                int(self.patch_size),
            )
            hidden_states = F.linear(
                hidden_states.flatten(1),
                self.proj.weight.view(int(self.embed_dim), -1).float(),
                None,
            )
        return hidden_states.to(dtype=target_dtype)

    patch_embed.forward = MethodType(patch_embed_forward, patch_embed)
    return model


def patch_qwen2_5vl_model_flops(model: torch.nn.Module) -> torch.nn.Module:
    """Inject local FLOPs estimation into the runtime model instance."""
    model.calculate_model_flops = MethodType(calculate_model_flops, model)
    return model


def disable_qwen2_5vl_cache(model: torch.nn.Module) -> torch.nn.Module:
    """Disable generation KV cache before enabling gradient checkpointing."""
    for target in (
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "language_model", None),
    ):
        config = getattr(target, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            setattr(config, "use_cache", False)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "use_cache"):
        setattr(generation_config, "use_cache", False)
    return model


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
    """Estimate local-rank logical FLOPs for one prepared batch."""
    batch = int(batch_size)
    tokens = int(seq_len)
    if batch <= 0 or tokens <= 0:
        raise ValueError("batch_size and seq_len must be > 0")

    text_cfg = self.config.text_config
    hidden = int(text_cfg.hidden_size)
    intermediate = int(text_cfg.intermediate_size)
    text_layers = int(text_cfg.num_hidden_layers)
    vocab = int(text_cfg.vocab_size)
    attention_pairs = _count_attention_token_pairs(batch=batch, tokens=tokens, attention_mask=attention_mask)
    language_per_layer = (
        8 * batch * tokens * hidden * hidden
        + 4 * attention_pairs * hidden
        + 6 * batch * tokens * hidden * intermediate
    )
    language_flops = float(text_layers * language_per_layer + 2 * batch * tokens * hidden * vocab)

    vit_flops, merger_flops = _calculate_vision_flops(
        vision_cfg=self.config.vision_config,
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


def _count_attention_token_pairs(
    *,
    batch: int,
    tokens: int,
    attention_mask: torch.Tensor | None,
) -> int:
    if attention_mask is None or attention_mask.ndim != 2:
        return int(batch * tokens * tokens)

    mask = attention_mask.detach().to(device="cpu", dtype=torch.long)
    if mask.numel() == 0:
        return 0

    if int(mask.max().item()) > 1:
        total = 0
        for row in mask:
            segment_ids = row[row > 0]
            if segment_ids.numel() == 0:
                continue
            lengths = torch.bincount(segment_ids)[1:]
            total += int(torch.square(lengths).sum().item())
        return total

    valid_lengths = mask.ne(0).sum(dim=-1)
    return int(torch.square(valid_lengths).sum().item())


def _calculate_vision_flops(*, vision_cfg, image_grid_thw: torch.Tensor | None) -> tuple[float, float]:
    if image_grid_thw is None or image_grid_thw.numel() == 0:
        return 0.0, 0.0

    grid = image_grid_thw.detach().to(device="cpu", dtype=torch.long).reshape(-1, 3)
    if torch.any(grid <= 0):
        raise ValueError("image_grid_thw must contain positive temporal/height/width values")

    spatial_merge_size = int(vision_cfg.spatial_merge_size)
    visual_seq_lens = grid.prod(dim=-1)
    merged_seq_lens = grid[:, 0] * (grid[:, 1] // spatial_merge_size) * (grid[:, 2] // spatial_merge_size)

    hidden = int(vision_cfg.hidden_size)
    layers = int(vision_cfg.depth)
    intermediate = int(vision_cfg.intermediate_size)
    out_hidden = int(vision_cfg.out_hidden_size)
    channels = int(vision_cfg.in_channels)
    patch_size = int(vision_cfg.patch_size)
    temporal_patch_size = int(vision_cfg.temporal_patch_size)
    visual_tokens = int(visual_seq_lens.sum().item())

    patch_dim = channels * temporal_patch_size * patch_size * patch_size
    patch_embed_flops = 2 * visual_tokens * patch_dim * hidden
    qkv_proj_flops = 8 * visual_tokens * hidden * hidden
    mlp_flops = 6 * visual_tokens * hidden * intermediate
    attention_pairs = _estimate_vision_attention_pairs(vision_cfg, grid, visual_seq_lens)
    attention_flops = 4 * attention_pairs * hidden
    vision_encoder_flops = layers * (qkv_proj_flops + mlp_flops) + attention_flops

    merger_input_hidden = hidden * (spatial_merge_size**2)
    merger_flops = 2 * int(merged_seq_lens.sum().item()) * merger_input_hidden * out_hidden
    return float(patch_embed_flops + vision_encoder_flops), float(merger_flops)


def _estimate_vision_attention_pairs(vision_cfg, grid: torch.Tensor, visual_seq_lens: torch.Tensor) -> int:
    full_indexes = set(getattr(vision_cfg, "fullatt_block_indexes", []) or [])
    full_layers = len(full_indexes)
    window_layers = max(int(vision_cfg.depth) - full_layers, 0)
    full_pairs = int(torch.square(visual_seq_lens).sum().item())

    patch_size = max(int(getattr(vision_cfg, "patch_size", 1)), 1)
    window_size = max(int(getattr(vision_cfg, "window_size", patch_size)), patch_size)
    window_tokens = max((window_size // patch_size) ** 2, 1)
    window_pairs = int((grid.prod(dim=-1) * window_tokens).sum().item())
    return full_layers * full_pairs + window_layers * window_pairs


__all__ = [
    "calculate_model_flops",
    "disable_qwen2_5vl_cache",
    "patch_qwen2_5vl_conv3d",
    "patch_qwen2_5vl_model_flops",
]
