"""Qwen3-VL model helpers for the OpenBee recipe."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLModelOutputWithPast,
)

# ---------------------------------------------------------------------------
# Parameter-name prefixes for each logical sub-module
# ---------------------------------------------------------------------------

# Visual encoder (ViT): patch embedding + transformer blocks
VIT_PREFIXES = (
    "model.visual.patch_embed.",
    "model.visual.blocks.",
)

# Projector / merger stack that maps visual tokens into the LLM embedding space
MERGER_PREFIXES = (
    "model.visual.merger.",
    "model.visual.deepstack_merger_list.",
)

# Language model backbone and output head
LLM_PREFIXES = (
    "model.language_model.",
    "lm_head.",
)


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(p) for p in prefixes)


def _slice_hidden(hidden_states: torch.Tensor, logits_to_keep: int | torch.Tensor) -> torch.Tensor:
    """Slice trailing hidden states based on ``logits_to_keep``."""
    sl = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    if hidden_states.dim() == 3:
        return hidden_states[:, sl, :]
    return hidden_states[sl, :]


def _shift_labels(labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Pad-right by one token then shift left for causal LM loss."""
    labels = F.pad(labels, (0, 1), value=ignore_index)
    return labels[..., 1:].contiguous()


def _get_per_token_ce(ignore_index: int):
    """Return a per-token CE callable, preferring liger when available."""
    try:
        from liger_kernel.transformers import LigerCrossEntropyLoss

        return LigerCrossEntropyLoss(reduction="none", ignore_index=ignore_index)
    except ImportError:
        return lambda logits, labels: F.cross_entropy(
            logits,
            labels,
            ignore_index=ignore_index,
            reduction="none",
        )


def apply_freeze_policy(
    model,
    *,
    freeze_vit: bool = True,
    freeze_merger: bool = False,
    freeze_llm: bool = False,
) -> dict[str, int]:
    """Freeze sub-modules of a Qwen3-VL model according to the given flags.

    Args:
        model: Loaded Qwen3-VL model instance.
        freeze_vit: When ``True``, freeze the visual encoder (ViT blocks +
            patch embedding).
        freeze_merger: When ``True``, freeze the projector / merger modules
            (``model.visual.merger`` and ``model.visual.deepstack_merger_list``).
        freeze_llm: When ``True``, freeze the language model backbone and the
            LM head (``model.language_model`` and ``lm_head``).

    Returns:
        A dict mapping sub-module name to the number of frozen parameters.
    """
    frozen_counts: dict[str, int] = {"vit": 0, "merger": 0, "llm": 0}

    for name, parameter in model.named_parameters():
        if freeze_vit and _matches(name, VIT_PREFIXES):
            parameter.requires_grad = False
            frozen_counts["vit"] += parameter.numel()
        elif freeze_merger and _matches(name, MERGER_PREFIXES):
            parameter.requires_grad = False
            frozen_counts["merger"] += parameter.numel()
        elif freeze_llm and _matches(name, LLM_PREFIXES):
            parameter.requires_grad = False
            frozen_counts["llm"] += parameter.numel()

    return frozen_counts


def inject_model_flops_calculation(model):
    """Inject per-process FLOPs estimation onto the loaded Qwen3-VL model."""

    def calculate_model_flops(
        self,
        *,
        batch_size: int,
        seq_len: int,
        image_grid_thw: torch.Tensor | None = None,
        is_training: bool = True,
        freeze_vit: bool = False,
        freeze_merger: bool = False,
        freeze_llm: bool = False,
    ) -> float:
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

        language_per_layer = (
            8 * batch * tokens * text_hidden * text_hidden
            + 4 * batch * tokens * tokens * text_hidden
            # SwiGLU has three matrices: gate_proj, up_proj (H→I each) and down_proj (I→H),
            # so MLP FLOPs are 6×B×T×H×I, not 4× as in a standard two-layer FFN.
            + 6 * batch * tokens * text_hidden * text_intermediate
        )
        language_flops = float(text_layers * language_per_layer + 2 * batch * tokens * text_hidden * vocab)

        vit_flops = 0.0
        merger_flops = 0.0
        if image_grid_thw is not None and image_grid_thw.numel() > 0:
            grid = image_grid_thw.detach().to(device="cpu", dtype=torch.long).reshape(-1, 3)
            if torch.any(grid <= 0):
                raise ValueError("image_grid_thw must contain positive temporal/height/width values")

            temporal_tokens = grid[:, 0]
            height_tokens = grid[:, 1]
            width_tokens = grid[:, 2]

            visual_seq_lens = temporal_tokens * height_tokens * width_tokens
            merged_seq_lens = (
                temporal_tokens
                * (height_tokens // int(vision_cfg.spatial_merge_size))
                * (width_tokens // int(vision_cfg.spatial_merge_size))
            )

            if torch.any(height_tokens % int(vision_cfg.spatial_merge_size) != 0) or torch.any(
                width_tokens % int(vision_cfg.spatial_merge_size) != 0
            ):
                raise ValueError("image_grid_thw height/width must be divisible by spatial_merge_size")

            vision_hidden = int(vision_cfg.hidden_size)
            vision_layers = int(vision_cfg.depth)
            vision_intermediate = int(vision_cfg.intermediate_size)
            vision_out_hidden = int(vision_cfg.out_hidden_size)
            channels = int(vision_cfg.in_channels)
            patch_size = int(vision_cfg.patch_size)
            temporal_patch_size = int(vision_cfg.temporal_patch_size)
            spatial_merge_size = int(vision_cfg.spatial_merge_size)

            patch_dim = channels * temporal_patch_size * patch_size * patch_size
            patch_embed_flops = 2 * int(visual_seq_lens.sum().item()) * patch_dim * vision_hidden
            attention_projection_flops = 8 * vision_hidden * vision_hidden * int(visual_seq_lens.sum().item())
            attention_scores_flops = 4 * vision_hidden * int(torch.square(visual_seq_lens).sum().item())
            mlp_flops = 4 * vision_hidden * vision_intermediate * int(visual_seq_lens.sum().item())
            vision_encoder_flops = vision_layers * (attention_projection_flops + attention_scores_flops + mlp_flops)

            merger_input_hidden = vision_hidden * (spatial_merge_size**2)
            vit_flops = float(patch_embed_flops + vision_encoder_flops)
            merger_flops = float(2 * int(merged_seq_lens.sum().item()) * merger_input_hidden * vision_out_hidden)

        if not is_training:
            return language_flops + vit_flops + merger_flops

        # Per-component backward multipliers.
        #
        # Gradient flow order (backward): loss → LLM → merger → ViT → pixels
        # Pixel values never require grad, so activation gradients always stop at ViT's input —
        # even when ViT is trainable.
        #
        # Rules:
        #   - Not frozen: weight grads + activation grads → 3× forward FLOPs
        #   - Frozen, but upstream module is trainable: activation grads still flow through
        #     this module so that upstream weight grads can be computed → 2× forward FLOPs
        #   - Frozen, nothing upstream is trainable (or this is ViT whose input has no grad):
        #     no backward at all → 1× forward FLOPs
        upstream_of_llm_is_trained = (not freeze_merger) or (not freeze_vit)
        upstream_of_merger_is_trained = not freeze_vit

        llm_mult = 3.0 if not freeze_llm else (2.0 if upstream_of_llm_is_trained else 1.0)
        merger_mult = 3.0 if not freeze_merger else (2.0 if upstream_of_merger_is_trained else 1.0)
        # ViT: input (pixel values) never requires grad, so activation grad never propagates
        # further back regardless of upstream modules.  Backward only happens for weight grads.
        vit_mult = 3.0 if not freeze_vit else 1.0

        return language_flops * llm_mult + merger_flops * merger_mult + vit_flops * vit_mult

    model.calculate_model_flops = MethodType(calculate_model_flops, model)
    return model


def _build_active_token_mask(
    input_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Return a per-token valid mask that ignores fully masked dummy-image suffixes."""
    if input_ids is None:
        return None
    if attention_mask is None:
        return torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask.ndim == 2:
        return attention_mask.ne(0)
    if attention_mask.ndim == 4:
        diagonal = torch.diagonal(attention_mask[:, 0], dim1=1, dim2=2)
        if torch.is_floating_point(diagonal):
            return diagonal.eq(0)
        return diagonal.ne(0)
    raise ValueError(f"Unsupported attention_mask shape for dummy-image handling: {tuple(attention_mask.shape)}.")


def _has_active_multimodal_tokens(
    input_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    *,
    token_id: int,
) -> bool:
    """Check whether a token id is present among active tokens only."""
    if input_ids is None or input_ids.numel() == 0:
        return False
    active_token_mask = _build_active_token_mask(input_ids, attention_mask)
    if active_token_mask is None:
        return False
    return bool(((input_ids == token_id) & active_token_mask).any().item())


def _attach_zero_visual_dependency(
    inputs_embeds: torch.Tensor,
    visual_embeds: torch.Tensor,
    deepstack_visual_embeds: list[torch.Tensor] | None,
) -> torch.Tensor:
    """Keep visual parameters in the autograd graph without injecting visual content."""
    inputs_embeds = inputs_embeds + visual_embeds.mean() * 0
    if deepstack_visual_embeds is not None:
        for deepstack_embed in deepstack_visual_embeds:
            inputs_embeds = inputs_embeds + deepstack_embed.mean() * 0
    return inputs_embeds


def inject_batch_level_dummy_image_handling(model):
    """Patch Qwen3-VL to follow HK-style local-batch dummy-image behaviour."""

    def model_forward_with_batch_level_dummy_image(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        has_image_tokens = _has_active_multimodal_tokens(
            input_ids,
            attention_mask,
            token_id=int(self.config.image_token_id),
        )
        has_video_tokens = _has_active_multimodal_tokens(
            input_ids,
            attention_mask,
            token_id=int(self.config.video_token_id),
        )

        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_image_embeds = image_outputs.deepstack_features
            if has_image_tokens:
                image_mask, _ = self.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    image_features=image_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            else:
                inputs_embeds = _attach_zero_visual_dependency(inputs_embeds, image_embeds, deepstack_image_embeds)

        if pixel_values_videos is not None:
            video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_video_embeds = video_outputs.deepstack_features
            if has_video_tokens:
                _, video_mask = self.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    video_features=video_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            else:
                inputs_embeds = _attach_zero_visual_dependency(inputs_embeds, video_embeds, deepstack_video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )

    model.model.forward = MethodType(model_forward_with_batch_level_dummy_image, model.model)
    return model


def apply_qwen3_vl_compat_patches(model):
    """Mirror the LLaMA-Factory Qwen3-VL vision/runtime patches.

    These are behavior patches, not optimizations:
    - run vision patch embedding as fp32 linear math instead of Conv3D
    - replace the vision RoPE helper with the LF implementation
    """

    def patch_embed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        proj_weight = self.proj.weight
        proj_bias = self.proj.bias

        with torch.amp.autocast(device_type="cuda", enabled=False):
            hidden_states_fp32 = hidden_states.float()
            weight_fp32 = proj_weight.view(self.embed_dim, -1).float()
            bias_fp32 = proj_bias.float() if proj_bias is not None else None
            hidden_states = F.linear(hidden_states_fp32, weight_fp32, bias_fp32)

        return hidden_states.to(dtype=target_dtype)

    model.model.visual.patch_embed.forward = MethodType(patch_embed_forward, model.model.visual.patch_embed)

    return model


def inject_sum_loss_forward(model, *, chunk_size: int = 4096):
    """Patch Qwen3-VL forward to return summed CE loss."""

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]

        loss = None
        logits = None
        if labels is not None:
            hs = _slice_hidden(hidden_states, logits_to_keep)
            shift_labels = _shift_labels(labels)
            flat_hs = hs.reshape(-1, hs.size(-1))
            flat_labels = shift_labels.reshape(-1)
            lm_head_bias = getattr(self.lm_head, "bias", None)
            ce_fn = _get_per_token_ce(ignore_index=-100)

            loss = flat_hs.new_zeros(())
            for start in range(0, flat_hs.size(0), chunk_size):
                end = min(start + chunk_size, flat_hs.size(0))
                chunk_logits = F.linear(flat_hs[start:end], self.lm_head.weight, lm_head_bias)
                loss = loss + ce_fn(chunk_logits, flat_labels[start:end]).sum()
                del chunk_logits
        else:
            logits = self.lm_head(_slice_hidden(hidden_states, logits_to_keep))

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    model.forward = MethodType(forward, model)
    return model


def build_qwen3_vl_model(model_config: Any):
    """Load the Qwen3-VL model checkpoint and apply the configured freeze policy.

    Args:
        model_config: Recipe model config (``OpenbeeModelConfig``) with load
            and freeze settings.

    Returns:
        The initialized Qwen3-VL model.
    """
    model = AutoModelForImageTextToText.from_pretrained(
        model_config.pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation=model_config.attn_implementation,
    )
    model = inject_model_flops_calculation(model)
    model = inject_batch_level_dummy_image_handling(model)
    model = inject_sum_loss_forward(model)

    apply_freeze_policy(
        model,
        freeze_vit=model_config.freeze_vit,
        freeze_merger=model_config.freeze_merger,
        freeze_llm=model_config.freeze_llm,
    )
    model = apply_qwen3_vl_compat_patches(model)

    return model
