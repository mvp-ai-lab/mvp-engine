"""Qwen3-VL model helpers for the OpenBee recipe."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast

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
    model = inject_sum_loss_forward(model)

    apply_freeze_policy(
        model,
        freeze_vit=model_config.freeze_vit,
        freeze_merger=model_config.freeze_merger,
        freeze_llm=model_config.freeze_llm,
    )

    return model
