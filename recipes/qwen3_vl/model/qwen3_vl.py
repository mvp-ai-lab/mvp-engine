"""Qwen3-VL model helpers for the Qwen3-VL recipe."""

from __future__ import annotations

from types import MethodType
from typing import Any

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


def patch_qwen3vl_context_parallel(model, cp_kit):
    """Patch Qwen3-VL forward for VeOmni-style vision+language context parallelism."""
    if cp_kit.context_size <= 1:
        return model

    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        ALL_ATTENTION_FUNCTIONS,
        BaseModelOutputWithDeepstackFeatures,
        Qwen3VLModelOutputWithPast,
        apply_rotary_pos_emb_vision,
        eager_attention_forward,
    )

    cp_module_config = dict(getattr(model, "CP_MODULE_CONFIG", {}))
    cp_module_config.update(
        {
            "Qwen3VLTextAttention": {"qkv_layout": "BHSD"},
            "Qwen3VLVisionAttention": {"qkv_layout": "BHSD"},
        }
    )
    model.CP_MODULE_CONFIG = cp_module_config
    model.model._cp_kit = cp_kit
    model.model.visual._cp_kit = cp_kit

    # ==================== BEGIN CP Patch #1: Vision Attention ====================
    # Route Qwen vision attention through the shared CP/Ulysses attention interface.
    # =============================================================================
    def vision_attention_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run Qwen vision attention through the CP attention interface."""
        del rotary_pos_emb
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
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

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

    # ===================== END CP Patch #1: Vision Attention =====================

    # ====================== BEGIN CP Patch #2: Vision Forward =====================
    # train_pre_step provides local pixel rows; this forward builds matching local
    # vision position metadata while preserving global image/video geometry.
    # =============================================================================
    def visual_forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Any,
    ) -> BaseModelOutputWithDeepstackFeatures:
        """Run Qwen vision on this context rank's visual merge groups."""
        merge_unit = int(self.spatial_merge_size) ** 2
        local_patch_indices = self._cp_kit.local_visual_patch_indices(
            grid_thw,
            pad_scale=merge_unit,
            device=hidden_states.device,
        )
        total_patches = int(grid_thw.to(dtype=torch.long).prod(dim=-1).sum().item())
        merged_tokens = total_patches // merge_unit
        padded_merged_tokens = (
            (merged_tokens + self._cp_kit.context_size - 1) // self._cp_kit.context_size
        ) * self._cp_kit.context_size
        padded_patches = (padded_merged_tokens - merged_tokens) * merge_unit
        if int(hidden_states.shape[0]) != int(local_patch_indices.numel()):
            raise ValueError(
                "Qwen3-VL CP vision expects context-local pixel_values from train_pre_step, "
                f"got {int(hidden_states.shape[0])} rows; expected {int(local_patch_indices.numel())}."
            )

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw).index_select(0, local_patch_indices)
        rotary_pos_emb = self.rot_pos_emb(grid_thw).index_select(0, local_patch_indices)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        if padded_patches:
            cu_seqlens = torch.cat([cu_seqlens, cu_seqlens[-1:] + padded_patches])

        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=self.merger(hidden_states),
            deepstack_features=deepstack_feature_lists,
        )

    # ======================= END CP Patch #2: Vision Forward ======================

    # ======================== BEGIN CP Patch #3: VLM Forward ======================
    # Accept ready local text/vision tensors, merge multimodal features in
    # full-sequence / hidden-sharded layout, then return local full-hidden states.
    # =============================================================================
    def model_forward(
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
        **kwargs: Any,
    ) -> Qwen3VLModelOutputWithPast:
        """Merge multimodal features from a ready local CP batch."""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if input_ids is None and (pixel_values is not None or pixel_values_videos is not None):
            raise ValueError("Qwen3-VL CP multimodal forward requires local input_ids.")
        if attention_mask is not None:
            raise ValueError("Qwen3-VL CP expects packed cu_seq_lens metadata instead of attention_mask.")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # ================== BEGIN CP Patch #3.1: Text Layout ==================
        # Local sequence / full hidden -> full sequence / hidden shard.
        # This gives placeholder matching a global sequence view without
        # materializing full hidden states on every CP rank.
        # =====================================================================
        inputs_embeds = self._cp_kit.gather_seq_scatter_hidden(inputs_embeds, sequence_dim=1, hidden_dim=-1)
        full_input_ids = self._cp_kit.gather_sequence(input_ids, sequence_dim=1) if input_ids is not None else None
        # =================== END CP Patch #3.1: Text Layout ===================

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None
        vision_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"}
        }

        if pixel_values is not None:
            image_outputs = self.visual(
                pixel_values.type(self.visual.dtype),
                grid_thw=image_grid_thw,
                **vision_kwargs,
            )
            image_token_count = int(image_grid_thw.to(dtype=torch.long).prod(dim=-1).sum().item()) // (
                int(self.visual.spatial_merge_size) ** 2
            )

            # ================= BEGIN CP Patch #3.2: Image Layout =================
            # Local ViT outputs -> full visual sequence / hidden shard, matching
            # the full-sequence hidden-sharded text embedding layout above.
            # =====================================================================
            image_embeds = self._cp_kit.gather_seq_scatter_hidden(
                image_outputs.pooler_output,
                sequence_dim=0,
                hidden_dim=-1,
            )
            image_embeds = image_embeds[:image_token_count]
            deepstack_image_embeds = [
                self._cp_kit.gather_seq_scatter_hidden(embeds, sequence_dim=0, hidden_dim=-1)[:image_token_count]
                for embeds in image_outputs.deepstack_features
            ]
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                full_input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            # ================== END CP Patch #3.2: Image Layout ==================

        if pixel_values_videos is not None:
            video_outputs = self.visual(
                pixel_values_videos.type(self.visual.dtype),
                grid_thw=video_grid_thw,
                **vision_kwargs,
            )
            video_token_count = int(video_grid_thw.to(dtype=torch.long).prod(dim=-1).sum().item()) // (
                int(self.visual.spatial_merge_size) ** 2
            )

            # ================= BEGIN CP Patch #3.3: Video Layout =================
            # Video follows the same local-visual to full-visual layout contract
            # as images before placeholder scatter.
            # =====================================================================
            video_embeds = self._cp_kit.gather_seq_scatter_hidden(
                video_outputs.pooler_output,
                sequence_dim=0,
                hidden_dim=-1,
            )
            video_embeds = video_embeds[:video_token_count]
            deepstack_video_embeds = [
                self._cp_kit.gather_seq_scatter_hidden(embeds, sequence_dim=0, hidden_dim=-1)[:video_token_count]
                for embeds in video_outputs.deepstack_features
            ]
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                full_input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            # ================== END CP Patch #3.3: Video Layout ==================

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
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            deepstack_visual_embeds = deepstack_video_embeds

        if visual_pos_masks is not None and deepstack_visual_embeds is not None:
            # =============== BEGIN CP Patch #3.4: Deepstack Layout ===============
            # Convert full visual deepstack states back to this rank's local
            # visual positions before handing them to the language model.
            # =====================================================================
            local_indices = self._cp_kit.local_sequence_indices(
                visual_pos_masks.shape[1],
                device=visual_pos_masks.device,
            )
            local_visual_pos_masks = visual_pos_masks.index_select(1, local_indices)
            converted_deepstack_visual_embeds = []
            for visual_embeds in deepstack_visual_embeds:
                full_visual_states = inputs_embeds.new_zeros(
                    *visual_pos_masks.shape,
                    visual_embeds.shape[-1],
                )
                full_visual_states = full_visual_states.masked_scatter(
                    visual_pos_masks[..., None].expand_as(full_visual_states),
                    visual_embeds.to(full_visual_states.device, full_visual_states.dtype),
                )
                local_visual_states = self._cp_kit.scatter_seq_gather_hidden(
                    full_visual_states,
                    sequence_dim=1,
                    hidden_dim=-1,
                )
                converted_deepstack_visual_embeds.append(local_visual_states[local_visual_pos_masks])
            deepstack_visual_embeds = converted_deepstack_visual_embeds
            visual_pos_masks = local_visual_pos_masks
            # ================ END CP Patch #3.4: Deepstack Layout ================

        # ================= BEGIN CP Patch #3.5: LLM Layout =================
        # Full sequence / hidden shard -> local sequence / full hidden.
        # The language model sees the same layout as a normal local batch.
        # =====================================================================
        inputs_embeds = self._cp_kit.scatter_seq_gather_hidden(inputs_embeds, sequence_dim=1, hidden_dim=-1)
        # ================== END CP Patch #3.5: LLM Layout ==================
        if position_ids is None:
            raise ValueError("Qwen3-VL CP train_pre_step must provide local position_ids.")

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )

    # ========================= END CP Patch #3: VLM Forward =======================

    for module in model.model.visual.modules():
        if module.__class__.__name__ == "Qwen3VLVisionAttention":
            module.forward = MethodType(vision_attention_forward, module)

    model.model.visual.forward = MethodType(visual_forward, model.model.visual)
    model.model.forward = MethodType(model_forward, model.model)
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
