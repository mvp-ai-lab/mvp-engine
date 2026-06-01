"""OneVision visual-tower swap for video MLLM strategies.

The recipe replaces Qwen3-VL's native vision tower with the OneVision encoder.
Dense strategies pass only pixels and grids; sparse codec-patch strategy also
routes selected ``patch_positions`` through the video feature path.

Apply via ``MLLMModelKit.apply_model_patches`` with a partial that binds the
encoder config, e.g.::

    from functools import partial
    apply_model_patches(model, [partial(apply_onevision_swap,
                                        vision_encoder_name_or_path=path,
                                        attn_implementation="eager",
                                        freeze_vision_encoder=True)])
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures


class OneVisionVisualTower(nn.Module):
    """Qwen3-VL visual-tower adapter backed by the OneVision encoder."""

    spatial_merge_size = 1

    def __init__(
        self,
        *,
        vision_encoder_name_or_path: str,
        output_hidden_size: int,
        attn_implementation: str,
        freeze_encoder: bool,
    ) -> None:
        """Load the OneVision encoder and a projection into the LLM hidden size."""
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            vision_encoder_name_or_path,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
        encoder_hidden_size = int(getattr(self.encoder.config, "hidden_size"))
        self.projection = nn.Linear(encoder_hidden_size, int(output_hidden_size), bias=True)

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the OneVision encoder parameters."""
        return next(self.encoder.parameters()).dtype

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor | None = None,
        patch_positions: torch.Tensor | None = None,
        **_: Any,
    ) -> BaseModelOutputWithDeepstackFeatures:
        """Encode image or codec-packed video tensors for placeholder insertion."""
        if pixel_values.dim() not in {4, 5}:
            raise ValueError("OneVisionVisualTower expects [B,C,H,W] images or [B,C,T,H,W] videos.")

        outputs = self.encoder(
            pixel_values,
            output_hidden_states=True,
            patch_positions=patch_positions,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = outputs.hidden_states[-1]
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        projected = self.projection(hidden_states)

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=projected,
            pooler_output=projected,
            deepstack_features=[],
        )


def replace_visual_tower_with_onevision(
    model,
    *,
    vision_encoder_name_or_path: str,
    attn_implementation: str,
    freeze_vision_encoder: bool,
):
    """Swap Qwen3-VL's native visual tower for the OneVision adapter."""
    vision_config = model.config.vision_config
    output_hidden_size = int(getattr(vision_config, "out_hidden_size", model.config.text_config.hidden_size))
    model.model.visual = OneVisionVisualTower(
        vision_encoder_name_or_path=vision_encoder_name_or_path,
        output_hidden_size=output_hidden_size,
        attn_implementation=attn_implementation,
        freeze_encoder=bool(freeze_vision_encoder),
    ).to(device=next(model.parameters()).device)
    return model


def patch_visual_feature_routing(model):
    """Route image/video tensors through OneVision and thread optional patch positions.

    ``get_video_features`` reads optional codec ``patch_positions`` off the hidden
    ``model.model._video_vlm_patch_positions`` attribute, which the engine sets
    per micro-batch before the forward pass.
    """

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        pixel_values = pixel_values.type(self.visual.dtype)
        kwargs.pop("return_dict", None)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        image_embeds = vision_output.pooler_output
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        vision_output.pooler_output = torch.split(image_embeds, split_sizes)
        return vision_output

    def get_video_features(self, pixel_values_videos, video_grid_thw=None, **kwargs):
        patch_positions = getattr(self, "_video_vlm_patch_positions", None)
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        kwargs.pop("return_dict", None)
        vision_output = self.visual(
            pixel_values_videos,
            grid_thw=video_grid_thw,
            patch_positions=patch_positions,
            return_dict=True,
            **kwargs,
        )
        video_embeds = vision_output.pooler_output
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        vision_output.pooler_output = torch.split(video_embeds, split_sizes)
        return vision_output

    model.model.get_image_features = MethodType(get_image_features, model.model)
    model.model.get_video_features = MethodType(get_video_features, model.model)
    return model


def apply_onevision_swap(
    model,
    *,
    vision_encoder_name_or_path: str,
    attn_implementation: str = "eager",
    freeze_vision_encoder: bool = True,
):
    """Recipe model patch: swap in OneVision and wire optional patch-position routing."""
    model = replace_visual_tower_with_onevision(
        model,
        vision_encoder_name_or_path=vision_encoder_name_or_path,
        attn_implementation=attn_implementation,
        freeze_vision_encoder=freeze_vision_encoder,
    )
    model = patch_visual_feature_routing(model)
    return model
