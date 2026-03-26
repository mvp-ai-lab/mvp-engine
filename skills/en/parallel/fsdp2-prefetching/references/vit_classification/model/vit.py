"""ViT FSDP2 prefetching reference snippet.

This is a focused reference for the skill, not a runtime module imported by the repo.
"""

import torch.nn as nn
from transformers.models.vit.modeling_vit import (
    ViTForImageClassification as HFViTForImageClassification,
)


def setup_vit_fsdp2_prefetching(model: nn.Module) -> None:
    """Wire forward/backward prefetch edges for a sequential ViT encoder."""
    if getattr(model, "_fsdp2_prefetching_configured", False):
        return

    encoder_layers = list(model.vit.encoder.layer)

    for layer_idx, layer in enumerate(encoder_layers):
        if layer_idx + 1 < len(encoder_layers):
            next_layer = encoder_layers[layer_idx + 1]
            layer.set_modules_to_forward_prefetch([next_layer])

        if layer_idx > 0:
            prev_layer = encoder_layers[layer_idx - 1]
            layer.set_modules_to_backward_prefetch([prev_layer])

    model._fsdp2_prefetching_configured = True


class ViTForImageClassification(HFViTForImageClassification):
    FSDP2_PREFETCHING = setup_vit_fsdp2_prefetching
