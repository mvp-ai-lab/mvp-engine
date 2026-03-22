"""Model builders for the ViT image classification recipe."""

from transformers import ViTConfig
from transformers.models.vit.modeling_vit import (
    ViTForImageClassification as HFViTForImageClassification,
)

from recipes.vit_classification.configs.schema import ViTModelConfig

VIT_TP_MODULE_CONFIG: dict[str, object] = {
    "ViTSelfAttention": {
        "query": "col",
        "key": "col",
        "value": "col",
    },
    "ViTSelfOutput": {
        "dense": "row",
    },
    "ViTIntermediate": {
        "dense": "col",
    },
    "ViTOutput": {
        "dense": "row",
    },
}


def _adjust_vit_self_attention_for_tp(module, tp_mesh) -> None:
    """Keep ViT attention metadata aligned with local TP shards."""
    tp_size = tp_mesh.size()
    if tp_size <= 1 or getattr(module, "_tp_heads_adjusted", False):
        return

    if module.num_attention_heads % tp_size != 0:
        raise ValueError(
            f"ViTSelfAttention.num_attention_heads ({module.num_attention_heads}) "
            f"must be divisible by TP size ({tp_size})."
        )

    module.num_attention_heads //= tp_size
    module.all_head_size = module.num_attention_heads * module.attention_head_size
    module._tp_heads_adjusted = True


class ViTForImageClassification(HFViTForImageClassification):
    """Recipe-local ViT wrapper that exposes tensor-parallel plans."""

    TP_MODULE_CONFIG = VIT_TP_MODULE_CONFIG
    TP_MODULE_POSTPROCESSORS = {
        "ViTSelfAttention": _adjust_vit_self_attention_for_tp,
    }


def build_vit_model(model_config: ViTModelConfig) -> ViTForImageClassification:
    """Build a ViT classifier with ViT-B/16 defaults."""
    model_name = model_config.pretrained_model_name_or_path
    num_labels = model_config.num_classes

    if model_config.load_pretrained_weights:
        return ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    # Keep the template offline-friendly by constructing the same ViT-B/16
    # architecture locally when pretrained weights are not requested.
    config = ViTConfig(
        image_size=model_config.image_size,
        patch_size=model_config.patch_size,
        num_channels=model_config.num_channels,
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        hidden_dropout_prob=model_config.hidden_dropout_prob,
        attention_probs_dropout_prob=model_config.attention_dropout_prob,
        qkv_bias=True,
        num_labels=num_labels,
    )
    return ViTForImageClassification(config)
