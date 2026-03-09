from transformers import ViTConfig, ViTForImageClassification

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


class TemplateViTForImageClassification(ViTForImageClassification):
    """HF ViT classifier with local tensor-parallel metadata."""

    TP_MODULE_CONFIG = VIT_TP_MODULE_CONFIG


def prepare_vit_model_for_tp(model: TemplateViTForImageClassification, tp_size: int) -> None:
    """Patch HF ViT attention metadata so local TP shards reshape correctly."""
    if tp_size <= 1:
        return

    for module in model.modules():
        if module.__class__.__name__ != "ViTSelfAttention":
            continue
        if getattr(module, "_tp_prepared", False):
            continue
        if module.num_attention_heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads({module.num_attention_heads}) must be divisible by tp_size({tp_size})."
            )

        module.num_attention_heads //= tp_size
        module.all_head_size = module.num_attention_heads * module.attention_head_size
        module._tp_prepared = True


def build_vit_model(model_config) -> TemplateViTForImageClassification:
    """Build a ViT classifier with ViT-B/16 defaults."""
    model_name = model_config.pretrained_model_name_or_path
    num_labels = int(model_config.num_classes)

    if model_config.load_pretrained_weights:
        return TemplateViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    # Keep the template offline-friendly by constructing the same ViT-B/16
    # architecture locally when pretrained weights are not requested.
    config = ViTConfig(
        image_size=int(model_config.image_size),
        patch_size=int(getattr(model_config, "patch_size", 16)),
        num_channels=int(getattr(model_config, "num_channels", 3)),
        hidden_size=int(getattr(model_config, "hidden_size", 768)),
        intermediate_size=int(getattr(model_config, "intermediate_size", 3072)),
        num_hidden_layers=int(getattr(model_config, "num_hidden_layers", 12)),
        num_attention_heads=int(getattr(model_config, "num_attention_heads", 12)),
        hidden_dropout_prob=float(model_config.hidden_dropout_prob),
        attention_probs_dropout_prob=float(model_config.attention_dropout_prob),
        qkv_bias=True,
        num_labels=num_labels,
    )
    return TemplateViTForImageClassification(config)
