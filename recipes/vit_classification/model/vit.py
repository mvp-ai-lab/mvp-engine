"""Model builders for the ViT image classification recipe."""

from transformers import ViTConfig, ViTForImageClassification

from ..vit_classification.configs.schema import ViTModelConfig


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
