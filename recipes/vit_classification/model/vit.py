"""Model builders for the ViT image classification recipe."""

from omegaconf import DictConfig
from transformers import ViTConfig, ViTForImageClassification


def build_vit_model(model_config: DictConfig) -> ViTForImageClassification:
    """Build a ViT classifier with ViT-B/16 defaults."""
    model_name = model_config.pretrained_model_name_or_path
    num_labels = int(model_config.num_classes)

    if model_config.load_pretrained_weights:
        return ViTForImageClassification.from_pretrained(
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
    return ViTForImageClassification(config)
