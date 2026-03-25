"""Model builders for the ViT image classification recipe."""

from types import MethodType

from transformers import ViTConfig, ViTForImageClassification

from ..configs.schema import ViTModelConfig


def inject_model_flops_calculation(model: ViTForImageClassification) -> ViTForImageClassification:
    """Inject a per-step FLOPs estimator onto the model instance."""

    def calculate_model_flops(
        self: ViTForImageClassification,
        *,
        batch_size: int,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        is_training: bool = True,
    ) -> float:
        if isinstance(image_size, int):
            image_h, image_w = image_size, image_size
        else:
            image_h, image_w = map(int, image_size)

        if isinstance(patch_size, int):
            patch_h, patch_w = patch_size, patch_size
        else:
            patch_h, patch_w = map(int, patch_size)

        batch = int(batch_size)
        if min(batch, image_h, image_w, patch_h, patch_w) <= 0:
            raise ValueError("batch_size, image_size, and patch_size must be > 0")
        if image_h % patch_h != 0 or image_w % patch_w != 0:
            raise ValueError("image_size must be divisible by patch_size")

        num_patches = (image_h // patch_h) * (image_w // patch_w)
        seq_len = num_patches + 1

        channels = int(getattr(self.config, "num_channels", 3))
        hidden = int(self.config.hidden_size)
        layers = int(self.config.num_hidden_layers)
        intermediate = int(self.config.intermediate_size)
        num_labels = int(getattr(self.config, "num_labels", 1000))

        patch_dim = channels * patch_h * patch_w
        patch_embed_flops = 2 * batch * num_patches * patch_dim * hidden

        qkv_flops = 6 * batch * seq_len * hidden * hidden
        attention_scores_flops = 2 * batch * seq_len * seq_len * hidden
        attention_weighted_sum_flops = 2 * batch * seq_len * seq_len * hidden
        attention_out_flops = 2 * batch * seq_len * hidden * hidden
        mlp_flops = 4 * batch * seq_len * hidden * intermediate
        block_flops = (
            qkv_flops + attention_scores_flops + attention_weighted_sum_flops + attention_out_flops + mlp_flops
        )

        head_flops = 2 * batch * hidden * num_labels
        forward_flops = float(patch_embed_flops + layers * block_flops + head_flops)

        # Training FLOPs are approximated as forward + backward + parameter-gradient work.
        return forward_flops * 3.0 if is_training else forward_flops

    model.calculate_model_flops = MethodType(calculate_model_flops, model)
    return model


def build_vit_model(model_config: ViTModelConfig) -> ViTForImageClassification:
    """Build a ViT classifier with ViT-B/16 defaults."""
    model_name = model_config.pretrained_model_name_or_path
    num_labels = model_config.num_classes

    if model_config.load_pretrained_weights:
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        return inject_model_flops_calculation(model)

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
    return inject_model_flops_calculation(ViTForImageClassification(config))
