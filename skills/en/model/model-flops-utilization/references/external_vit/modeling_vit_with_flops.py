from __future__ import annotations

from transformers import ViTForImageClassification


class ViTForImageClassificationWithFlops(ViTForImageClassification):
    """Reference adapter that adds the MFU FLOPs contract to an external ViT."""

    def calculate_model_flops(
        self,
        *,
        batch_size: int,
        seq_len: int | None = None,
        image_size: int | tuple[int, int] | None = None,
        patch_size: int | tuple[int, int] | None = None,
        is_training: bool = True,
    ) -> float:
        if image_size is None or patch_size is None:
            raise ValueError("ViT FLOPs requires image_size and patch_size.")

        batch = int(batch_size)
        if isinstance(image_size, int):
            img_h, img_w = image_size, image_size
        else:
            img_h, img_w = map(int, image_size)
        if isinstance(patch_size, int):
            patch_h, patch_w = patch_size, patch_size
        else:
            patch_h, patch_w = map(int, patch_size)

        if min(batch, img_h, img_w, patch_h, patch_w) <= 0:
            raise ValueError("batch_size, image_size, and patch_size must be > 0")
        if img_h % patch_h != 0 or img_w % patch_w != 0:
            raise ValueError("image_size must be divisible by patch_size")

        num_patches = (img_h // patch_h) * (img_w // patch_w)
        channels = int(getattr(self.config, "num_channels", 3))
        hidden = int(self.config.hidden_size)
        layers = int(self.config.num_hidden_layers)
        intermediate = int(self.config.intermediate_size)
        labels = int(getattr(self.config, "num_labels", 1000))

        patch_embed_flops = 2 * batch * num_patches * (channels * patch_h * patch_w) * hidden
        block_flops = (
            8 * batch * num_patches * hidden * hidden
            + 4 * batch * num_patches * num_patches * hidden
            + 4 * batch * num_patches * hidden * intermediate
        )
        backbone_flops = layers * block_flops
        head_flops = 2 * batch * hidden * labels

        forward_flops = float(patch_embed_flops + backbone_flops + head_flops)
        return forward_flops * 3.0 if is_training else forward_flops
