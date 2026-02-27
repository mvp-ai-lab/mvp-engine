import sys

sys.path.insert(0, "/mnt/data-alpha-sg-01/team-camera/home/k00885418/projects/mvp-engine")
import io
from functools import partial
from typing import Optional, Tuple

import torch
from PIL import Image
from timm.data import create_transform
from transformers import AutoTokenizer

from mvp_engine.dataset.webdataset import WebDatasetBuilder

# from .augment3 import new_data_aug_generator


def _find_key(sample: dict, prefix: str) -> Optional[str]:
    """Find a key in sample that starts with the given prefix.

    Args:
        sample: Sample dict from webdataset.
        prefix: Key prefix to search for (e.g., "images.", "depths.").

    Returns:
        The matching key, or None if not found.
    """
    return next((k for k in sample.keys() if k.startswith(prefix)), None)


def decode_data(
    sample: dict,
    transform: object,
) -> Tuple[torch.Tensor, str]:
    """Process scannetpp-specific data.

    Args:
        sample: Raw sample dict from webdataset containing keys starting with
            "images." and "depths.".

    Returns:
        Tuple of (image, depth)
    """

    images_key = _find_key(sample, "jpg")
    if images_key is None:
        raise KeyError("No image key found in json (expected key starting with 'images')")

    text_key = _find_key(sample, "txt")
    if text_key is None:
        raise KeyError("No depth key found in sample (expected key starting with 'depths.')")

    image = Image.open(io.BytesIO(sample[images_key]))
    image = image.convert("RGB")

    image = transform(image)
    text = sample[text_key]

    return image, text


def collate_fn(batch, tokenizer):
    images, text = zip(*batch)
    text_ids = tokenizer(text, padding="max_length", return_tensors="pt", truncation=True)
    images = torch.stack(images, axis=0)
    return images, text_ids


def get_train_transforms(config):
    train_transform = create_transform(
        input_size=config.resize,
        is_training=True,
        color_jitter=config.color_jitter,
        auto_augment=config.auto_augment,  # use m7 for i21k
        hflip=config.hflip,
        no_aug=config.no_aug,
    )
    return train_transform


def main():
    label_path = "/mnt/data-alpha-sg-01/team-camera/shared/SigLIP/data/"
    tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
    train_transform = get_train_transforms()
    wds = WebDatasetBuilder(label_path).build(
        batch_size=16,
        make_sample_fn=partial(decode_data, transform=train_transform),
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )
    image, text = next(iter(wds))


if __name__ == "__main__":
    main()
