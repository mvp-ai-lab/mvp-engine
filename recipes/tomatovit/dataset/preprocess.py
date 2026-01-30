from typing import Optional, Tuple

import numpy as np
import torch


def _find_key(sample: dict, prefix: str) -> Optional[str]:
    """Find a key in sample that starts with the given prefix.

    Args:
        sample: Sample dict from webdataset.
        prefix: Key prefix to search for (e.g., "images.", "depths.").

    Returns:
        The matching key, or None if not found.
    """
    return next((k for k in sample.keys() if k.startswith(prefix)), None)


def _decode_data(
    sample: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process scannetpp-specific data.

    Args:
        sample: Raw sample dict from webdataset containing keys starting with
            "images." and "depths.".

    Returns:
        Tuple of (image, depth)
    """
    image_key = _find_key(sample, "images.")
    if image_key is None:
        raise KeyError("No image key found in sample (expected key starting with 'images.')")

    depth_key = _find_key(sample, "depths.")
    if depth_key is None:
        raise KeyError("No depth key found in sample (expected key starting with 'depths.')")

    image = np.frombuffer(sample[image_key], dtype=np.uint8)
    depth = np.frombuffer(sample[depth_key], dtype=np.uint8)

    return image, depth


def make_sample(
    sample: dict,
    labels: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a decoded WebDataset sample into an (image, depth, label) tuple.
    The input sample is expected to be a mapping produced by a webdataset.WebDataset
    pipeline after decoding. It must contain:

    - "meta.json": a metadata dictionary for the sample, including an "id" field.
    - An image key starting with "images." (e.g., "images.jpg", "images.png"):
      either encoded image bytes or a PIL.Image instance.
    - A depth key starting with "depths." (e.g., "depths.png", "depths.exr"):
      either encoded depth bytes or a PIL.Image instance.

    This method decodes the image and depth if they are provided as bytes,
    and returns them as torch.Tensors along with the label and metadata.

    Args:
        sample: A mapping with at least the keys "meta.json", an image key
            starting with "images.", and a depth key starting with "depths.".
        labels: A dictionary mapping sample IDs to labels.

    Returns:
        A tuple (image, depth, label)
    """
    meta = sample["meta.json"]
    image, depth = _decode_data(sample)

    label = torch.tensor(labels[meta["id"]])
    return image, depth, label


def collate_fn(batch):
    images, depths, labels = zip(*batch)
    labels = torch.stack(labels)
    return images, depths, labels
