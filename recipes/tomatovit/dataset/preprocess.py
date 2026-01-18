import io
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2.functional import to_dtype, to_image


def _find_key(sample: dict, prefix: str) -> Optional[str]:
    """Find a key in sample that starts with the given prefix.

    Args:
        sample: Sample dict from webdataset.
        prefix: Key prefix to search for (e.g., "images.", "depths.").

    Returns:
        The matching key, or None if not found.
    """
    return next((k for k in sample.keys() if k.startswith(prefix)), None)

def _decode_image(
    image_data: Union[bytes, Image.Image],
    resize: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """Decode image from bytes or return as-is if already decoded.

    Args:
        image_data: Image bytes or PIL Image.
        resize: Optional (width, height) tuple for resizing.

    Returns:
        RGB image as a torch.Tensor with dtype float32, shape (3, H, W).
    """
    if isinstance(image_data, bytes):
        image = Image.open(io.BytesIO(image_data))
    else:
        image = image_data

    if resize is not None:
        image = image.resize(resize, Image.BILINEAR)
    image = image.convert('RGB')

    return to_dtype(to_image(image), torch.float32)

def _decode_depth(
    depth_data: Union[bytes, Image.Image],
    scale: float = 1.0,
    resize: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """Decode depth from bytes or return as-is if already decoded.

    Args:
        depth_data: Depth bytes or PIL Image.
        scale: Scale factor to apply (e.g., 1/1000 for mm to meters).
        resize: Optional (width, height) tuple for resizing.

    Returns:
        Depth map as a torch.Tensor with dtype float32, shape (1, H, W).
    """
    if isinstance(depth_data, bytes):
        depth = Image.open(io.BytesIO(depth_data))
    else:
        depth = depth_data

    if resize is not None:
        depth = depth.resize(resize, Image.NEAREST)

    depth = np.array(depth).astype(np.float32)
    if scale != 1.0:
        depth = depth * scale

    return to_dtype(to_image(depth), torch.float32)

def _decode_data(
    sample: dict,
    resize: Optional[Tuple[int, int]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process scannetpp-specific data.

    Args:
        sample: Raw sample dict from webdataset containing keys starting with
            "images." and "depths.".
        resize: Optional (width, height) tuple for resizing.

    Returns:
        Tuple of (image, depth) where both are torch.Tensors with dtype float32.
        Image has shape (3, H, W), depth has shape (1, H, W).
    """
    image_key = _find_key(sample, "images.")
    if image_key is None:
        raise KeyError("No image key found in sample (expected key starting with 'images.')")

    depth_key = _find_key(sample, "depths.")
    if depth_key is None:
        raise KeyError("No depth key found in sample (expected key starting with 'depths.')")

    image = _decode_image(sample[image_key], resize=resize)
    # Do the scaling in the engine after moving to GPU.
    depth = _decode_depth(sample[depth_key], scale=1.0, resize=resize)
    return image, depth


def make_sample(
    sample: dict,
    labels: dict,
    resize: Optional[Tuple[int, int]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Convert a decoded WebDataset sample into an (image, depth, label, meta) tuple.

    The input sample is expected to be a mapping produced by a webdataset.WebDataset
    pipeline after decoding. It must contain:

    - "meta.json": a metadata dictionary for the sample, including an "id" field.
    - An image key starting with "images." (e.g., "images.jpg", "images.png"):
      either encoded image bytes or a PIL.Image instance.
    - A depth key starting with "depths." (e.g., "depths.png", "depths.exr"):
      either encoded depth bytes or a PIL.Image instance.

    This method decodes the image and depth if they are provided as bytes,
    optionally resizes them, converts the depth map to meters (by dividing by 1000.0),
    and returns them as torch.Tensors along with the label and metadata.

    Args:
        sample: A mapping with at least the keys "meta.json", an image key
            starting with "images.", and a depth key starting with "depths.".
        labels: A dictionary mapping sample IDs to labels.
        resize: Optional (width, height) tuple for resizing both image and depth.

    Returns:
        A tuple (image, depth, label, meta) where:
        - image: RGB image as a torch.Tensor with dtype float32, shape (3, H, W).
        - depth: Depth map as a torch.Tensor with dtype float32, shape (1, H, W),
          values in meters.
        - label: Label as a torch.Tensor.
        - meta: The metadata dictionary from sample["meta.json"].
    """
    meta = sample["meta.json"]
    image, depth = _decode_data(sample, resize=resize)

    label = torch.tensor(labels[meta['id']])
    return image, depth, label, meta