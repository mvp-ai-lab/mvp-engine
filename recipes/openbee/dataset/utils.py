from typing import Any

import torch


def summarize_sample_for_log(sample: Any) -> str:
    """Return compact sample metadata for warning logs."""
    if not isinstance(sample, dict):
        return repr(sample)

    summary: dict[str, Any] = {}
    for key in ("id", "source", "__source__", "__key__", "__global_index__"):
        if key in sample:
            summary[key] = sample[key]

    messages = sample.get("messages") or sample.get("conversations")
    if isinstance(messages, list):
        summary["message_count"] = len(messages)

    images = sample.get("images")
    if isinstance(images, (list, tuple)):
        summary["image_count"] = len(images)
    elif images is not None:
        summary["image_count"] = 1

    image_size = sample.get("image_size")
    if image_size is None:
        image_size = sample.get("img_size")
    if isinstance(image_size, (list, tuple)):
        summary["image_size_count"] = len(image_size)
    elif image_size is not None:
        summary["image_size_count"] = 1

    input_ids = sample.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        summary["input_ids_shape"] = tuple(input_ids.shape)

    return repr(summary)
