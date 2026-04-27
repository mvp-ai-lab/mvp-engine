import os
import sys
from collections import Counter
from typing import Any

import torch

_SKIP_COUNTS: Counter[str] = Counter()
_SKIP_TOTAL = 0


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


def record_skip(reason: str, sample: Any = None, detail: str | None = None, log_every: int = 10) -> None:
    """Record one skipped sample and periodically print per-process counts."""
    global _SKIP_TOTAL

    _SKIP_TOTAL += 1
    _SKIP_COUNTS[reason] += 1
    if _SKIP_TOTAL != 1 and (log_every <= 0 or _SKIP_TOTAL % log_every != 0):
        return

    counts = " ".join(f"{key}={_SKIP_COUNTS[key]}" for key in sorted(_SKIP_COUNTS))
    message = (
        "OpenBee skip stats "
        f"rank={os.getenv('RANK', '0')} "
        f"local_rank={os.getenv('LOCAL_RANK', '0')} "
        f"pid={os.getpid()} "
        f"total_skip={_SKIP_TOTAL} "
        f"counts={counts} "
        f"last_reason={reason}"
    )
    if sample is not None:
        message += f" sample={summarize_sample_for_log(sample)}"
    if detail:
        detail = detail.replace("\n", "\\n")
        if len(detail) > 500:
            detail = detail[:500] + "..."
        message += f" detail={detail}"

    print(message, file=sys.stderr, flush=True)
