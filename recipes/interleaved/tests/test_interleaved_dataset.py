"""Tests for interleaved recipe data adapters."""

import io

from PIL import Image

from recipes.interleaved.dataset import (
    InterleavedDataGuard,
    InterleavedSampleKit,
    infer_image_size,
)


def test_single_user_content_blocks_become_pretrain_response():
    """Single-message interleaved rows follow the LLaMA-Factory pretrain convention."""
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image", "image_file": {"image": {"bytes": _image_bytes(), "path": None}}},
                    {"type": "text", "text": "after"},
                ],
            }
        ]
    }

    canonical = InterleavedSampleKit().normalize(sample)

    assert canonical.messages[0] == {"role": "user", "content": []}
    assert canonical.messages[1]["role"] == "assistant"
    assert canonical.messages[1]["content"][0] == {"type": "text", "text": "before"}
    assert canonical.messages[1]["content"][1]["type"] == "image"
    assert canonical.messages[1]["content"][2] == {"type": "text", "text": "after"}
    assert len(canonical.media) == 1
    assert canonical.media[0].size is None


def test_inline_image_size_fallback_from_bytes():
    """Inline image records can provide dimensions without top-level image_size metadata."""
    assert infer_image_size({"bytes": _image_bytes(), "path": None}) == [5, 7]


def test_data_guard_accepts_inline_interleaved_without_top_level_images():
    """OpenAI content-block rows do not need top-level images before normalization."""
    guard = InterleavedDataGuard(check_basic_formats=True, check_image_sizes=True)
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image", "image_file": {"image": {"bytes": _image_bytes(), "path": None}}},
                ],
            }
        ]
    }

    assert guard.check(sample).is_valid


def _image_bytes() -> bytes:
    image = Image.new("RGB", (7, 5), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
