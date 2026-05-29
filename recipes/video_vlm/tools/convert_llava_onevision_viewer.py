#!/usr/bin/env python3
"""Convert tiny LLaVA-OneVision-2 viewer parquet files into Video VLM demo parquet."""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

USER_RE = re.compile(r"\[user\]\s*(.*?)(?=\s*\[assistant\]|\Z)", re.DOTALL | re.IGNORECASE)
ASSISTANT_RE = re.compile(r"\[assistant\]\s*(.*)", re.DOTALL | re.IGNORECASE)


def parse_preview(preview: Any) -> tuple[str, str]:
    """Split the viewer preview transcript into user and assistant messages."""
    text = "" if preview is None else str(preview)
    user_match = USER_RE.search(text)
    assistant_match = ASSISTANT_RE.search(text)
    user = user_match.group(1).strip() if user_match else text.strip()
    assistant = assistant_match.group(1).strip() if assistant_match else "This is a preview sample."
    user = user.replace("<image>", "").replace("<video>", "").strip()
    return user, assistant


def media_bytes(value: Any) -> tuple[bytes | None, str | None]:
    """Extract optional bytes/path from HF Image or Video parquet cell values."""
    if value is None:
        return None, None
    if isinstance(value, dict):
        raw = value.get("bytes")
        path = value.get("path")
        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        if isinstance(raw, bytearray):
            raw = bytes(raw)
        return (raw if isinstance(raw, bytes) else None), (path if isinstance(path, str) else None)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value), None
    if isinstance(value, str):
        return None, value
    return None, None


def write_image(value: Any, output_dir: Path, stem: str) -> tuple[str, list[int]]:
    """Write one embedded image to disk and return relative path plus size metadata."""
    raw, path = media_bytes(value)
    image_dir = output_dir / "media" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if raw is None and path is not None:
        resolved = Path(path).expanduser()
        if resolved.is_file():
            with Image.open(resolved) as opened:
                width, height = opened.size
            return str(resolved), [int(width), int(height)]
        raise ValueError(f"image row references a path that is not present locally: {path}")
    if raw is None:
        raise ValueError("image row does not contain bytes or a usable path.")

    with Image.open(io.BytesIO(raw)) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        rel_path = Path("media") / "images" / f"{stem}.png"
        image.save(output_dir / rel_path)
    return str((output_dir / rel_path).resolve()), [int(width), int(height)]


def write_video(value: Any, output_dir: Path, stem: str) -> str:
    """Write one embedded video to disk and return a relative path."""
    raw, path = media_bytes(value)
    video_dir = output_dir / "media" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    if raw is None and path is not None:
        resolved = Path(path).expanduser()
        if resolved.is_file():
            return str(resolved)
        raise ValueError(f"video row references a path that is not present locally: {path}")
    if raw is None:
        raise ValueError("video row does not contain bytes or a usable path.")

    suffix = ".mp4"
    if path:
        candidate_suffix = Path(path).suffix
        if candidate_suffix:
            suffix = candidate_suffix
    rel_path = Path("media") / "videos" / f"{stem}{suffix}"
    (output_dir / rel_path).write_bytes(raw)
    return str((output_dir / rel_path).resolve())


def convert_spatial(parquet_path: Path, output_dir: Path, limit: int) -> list[dict[str, Any]]:
    """Convert image/spatial preview rows."""
    rows = pq.read_table(parquet_path).to_pylist()
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows[:limit]):
        user, assistant = parse_preview(row.get("preview"))
        image_path, image_size = write_image(row.get("image"), output_dir, f"spatial_{index:04d}")
        converted.append(
            {
                "id": str(row.get("id", f"spatial_{index:04d}")),
                "messages": [
                    {"role": "user", "content": f"<image>\n{user}"},
                    {"role": "assistant", "content": assistant},
                ],
                "images": [image_path],
                "image_size": [image_size],
                "videos": [],
            }
        )
    return converted


def convert_caption(parquet_path: Path, output_dir: Path, limit: int) -> list[dict[str, Any]]:
    """Convert video caption preview rows."""
    rows = pq.read_table(parquet_path).to_pylist()
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows[:limit]):
        user, assistant = parse_preview(row.get("preview"))
        video_path = write_video(row.get("video"), output_dir, f"{parquet_path.stem}_{index:04d}")
        converted.append(
            {
                "id": f"{parquet_path.stem}_{index:04d}",
                "messages": [
                    {"role": "user", "content": f"<video>\n{user}"},
                    {"role": "assistant", "content": assistant},
                ],
                "images": [],
                "image_size": [],
                "videos": [video_path],
            }
        )
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing the downloaded viewer/*.parquet files."
    )
    parser.add_argument("--output-dir", required=True, help="Directory where demo parquet and media files are written.")
    parser.add_argument("--image-limit", type=int, default=2, help="Maximum spatial/image rows to convert.")
    parser.add_argument("--video-limit", type=int, default=1, help="Maximum video rows per caption parquet to convert.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    spatial_path = input_dir / "viewer" / "spatial.parquet"
    if spatial_path.is_file():
        rows.extend(convert_spatial(spatial_path, output_dir, args.image_limit))

    caption_paths = sorted((input_dir / "viewer").glob("caption_*.parquet"))
    for caption_path in caption_paths:
        rows.extend(convert_caption(caption_path, output_dir, args.video_limit))

    if not rows:
        raise FileNotFoundError(f"No viewer parquet files found under {input_dir / 'viewer'}")

    output_path = output_dir / "demo.parquet"
    pq.write_table(pa.Table.from_pylist(rows), output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
