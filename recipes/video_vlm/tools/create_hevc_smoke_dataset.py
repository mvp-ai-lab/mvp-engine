#!/usr/bin/env python3
"""Create tiny synthetic H.264/H.265 parquet data for Video VLM codec smoke tests."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def create_codec_video(video_path: Path, *, codec: str, duration: float, size: int, fps: int) -> None:
    """Generate a deterministic yuv420p H.264 or H.265 clip with ffmpeg."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if codec == "h264":
        codec_args = ["-c:v", "libx264", "-preset", "veryfast"]
    elif codec == "h265":
        codec_args = ["-c:v", "libx265", "-x265-params", "log-level=error"]
    else:
        raise ValueError("codec must be 'h264' or 'h265'.")

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={size}x{size}:rate={fps}:duration={duration}",
        *codec_args,
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    subprocess.run(command, check=True)


def write_parquet(output_dir: Path, video_paths: list[tuple[str, Path]]) -> Path:
    """Write Video VLM parquet rows pointing at generated codec clips."""
    rows = []
    for codec, video_path in video_paths:
        rows.append(
            {
                "id": f"codec_smoke_{codec}",
                "messages": [
                    {"role": "user", "content": f"<video>\nDescribe the {codec.upper()} motion pattern."},
                    {"role": "assistant", "content": "A synthetic test pattern moves across the frame."},
                ],
                "images": [],
                "image_size": [],
                "videos": [str(video_path.resolve())],
            }
        )
    output_path = output_dir / "codec_smoke.parquet"
    pq.write_table(pa.Table.from_pylist(rows), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="./data/video_vlm/codec_smoke",
        help="Directory where the synthetic codec clips and parquet are written.",
    )
    parser.add_argument("--duration", type=float, default=2.0, help="Synthetic clip duration in seconds.")
    parser.add_argument("--size", type=int, default=224, help="Synthetic square video size.")
    parser.add_argument("--fps", type=int, default=8, help="Synthetic clip frame rate.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    video_dir = output_dir / "media" / "videos"
    video_paths = [
        ("h264", video_dir / "synthetic_h264.mp4"),
        ("h265", video_dir / "synthetic_h265.mp4"),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    for codec, video_path in video_paths:
        create_codec_video(video_path, codec=codec, duration=args.duration, size=args.size, fps=args.fps)
    parquet_path = write_parquet(output_dir, video_paths)

    for codec, video_path in video_paths:
        print(f"{codec.upper()} smoke video written to: {video_path}")
    print(f"Codec smoke parquet written to: {parquet_path}")


if __name__ == "__main__":
    main()
