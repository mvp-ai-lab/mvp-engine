"""Build a Stage-2 MIXED (image+video) mid-training Lance dataset for video_mllm.

Interleaves two row kinds at a target ratio into ONE inline Lance dataset that the
recipe consumes with ``data.modality: mixed`` (per-row dispatch by source field):

- IMAGE rows  (OpenBee): inline ``images`` bytes + computed ``img_size`` + ``conversations``
  (``<image>``). Encoded as one OneVision frame.
- VIDEO rows  (OV2 30s): ``images_source`` = relative ``.mp4`` path (online decode at train
  time, resolved against ``data.video_root``) + ``conversations`` (``<video>``).

Both kinds share the schema (nullable columns); rows are interleaved ~proportionally so a
fragment-aware shuffle mixes them. Images stay inline (kit ref-resolution can't consume
packed samples — see build_openbee_stage1_lance.py).

Example (small validation): 20 video + 100 image (= 1:5 video:image):
    python -m recipes.video_mllm.scripts.data.build_stage2_mixed_lance \
        --video-captions /.../mid_training_video/caption_v0/split_30s.jsonl \
        --video-root /.../OV2-mid-video/extracted \
        --image-parquet /.../Open-Bee/Bee-Training-Data-Stage2 \
        --n-video 20 --n-image 100 \
        --tmp ./tmp/stage2_mixed_aug_smoke --out ./tmp/stage2_mixed_lance_smoke --workers 8
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

_ROLE_MAP = {"user": "human", "assistant": "gpt", "system": "system"}

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("conversations", pa.list_(pa.struct([("from", pa.string()), ("value", pa.string())]))),
        ("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
        ("img_size", pa.list_(pa.list_(pa.int64()))),
        ("images_source", pa.list_(pa.string())),
    ]
)


def _image_size(image: dict) -> list[int]:
    """Return ``[width, height]`` for one image record, or ``[1, 1]`` if unreadable."""
    data = image.get("bytes") if isinstance(image, dict) else None
    if not isinstance(data, (bytes, bytearray, memoryview)):
        return [1, 1]
    try:
        with Image.open(io.BytesIO(bytes(data))) as decoded:
            width, height = decoded.size
        return [int(width), int(height)]
    except Exception:
        return [1, 1]


def _to_conversations(messages: list[dict]) -> list[dict]:
    """Convert OV2 role/content messages to from/value conversations with a leading <video>."""
    conv = [{"from": _ROLE_MAP[m["role"]], "value": m["content"]} for m in messages]
    first_human = next((c for c in conv if c["from"] == "human"), None)
    if first_human is None:
        raise ValueError("no human turn")
    if "<video>" not in first_human["value"]:
        first_human["value"] = "<video>\n" + first_human["value"]
    return conv


def _extracted_relpaths(video_root: Path) -> set[str]:
    """Set of extracted ``.mp4`` paths relative to video_root (for O(1) caption matching)."""
    relpaths: set[str] = set()
    for dirpath, _, files in os.walk(video_root / "30s"):
        rel_dir = Path(dirpath).relative_to(video_root)
        relpaths.update(str(rel_dir / f) for f in files if f.endswith(".mp4"))
    return relpaths


def _video_rows(captions: Path, video_root: Path, max_unique: int) -> list[dict]:
    """Collect UNIQUE OV2 rows whose extracted .mp4 exists (caption order == tar order).

    Matches against the set of extracted files and stops once all extracted videos are
    found, so it does not scan the full multi-GB caption file.
    """
    extracted = _extracted_relpaths(video_root)
    print(f"extracted videos available under {video_root}: {len(extracted)}")
    rows: list[dict] = []
    seen: set[str] = set()
    with open(captions) as handle:
        for line in handle:
            if len(rows) >= max_unique or len(rows) >= len(extracted):
                break
            rec = json.loads(line)
            sources = rec.get("images_source") or []
            src = sources[0] if sources else None
            if src is None or src in seen or src not in extracted:
                continue
            seen.add(src)
            rows.append(
                {
                    "id": rec.get("id") or src,
                    "conversations": _to_conversations(rec["messages"]),
                    "images": None,
                    "img_size": None,
                    "images_source": [src],
                }
            )
    print(f"collected {len(rows)} unique video rows")
    return rows


def _iter_image_rows(image_parquet: Path, n_image: int, pool: ThreadPoolExecutor):
    """Yield up to n_image OpenBee image rows with computed img_size."""
    if image_parquet.is_dir():
        # Recurse: Stage-2 OpenBee is nested by source (coyo/laion2B/nemotron/synthdog/ureader).
        # Shuffle deterministically so the n_image draw mixes sources instead of front-loading coyo.
        files = list(image_parquet.rglob("*.parquet"))
        random.Random(0).shuffle(files)
    else:
        files = [image_parquet]
    emitted = 0
    for parquet_file in files:
        if emitted >= n_image:
            break
        reader = pq.ParquetFile(parquet_file)
        cols = ["images", "conversations", "id"]
        has_size = "img_size" in reader.schema_arrow.names  # Stage-2 ships precomputed img_size
        if has_size:
            cols.append("img_size")
        for batch in reader.iter_batches(batch_size=256, columns=cols):
            for row in batch.to_pylist():
                if emitted >= n_image:
                    return
                if not (has_size and row.get("img_size")):
                    row["img_size"] = list(pool.map(_image_size, row.get("images") or []))
                row["images_source"] = None
                yield {k: row.get(k) for k in ("id", "conversations", "images", "img_size", "images_source")}
                emitted += 1


def build(args: argparse.Namespace) -> None:
    tmp = Path(args.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    unique_video = _video_rows(Path(args.video_captions), Path(args.video_root), args.n_video)
    if not unique_video:
        raise SystemExit("no extracted video rows found; check --video-root / --video-captions")
    n_video, n_image = args.n_video, args.n_image
    # Video rows cycle (each unique video repeated ~n_video/len) when n_video exceeds the
    # number of extracted videos; image rows stay unique.
    print(f"video repeat factor ~{n_video / len(unique_video):.2f}x ({n_video} rows / {len(unique_video)} unique)")

    out_parquet = tmp / "part-00000.parquet"
    writer = pq.ParquetWriter(out_parquet, SCHEMA)
    buf: list[dict] = []
    written = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        image_iter = _iter_image_rows(Path(args.image_parquet), n_image, pool)
        vi = ii = 0
        while vi < n_video or ii < n_image:
            take_video = ii >= n_image or (vi < n_video and vi * n_image <= ii * n_video)
            if take_video:
                buf.append(unique_video[vi % len(unique_video)])
                vi += 1
            else:
                try:
                    buf.append(next(image_iter))
                    ii += 1
                except StopIteration:
                    ii = n_image
                    continue
            if len(buf) >= 512:
                writer.write_table(pa.Table.from_pylist(buf, schema=SCHEMA))
                written += len(buf)
                buf = []
        if buf:
            writer.write_table(pa.Table.from_pylist(buf, schema=SCHEMA))
            written += len(buf)
    writer.close()
    print(f"interleaved {written} rows ({vi} video / {ii} image) -> {tmp}")

    import lance
    import pyarrow.dataset as pads

    out_lance = Path(args.out) / "samples.lance"
    out_lance.parent.mkdir(parents=True, exist_ok=True)
    reader = pads.dataset(str(tmp), format="parquet").scanner(batch_size=256).to_reader()
    lance.write_dataset(reader, str(out_lance), mode="overwrite")
    print(f"built mixed Lance dataset at {out_lance} ({written} rows; video_root resolves images_source)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-captions", required=True, help="OV2 split_<dur>.jsonl caption file.")
    parser.add_argument("--video-root", required=True, help="Dir with extracted 30s/... .mp4 (relative roots).")
    parser.add_argument("--image-parquet", required=True, help="OpenBee image parquet file or directory.")
    parser.add_argument("--n-video", type=int, required=True, help="Number of video rows.")
    parser.add_argument("--n-image", type=int, required=True, help="Number of image rows.")
    parser.add_argument("--tmp", required=True, help="Scratch dir for the interleaved parquet.")
    parser.add_argument("--out", required=True, help="Output Lance dataset directory.")
    parser.add_argument("--workers", type=int, default=8, help="Image-decode threads.")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
