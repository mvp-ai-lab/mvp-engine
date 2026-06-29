"""Build an OpenBee Stage-1 image-text alignment Lance dataset for video_mllm.

The shared OpenBee Stage-1 parquet has inline image bytes but a NULL ``img_size``
column, which the MLLM data kit's raw-row guard rejects (it requires
``len(images) == len(img_size)``). This script streams the parquet, fills
``img_size`` by decoding each image, writes augmented parquet shards into ``--tmp``,
then writes one inline-image ``samples.lance`` via ``lance.write_dataset`` (mvp_dataset
0.2.x dropped its convert CLI). The recipe reads ``<out>/samples.lance`` with
``data.ref_columns: []`` (images stay inline; the kit's post-packing ref-resolver
cannot consume packed samples).

``--tmp`` is wiped at the start of every run so stale parts cannot contaminate output.

Example (small smoke subset):
    python -m recipes.video_mllm.scripts.data.build_openbee_stage1_lance \
        --src /.../Open-Bee/Bee-Training-Data-Stage1/data \
        --tmp ./tmp/openbee_stage1_aug \
        --out ./data/Open-Bee-Lance/stage1 \
        --limit 300 --workers 8

Full build: drop ``--limit`` (~1M rows, ~54GB). Delete ``--tmp`` afterwards.
"""

from __future__ import annotations

import argparse
import io
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

KEEP_COLUMNS = ("images", "conversations", "id")


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


def _augment_batch(rows: list[dict], pool: ThreadPoolExecutor) -> list[dict]:
    """Fill ``img_size`` for one batch of rows in ``[[w, h], ...]`` order."""
    for row in rows:
        images = row.get("images") or []
        row["img_size"] = list(pool.map(_image_size, images))
    return rows


def _iter_parquet_files(src: Path, shards: int) -> list[Path]:
    files = sorted(src.glob("*.parquet")) if src.is_dir() else [src]
    if not files:
        raise SystemExit(f"no parquet files under {src}")
    return files[:shards] if shards > 0 else files


def build_augmented_parquet(src: Path, tmp: Path, *, limit: int, shards: int, workers: int) -> int:
    """Stream source parquet, fill img_size, and write augmented parquet to ``tmp``.

    ``tmp`` is wiped first so a partial/stale prior run cannot contaminate the
    later whole-directory Lance scan (or silently defeat ``--limit``).
    """
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    written = 0
    fallbacks = 0
    out_schema = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for file_index, parquet_file in enumerate(_iter_parquet_files(src, shards)):
            reader = pq.ParquetFile(parquet_file)
            out_path = tmp / f"part-{file_index:05d}.parquet"
            writer = None
            for batch in reader.iter_batches(batch_size=512, columns=list(KEEP_COLUMNS)):
                rows = _augment_batch(batch.to_pylist(), pool)
                if limit > 0 and written + len(rows) > limit:
                    rows = rows[: limit - written]
                fallbacks += sum(1 for row in rows for size in row["img_size"] if size == [1, 1])
                table = pa.Table.from_pylist(rows)
                if writer is None:
                    out_schema = table.schema
                    writer = pq.ParquetWriter(out_path, out_schema)
                writer.write_table(table.cast(out_schema))
                written += len(rows)
                if limit > 0 and written >= limit:
                    break
            if writer is not None:
                writer.close()
            if limit > 0 and written >= limit:
                break
    print(f"wrote {written} augmented rows to {tmp}")
    if fallbacks:
        print(f"WARNING: {fallbacks} image(s) had unreadable headers -> img_size=[1,1] (dropped at train-time decode)")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="OpenBee Stage-1 parquet file or directory.")
    parser.add_argument("--tmp", required=True, help="Scratch dir for img_size-augmented parquet.")
    parser.add_argument("--out", required=True, help="Output Lance dataset directory.")
    parser.add_argument("--limit", type=int, default=0, help="Max rows (0 = all).")
    parser.add_argument("--shards", type=int, default=0, help="Max parquet shards (0 = all).")
    parser.add_argument("--workers", type=int, default=8, help="Image-decode threads and convert workers.")
    args = parser.parse_args()

    rows = build_augmented_parquet(
        Path(args.src),
        Path(args.tmp),
        limit=args.limit,
        shards=args.shards,
        workers=args.workers,
    )
    if rows == 0:
        raise SystemExit("no rows written; aborting before convert")

    write_lance(Path(args.tmp), Path(args.out) / "samples.lance")
    print(f"built Lance dataset at {Path(args.out) / 'samples.lance'} (inline images)")


def write_lance(aug_parquet_dir: Path, out_lance: Path) -> None:
    """Stream augmented parquet into one inline-image Lance dataset.

    Images stay INLINE in ``samples.lance`` (no ref columns): the kit runs ref
    resolution after packing, where the lance resolver rejects MLLMPack samples,
    so the recipe consumes inline images instead (config sets data.ref_columns: []).
    Uses the ``lance`` library directly (mvp_dataset dropped its convert CLI in 0.2.x).
    """
    import lance
    import pyarrow.dataset as pads

    out_lance.parent.mkdir(parents=True, exist_ok=True)
    reader = pads.dataset(str(aug_parquet_dir), format="parquet").scanner(batch_size=512).to_reader()
    lance.write_dataset(reader, str(out_lance), mode="overwrite")


if __name__ == "__main__":
    main()
