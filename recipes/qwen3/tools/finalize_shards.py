"""Combine per-shard Lance tables into one dataset by writing a dataset-level
``meta.json`` — but only after PROVING the shards form an exact, non-overlapping
cover of the original input.

Each shard (written by ``convert_parquet_to_lance.py --num-shards/--shard-index``)
drops a ``shard_manifest.json`` next to its ``text.lance``. This script reads all
of them and refuses to publish unless:
  * every shard agrees on input / input_format / total_files / num_shards
  * shard indices are exactly ``0 .. num_shards-1`` (none missing, none extra)
  * the union of selected files equals total_files with NO overlap (exact cover)
  * every shard's ``text.lance`` opens and is non-empty

Only then does it atomically write ``<dataset_dir>/meta.json`` listing the shards
(relative ``shard_NN/text.lance`` paths), which ``mvp_dataset`` reads as one
concatenated dataset. Refuses to overwrite an existing meta.json.

Usage:
    python finalize_shards.py --dataset-dir /mnt/.../lance/allenai_dolma3_mix-6T-1025-7B
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import lance


def main() -> None:
    parser = argparse.ArgumentParser(description="combine verified shards into a dataset meta.json")
    parser.add_argument("--dataset-dir", required=True, help="lance/<dataset> dir holding shard_NN/ subdirs")
    args = parser.parse_args()

    dataset_dir = os.path.normpath(args.dataset_dir)
    meta_path = os.path.join(dataset_dir, "meta.json")
    if os.path.exists(meta_path):
        raise SystemExit(f"refusing to overwrite existing {meta_path}")

    shard_dirs = sorted(d for d in glob.glob(os.path.join(dataset_dir, "shard_*")) if os.path.isdir(d))
    if not shard_dirs:
        raise SystemExit(f"no shard_* dirs under {dataset_dir}")

    manifests = {}
    for d in shard_dirs:
        mpath = os.path.join(d, "shard_manifest.json")
        if not os.path.exists(mpath):
            raise SystemExit(f"missing shard_manifest.json in {d}")
        with open(mpath, encoding="utf-8") as handle:
            manifests[d] = json.load(handle)

    # 1) all shards must agree on the global split parameters AND on the digest
    #    of the full input file list (proves every shard saw the identical input,
    #    not just the same file count).
    first = next(iter(manifests.values()))
    num_shards = first["num_shards"]
    for d, m in manifests.items():
        for key in ("input", "input_format", "total_files", "full_files_digest", "num_shards"):
            if m[key] != first[key]:
                raise SystemExit(f"manifest mismatch on '{key}': {d}={m[key]} vs {first[key]}")

    # 2) shard indices must be exactly 0..num_shards-1
    indices = sorted(m["shard_index"] for m in manifests.values())
    if indices != list(range(num_shards)):
        raise SystemExit(f"shard indices {indices} != 0..{num_shards - 1} (missing or extra shards)")
    if len(shard_dirs) != num_shards:
        raise SystemExit(f"found {len(shard_dirs)} shard dirs but num_shards={num_shards}")

    # 3) union of selected files must be an exact, non-overlapping cover
    seen: set[str] = set()
    total = 0
    for d, m in manifests.items():
        files = m["files"]
        if len(files) != m["selected_files"]:
            raise SystemExit(f"{d}: selected_files={m['selected_files']} but listed {len(files)} files")
        for f in files:
            if f in seen:
                raise SystemExit(f"file covered by more than one shard: {f}")
            seen.add(f)
        total += len(files)
    if total != first["total_files"] or len(seen) != first["total_files"]:
        raise SystemExit(f"union {len(seen)} (sum {total}) != total_files {first['total_files']} — not an exact cover")

    # 4) every shard's lance opens and is non-empty
    shards_rel = []
    for d in shard_dirs:
        lance_path = os.path.join(d, "text.lance")
        if lance.dataset(lance_path).count_rows() <= 0:
            raise SystemExit(f"{lance_path} is empty")
        shards_rel.append(os.path.join(os.path.basename(d), "text.lance"))

    # 5) atomically publish meta.json. Unique tmp (no concurrent-finalizer clash)
    #    + os.link, which fails if meta.json already exists — a no-overwrite atomic
    #    publish that also closes the exists()-check-to-write race above.
    tmp_path = os.path.join(dataset_dir, f".meta.json.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump({"shards": shards_rel}, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp_path, meta_path)
    except FileExistsError:
        raise SystemExit(f"meta.json was published concurrently: {meta_path}")
    finally:
        os.unlink(tmp_path)

    print(f"verified {num_shards} shards cover {first['total_files']} files with no overlap")
    print(f"meta.json -> {meta_path}   (set data.train_path to this)")
    for s in shards_rel:
        print(f"  - {s}")


if __name__ == "__main__":
    main()
