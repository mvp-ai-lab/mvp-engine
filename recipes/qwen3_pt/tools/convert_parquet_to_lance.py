"""Convert raw HF datasets (parquet / arrow IPC / jsonl[.gz|.zst]) into an
HKdata-style ``text.lance`` table.

Why this exists: our datasets land on the cluster as raw HuggingFace files, but
the ``qwen3_pt`` recipe reads Lance (``text_field="data"``). This script *streams*
the source files, maps the document text onto the ``data`` column, drops empty
rows, and writes a Lance table with the HKdata text schema
(``id / info / data / tags``). It also writes a sibling ``meta.json`` so
``data.train_path`` can point straight at it.

Supported inputs (auto-detected, or forced with ``--input-format``):
  * parquet        — e.g. starcoderdata / fineweb-edu / dclm-parquet
  * arrow          — HF ``datasets`` IPC files (e.g. the-pile-splitted)
  * jsonl(.gz/.zst) — e.g. pile-uncopyrighted / dolma3

If the text column is a *list of pages* (e.g. institutional-books
``text_by_page_gen``: ``list<large_string>``), pages are joined with a newline
into one document per row.

It produces a *reader-compatible subset* of an HKdata Warehouse dataset (the
``text.lance`` table + a ``meta.json`` shard list) — not the full warehouse
layout (no ``manifest.yaml`` / versioning), which ``mvp_dataset`` does not need.

Dependencies (``lance``/``pylance`` 7.0.0, ``pyarrow``) come with the cluster
``mvp_engine`` env via ``mvp_dataset``; run inside that env, on a CPU compute
node, never the login node. For a big run pass ``--expected-files`` /
``--expected-rows`` so a silent under-scan fails loudly::

    python recipes/qwen3_pt/tools/convert_parquet_to_lance.py \
        --input  /mnt/.../text/bigcode_starcoderdata \
        --output /mnt/.../text/lance/.staging/bigcode_starcoderdata/text.lance \
        --expected-files 863 --expected-rows 206642239
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import io
import json
import os
import sys
import time

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads

# Column holding the document text, tried in order when --text-column is unset.
TEXT_COLUMN_CANDIDATES = ("content", "text", "data")

# How many books/pages-rows per scan batch: page-list rows are huge (a whole
# book each), so big parquet batches would balloon memory.
LIST_COLUMN_MAX_BATCH = 512

# Lines per record batch on the jsonl path (docs are read one line at a time).
JSONL_BATCH_ROWS = 8192

# HKdata text.lance schema: id / info / data / tags (the text body lives in `data`).
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("info", pa.string()),
        pa.field("data", pa.large_string()),
        pa.field("tags", pa.list_(pa.string())),
    ]
)

FORMAT_GLOBS = {
    "parquet": ("*.parquet",),
    "arrow": ("*.arrow",),
    "jsonl": ("*.jsonl", "*.jsonl.gz", "*.jsonl.zst"),
}


def detect_input(input_path: str, format_override: str | None) -> tuple[str, list[str]]:
    """Detect the input format and list its files.

    A directory may contain exactly one format family unless --input-format
    picks one explicitly; mixing formats without a choice is an error.
    """
    if os.path.isfile(input_path):
        low = input_path.lower()
        for fmt, patterns in FORMAT_GLOBS.items():
            if any(low.endswith(pattern.lstrip("*")) for pattern in patterns):
                if format_override and format_override != fmt:
                    raise SystemExit(f"--input-format {format_override} does not match file type of {input_path}")
                return fmt, [input_path]
        raise SystemExit(f"Unrecognized input file type: {input_path}")

    found: dict[str, list[str]] = {}
    for fmt, patterns in FORMAT_GLOBS.items():
        files: set[str] = set()
        for pattern in patterns:
            files.update(glob.glob(os.path.join(input_path, "**", pattern), recursive=True))
        if files:
            found[fmt] = sorted(files)

    if format_override:
        if format_override not in found:
            raise SystemExit(f"No {format_override} files found under {input_path} (found: {sorted(found)})")
        return format_override, found[format_override]
    if not found:
        raise SystemExit(f"No parquet/arrow/jsonl files found under {input_path}")
    if len(found) > 1:
        raise SystemExit(f"Multiple formats under {input_path}: {sorted(found)} — pass --input-format.")
    return next(iter(found.items()))


def pick_text_column(column_names: list[str], override: str | None) -> str:
    """Choose which source column/key holds the text body."""
    if override:
        if override not in column_names:
            raise ValueError(f"--text-column '{override}' not in source columns {column_names}")
        return override
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in column_names:
            return candidate
    raise ValueError(f"No text column auto-detected in {column_names}; pass --text-column.")


def jsonl_compression(path: str) -> str | None:
    """Map a jsonl file extension to its pyarrow decompression codec."""
    low = path.lower()
    if low.endswith(".gz"):
        return "gzip"
    if low.endswith(".zst"):
        return "zstd"
    return None


def arrow_schema_names(path: str) -> list[str]:
    """Read the schema of one Arrow IPC file (stream format, file as fallback)."""
    with pa.memory_map(path, "r") as source:
        try:
            return pa.ipc.open_stream(source).schema.names
        except pa.ArrowInvalid:
            source.seek(0)
            return pa.ipc.open_file(source).schema.names


def jsonl_first_doc_keys(path: str) -> list[str]:
    """Return the keys of the first parseable JSON object in one jsonl file.

    Broken lines are skipped (same drop semantics as the streaming reader).
    """
    skipped = 0
    with pa.input_stream(path, compression=jsonl_compression(path)) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    doc = json.loads(line)
                except ValueError:
                    skipped += 1
                    continue
                if isinstance(doc, dict):
                    if skipped:
                        print(f"WARNING: skipped {skipped} broken line(s) detecting keys in {path}", flush=True)
                    return list(doc.keys())
                skipped += 1
    raise SystemExit(f"{path} contains no parseable JSON objects.")


def iter_arrow_batches(files: list[str], text_column: str):
    """Stream record batches from Arrow IPC files, keeping only the text column."""
    for path in files:
        with pa.memory_map(path, "r") as source:
            try:
                reader = pa.ipc.open_stream(source)
                batches = iter(reader)
            except pa.ArrowInvalid:
                source.seek(0)
                file_reader = pa.ipc.open_file(source)
                batches = (file_reader.get_batch(i) for i in range(file_reader.num_record_batches))
            for batch in batches:
                yield batch.select([text_column])


def iter_jsonl_batches(files: list[str], text_column: str, counters: dict):
    """Stream record batches from jsonl(.gz/.zst) files, one doc per line.

    Docs missing the text key or with broken JSON become nulls (dropped by the
    downstream empty-text filter); broken lines are also tallied in
    ``counters["bad_json"]``.
    """
    for path in files:
        with pa.input_stream(path, compression=jsonl_compression(path), buffer_size=1 << 20) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="") as handle:
                rows: list[str | None] = []
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        doc = json.loads(line)
                    except ValueError:
                        doc = None
                    if isinstance(doc, dict):
                        value = doc.get(text_column)
                    else:  # broken JSON, or valid JSON that is not an object
                        counters["bad_json"] += 1
                        value = None
                    rows.append(value if isinstance(value, str) else None)
                    if len(rows) >= JSONL_BATCH_ROWS:
                        yield pa.record_batch([pa.array(rows, type=pa.large_string())], names=[text_column])
                        rows = []
                if rows:
                    yield pa.record_batch([pa.array(rows, type=pa.large_string())], names=[text_column])


def limit_batches(batches, limit: int):
    """Pass batches through until ``limit`` rows have been emitted (for smoke runs)."""
    remaining = limit
    for batch in batches:
        if remaining <= 0:
            break
        if batch.num_rows > remaining:
            batch = batch.slice(0, remaining)
        yield batch
        remaining -= batch.num_rows


def join_pages(column: pa.Array) -> pa.Array:
    """Join a list-of-pages column into one newline-separated string per row.

    Plain Python on purpose: page-list datasets are small (books ≈ 1M rows), and
    Arrow's binary_join nulls out a whole row when any page is null.
    """
    joined = ["\n".join(page or "" for page in pages) if pages is not None else None for pages in column.to_pylist()]
    return pa.array(joined, type=pa.large_string())


def to_hkdata_batches(source_batches, text_column, id_column, info_json, tag, counters, log_every, id_prefix=""):
    """Map raw source batches to HKdata-schema batches, dropping empty text rows.

    Emits a periodic ``[progress]`` line every ``log_every`` scanned rows so a long
    run is observable in the job log (rate lets us catch a stall or wrong ETA early).
    ``id_prefix`` (e.g. ``"3-"`` for shard 3) keeps row-index ids unique across shards.
    """
    one_row_tags = [tag] if tag else []
    next_row_id = 0
    start = time.monotonic()
    next_log = log_every
    for batch in source_batches:
        text = batch.column(text_column)
        if pa.types.is_list(text.type) or pa.types.is_large_list(text.type):
            text = join_pages(text)
        # Keep only non-null, non-blank text (matches what the recipe's DataGuard accepts).
        trimmed_len = pc.utf8_length(pc.utf8_trim_whitespace(text))
        keep = pc.and_(pc.is_valid(text), pc.fill_null(pc.greater(trimmed_len, 0), False))

        if id_column:
            ids = batch.column(id_column).cast(pa.string())
        else:
            ids = pa.array([f"{id_prefix}{next_row_id + i}" for i in range(batch.num_rows)], type=pa.string())
        next_row_id += batch.num_rows
        counters["seen"] += batch.num_rows

        if log_every and counters["seen"] >= next_log:
            next_log = counters["seen"] + log_every
            elapsed = time.monotonic() - start
            rate = counters["seen"] / elapsed if elapsed > 0 else 0.0
            print(
                f"[progress] seen={counters['seen']:,} kept={counters['kept']:,} "
                f"{rate:,.0f} rows/s elapsed={elapsed:.0f}s",
                flush=True,
            )

        data = pc.filter(text, keep).cast(pa.large_string())
        kept = len(data)
        if kept == 0:
            continue
        counters["kept"] += kept
        ids = pc.filter(ids, keep)
        info = pa.array([info_json] * kept, type=pa.string())
        tags = pa.array([one_row_tags] * kept, type=pa.list_(pa.string()))
        yield pa.record_batch([ids, info, data, tags], schema=OUTPUT_SCHEMA)


def peek_first_batch(batch_iter):
    """Pull the first batch eagerly so all-empty input never writes a file.

    Returns a generator that re-emits the first batch then the rest, or ``None``
    if every row was dropped (the iterator yields nothing).
    """
    iterator = iter(batch_iter)
    try:
        first = next(iterator)
    except StopIteration:
        return None

    def chained():
        yield first
        yield from iterator

    return chained()


def write_meta_json(output_lance_path: str) -> str:
    """Write a sibling meta.json listing this Lance table, for data.train_path."""
    meta_path = os.path.join(os.path.dirname(output_lance_path) or ".", "meta.json")
    shard = os.path.basename(output_lance_path)  # relative; resolved against meta.json's dir
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump({"shards": [shard]}, handle, indent=2)
    return meta_path


def write_shard_manifest(
    output_lance_path, input_path, input_format, all_files, selected_files, num_shards, shard_index
):
    """Record how this shard split the file list, so the finalizer can prove the
    shards form an exact, non-overlapping cover of the full input.

    Stores files relative to ``input_path`` to keep the manifest portable.
    """
    manifest_path = os.path.join(os.path.dirname(output_lance_path) or ".", "shard_manifest.json")
    selected_rel = sorted(os.path.relpath(f, input_path) for f in selected_files)
    # Digest of the FULL sorted file list so the finalizer can prove every shard
    # saw the identical input (catches a "delete A, add B, same count" mutation).
    all_rel = sorted(os.path.relpath(f, input_path) for f in all_files)
    full_files_digest = hashlib.sha256("\n".join(all_rel).encode("utf-8")).hexdigest()
    manifest = {
        "input": os.path.normpath(input_path),
        "input_format": input_format,
        "total_files": len(all_files),
        "full_files_digest": full_files_digest,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "selected_files": len(selected_files),
        "files": selected_rel,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    return manifest_path


def main() -> None:
    """Stream source files -> HKdata text.lance + meta.json, dropping empty rows."""
    parser = argparse.ArgumentParser(description="parquet/arrow/jsonl -> HKdata text.lance")
    parser.add_argument("--input", required=True, help="source file or directory")
    parser.add_argument("--output", required=True, help="output path ending in text.lance")
    parser.add_argument(
        "--input-format",
        choices=("parquet", "arrow", "jsonl"),
        default=None,
        help="force the input format (default: auto-detect)",
    )
    parser.add_argument("--text-column", default=None, help="text column/key name (default: auto-detect)")
    parser.add_argument("--id-column", default=None, help="id column name (parquet only; default: row index)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap rows for a quick smoke run (parquet path materializes; small values only)",
    )
    parser.add_argument("--tag", default=None, help="optional tag string added to every row")
    # Memory/throughput controls for large parquet scans (keep Arrow's buffers small).
    parser.add_argument("--batch-size", type=int, default=32768, help="parquet scan batch size")
    parser.add_argument("--batch-readahead", type=int, default=2, help="parquet batches read ahead")
    parser.add_argument("--fragment-readahead", type=int, default=1, help="parquet fragments read ahead")
    parser.add_argument("--log-every", type=int, default=5_000_000, help="progress log interval (rows)")
    # Sanity asserts for big runs (fail loudly on a silent under-scan).
    parser.add_argument("--expected-files", type=int, default=None, help="assert this many source files")
    parser.add_argument("--expected-rows", type=int, default=None, help="assert this many rows scanned")
    # Sharding: split one dataset across N independent jobs (for sources too big
    # for a single 72h job). Shard I converts files[I::N]; ids get an "I-" prefix
    # so they stay unique across shards; a shard_manifest.json records the split
    # so the finalizer can prove the shards form an exact, non-overlapping cover.
    parser.add_argument("--num-shards", type=int, default=None, help="total shard count (split the file list)")
    parser.add_argument("--shard-index", type=int, default=None, help="this shard's index (0..num_shards-1)")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer.")
    if args.limit is not None and args.expected_rows is not None:
        raise SystemExit("--limit is for smoke runs; do not combine it with --expected-rows.")
    sharded = args.num_shards is not None or args.shard_index is not None
    if sharded:
        if args.num_shards is None or args.shard_index is None:
            raise SystemExit("--num-shards and --shard-index must be given together.")
        if args.num_shards <= 0 or not (0 <= args.shard_index < args.num_shards):
            raise SystemExit(
                f"need num_shards>0 and 0<=shard_index<num_shards, got {args.num_shards}/{args.shard_index}"
            )
        if args.expected_rows is not None:
            raise SystemExit("--expected-rows can't be used with sharding (per-shard row count is unknown).")
        if args.id_column is not None:
            raise SystemExit("--id-column can't be used with sharding (ids are prefixed per shard).")

    input_format, files = detect_input(args.input, args.input_format)
    if args.expected_files is not None and len(files) != args.expected_files:
        raise SystemExit(f"Expected {args.expected_files} source files, found {len(files)} under {args.input}")
    if args.id_column and input_format != "parquet":
        raise SystemExit("--id-column is only supported for parquet input.")

    # Slice this shard's files out of the FULL (already asserted) sorted list.
    all_files = files
    if sharded:
        files = all_files[args.shard_index :: args.num_shards]
        if not files:
            raise SystemExit(f"shard {args.shard_index}/{args.num_shards} got 0 files (num_shards > file count?)")
        print(f"shard {args.shard_index}/{args.num_shards}: {len(files)} of {len(all_files)} files", flush=True)

    # Resolve the text column from the source schema (format-specific lookup).
    if input_format == "parquet":
        dataset = pads.dataset(files, format="parquet")
        column_names = dataset.schema.names
    elif input_format == "arrow":
        column_names = arrow_schema_names(files[0])
    else:
        column_names = jsonl_first_doc_keys(files[0])
    text_column = pick_text_column(column_names, args.text_column)
    if args.id_column:
        if args.id_column == text_column:
            raise SystemExit(f"--id-column '{args.id_column}' must differ from text column '{text_column}'.")
        if args.id_column not in column_names:
            raise SystemExit(f"--id-column '{args.id_column}' not in source columns {column_names}.")
    print(f"input format    : {input_format}  ({len(files)} file(s))", flush=True)
    print(f"source columns  : {column_names}", flush=True)
    print(f"text column     : {text_column}", flush=True)

    # Build the per-format source batch stream (text column only, plus optional id).
    if input_format == "parquet":
        read_columns = list(dict.fromkeys(c for c in (text_column, args.id_column) if c))
        if args.limit is not None:
            # head() materializes (smoke-size only) and tears down its scan cleanly;
            # abandoning a half-read streaming scanner can segfault at interpreter exit.
            source_batches = dataset.head(args.limit, columns=read_columns).to_batches()
        else:
            batch_size = args.batch_size
            text_type = dataset.schema.field(text_column).type
            if (
                pa.types.is_list(text_type) or pa.types.is_large_list(text_type)
            ) and batch_size > LIST_COLUMN_MAX_BATCH:
                batch_size = LIST_COLUMN_MAX_BATCH  # page-list rows are whole books; keep batches small
                print(f"text column is {text_type}; shrinking batch size to {batch_size}", flush=True)
            source_batches = dataset.scanner(
                columns=read_columns,
                batch_size=batch_size,
                batch_readahead=args.batch_readahead,
                fragment_readahead=args.fragment_readahead,
            ).to_batches()
    elif input_format == "arrow":
        source_batches = iter_arrow_batches(files, text_column)
    else:
        source_batches = None  # built below, needs counters

    counters = {"seen": 0, "kept": 0, "bad_json": 0}
    if input_format == "jsonl":
        source_batches = iter_jsonl_batches(files, text_column, counters)
    if args.limit is not None and input_format != "parquet":
        source_batches = limit_batches(source_batches, args.limit)

    info_json = json.dumps({"source": os.path.basename(os.path.normpath(args.input))})
    id_prefix = f"{args.shard_index}-" if sharded else ""
    batches = peek_first_batch(
        to_hkdata_batches(
            source_batches,
            text_column,
            args.id_column,
            info_json,
            args.tag,
            counters,
            args.log_every,
            id_prefix=id_prefix,
        )
    )
    if batches is None:
        raise SystemExit(f"All {counters['seen']} rows had empty/null text — refusing to write an empty table.")

    reader = pa.RecordBatchReader.from_batches(OUTPUT_SCHEMA, batches)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    lance.write_dataset(reader, args.output, mode="overwrite", data_storage_version="2.1")

    if args.expected_rows is not None and counters["seen"] != args.expected_rows:
        raise SystemExit(
            f"Expected {args.expected_rows} rows scanned, saw {counters['seen']} — aborting (under-scan?)."
        )

    if sharded:
        # A sharded run is one piece of a dataset; the finalizer combines them.
        write_shard_manifest(args.output, args.input, input_format, all_files, files, args.num_shards, args.shard_index)
        meta_path = None
    else:
        meta_path = write_meta_json(args.output)
    dropped = counters["seen"] - counters["kept"]
    print(f"wrote {counters['kept']:,} rows ({dropped:,} empty dropped) -> {args.output}", flush=True)
    if counters["bad_json"]:
        print(f"WARNING: {counters['bad_json']:,} lines had broken JSON (counted in dropped).", flush=True)
    if meta_path:
        print(f"meta.json -> {meta_path}   (set data.train_path to this)", flush=True)
    else:
        print("shard_manifest.json written (run finalize_shards to build the dataset meta.json)", flush=True)

    # All outputs are flushed and on disk. Tear down Arrow objects while the
    # interpreter is fully alive, then exit WITHOUT interpreter shutdown:
    # mixed C++/Rust destructor order at exit can segfault on many-fragment
    # datasets, and a post-success crash would make the sbatch wrapper treat
    # the run as failed and delete a finished staging directory.
    del reader, batches, source_batches
    if input_format == "parquet":
        del dataset
    gc.collect()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
