"""Shuffle Basic VLM stage2 parquet shards into globally mixed output shards.

This tool scans ``./data/openbee/stage2/**/*.parquet``, treats the parquet
file's parent directory relative to ``--input-root`` as the source name, and
rewrites the dataset into a new flat shard directory where each output parquet
contains all sources whenever possible and rows are shuffled globally inside
each shard.

Implementation notes:

- planning happens on parquet row-group metadata, so the script avoids loading
  the full dataset into memory up front
- workers operate on mixed-source shard plans instead of one source at a time
- progress bars are rendered in the parent process: one global bar plus one bar
  per worker
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import queue
import random
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

DEFAULT_INPUT_ROOT = Path("data/openbee/stage2")
DEFAULT_OUTPUT_ROOT = Path("data/openbee/stage2_shuffled")
DEFAULT_TARGET_ROWS_PER_SHARD = 50_000
DEFAULT_ROW_GROUP_SIZE = 8_192
DEFAULT_SHUFFLE_BATCH_SIZE = 5120
DEFAULT_BATCH_BUFFER_SIZE = 1280

ProgressKind = Literal["advance", "done", "error"]


@dataclass(frozen=True)
class RowGroupSpec:
    """One concrete parquet row group."""

    file_path: Path
    source_name: str
    row_group_index: int
    num_rows: int


@dataclass(frozen=True)
class ShardPlan:
    """One output parquet shard assembled from multiple row groups."""

    shard_index: int
    row_groups: tuple[RowGroupSpec, ...]
    total_rows: int
    source_names: tuple[str, ...]


@dataclass(frozen=True)
class ProgressEvent:
    """One worker progress update sent back to the parent process."""

    kind: ProgressKind
    worker_id: int
    rows: int = 0
    detail: str = ""


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Recursive parquet input root. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Flat output directory for mixed parquet shards. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (mp.cpu_count() or 1) // 2),
        help="Number of worker processes. Default: half the visible CPU count.",
    )
    parser.add_argument(
        "--target-rows-per-shard",
        type=int,
        default=DEFAULT_TARGET_ROWS_PER_SHARD,
        help=f"Approximate rows per output parquet shard. Default: {DEFAULT_TARGET_ROWS_PER_SHARD}",
    )
    parser.add_argument(
        "--output-row-group-size",
        type=int,
        default=DEFAULT_ROW_GROUP_SIZE,
        help=f"Row group size used when writing output parquet files. Default: {DEFAULT_ROW_GROUP_SIZE}",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec for output shards. Default: zstd",
    )
    parser.add_argument(
        "--shuffle-batch-size",
        type=int,
        default=DEFAULT_SHUFFLE_BATCH_SIZE,
        help=(f"Maximum rows shuffled at once inside one worker-side batch. Default: {DEFAULT_SHUFFLE_BATCH_SIZE}"),
    )
    parser.add_argument(
        "--batch-buffer-size",
        type=int,
        default=DEFAULT_BATCH_BUFFER_SIZE,
        help=(
            "Number of shuffled mini-batches kept in memory before randomly flushing one. "
            f"Default: {DEFAULT_BATCH_BUFFER_SIZE}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260419,
        help="Random seed for source planning and in-shard row shuffling.",
    )
    return parser.parse_args()


def infer_source_name(input_root: Path, parquet_path: Path) -> str:
    """Infer the logical source name from the parquet path."""
    relative_parent = parquet_path.relative_to(input_root).parent
    if relative_parent == Path("."):
        return parquet_path.parent.name
    return relative_parent.as_posix()


def validate_paths(input_root: Path, output_dir: Path) -> None:
    """Validate input/output path safety constraints."""
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if input_root.resolve() == output_dir.resolve() or input_root.resolve() in output_dir.resolve().parents:
        raise ValueError("--output-dir must not live inside --input-root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory already exists and is not empty: {output_dir}")


def collect_row_groups(input_root: Path) -> tuple[dict[str, deque[RowGroupSpec]], int, int]:
    """Collect parquet row-group metadata grouped by source."""
    groups_by_source: dict[str, deque[RowGroupSpec]] = defaultdict(deque)
    parquet_paths = sorted(input_root.rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {input_root}")

    total_rows = 0
    total_files = 0
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        source_name = infer_source_name(input_root, parquet_path)
        metadata = parquet_file.metadata
        total_files += 1
        for row_group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_index)
            spec = RowGroupSpec(
                file_path=parquet_path,
                source_name=source_name,
                row_group_index=row_group_index,
                num_rows=row_group.num_rows,
            )
            groups_by_source[source_name].append(spec)
            total_rows += row_group.num_rows

    return groups_by_source, total_files, total_rows


def shuffle_source_groups(
    groups_by_source: dict[str, deque[RowGroupSpec]], *, seed: int
) -> dict[str, deque[RowGroupSpec]]:
    """Shuffle row groups within each source to avoid source-local ordering."""
    rng = random.Random(seed)
    shuffled: dict[str, deque[RowGroupSpec]] = {}
    for source_name in sorted(groups_by_source):
        row_groups = list(groups_by_source[source_name])
        rng.shuffle(row_groups)
        shuffled[source_name] = deque(row_groups)
    return shuffled


def choose_output_shard_count(
    groups_by_source: dict[str, deque[RowGroupSpec]],
    *,
    total_rows: int,
    target_rows_per_shard: int,
) -> int:
    """Choose an output shard count that keeps all-source coverage feasible.

    The planner caps the shard count by the smallest per-source row-group count so
    every shard can receive at least one row group from every source without
    duplicating samples. This intentionally prefers larger shards over dropping a
    source from tail shards when sources are highly imbalanced.
    """
    if not groups_by_source:
        raise ValueError("No sources available to plan output shards")

    target_shard_count = max(1, math.ceil(total_rows / target_rows_per_shard))
    max_all_source_shards = min(len(row_groups) for row_groups in groups_by_source.values())
    return max(1, min(target_shard_count, max_all_source_shards))


def partition_source_row_groups(
    row_groups: deque[RowGroupSpec],
    *,
    shard_count: int,
) -> list[list[RowGroupSpec]]:
    """Split one shuffled source into near-equal row buckets across shards."""
    assignments = [[] for _ in range(shard_count)]
    row_groups_list = list(row_groups)
    if not row_groups_list:
        return assignments

    total_source_rows = sum(row_group.num_rows for row_group in row_groups_list)
    boundaries = [total_source_rows * (shard_index + 1) / shard_count for shard_index in range(shard_count - 1)]
    current_shard = 0
    assigned_rows = 0

    for row_group_index, row_group in enumerate(row_groups_list):
        assignments[current_shard].append(row_group)
        assigned_rows += row_group.num_rows

        if current_shard >= shard_count - 1:
            continue

        remaining_groups = len(row_groups_list) - row_group_index - 1
        remaining_shards = shard_count - current_shard - 1
        must_advance = remaining_groups == remaining_shards
        meets_boundary = assigned_rows >= boundaries[current_shard]
        if must_advance or meets_boundary:
            current_shard += 1

    return assignments


def build_shard_plans(
    groups_by_source: dict[str, deque[RowGroupSpec]],
    *,
    target_rows_per_shard: int,
    seed: int,
) -> list[ShardPlan]:
    """Plan globally balanced all-source output shards from row-group metadata."""
    total_rows = sum(row_group.num_rows for row_groups in groups_by_source.values() for row_group in row_groups)
    shard_count = choose_output_shard_count(
        groups_by_source,
        total_rows=total_rows,
        target_rows_per_shard=target_rows_per_shard,
    )
    shard_row_groups: list[list[RowGroupSpec]] = [[] for _ in range(shard_count)]
    shard_rows = [0 for _ in range(shard_count)]
    shard_sources: list[set[str]] = [set() for _ in range(shard_count)]
    rng = random.Random(seed)

    for source_name in sorted(groups_by_source):
        per_shard_assignments = partition_source_row_groups(groups_by_source[source_name], shard_count=shard_count)
        if shard_count > 1:
            rotation = rng.randrange(shard_count)
            per_shard_assignments = per_shard_assignments[rotation:] + per_shard_assignments[:rotation]

        for shard_index, assigned_row_groups in enumerate(per_shard_assignments):
            if not assigned_row_groups:
                continue
            shard_row_groups[shard_index].extend(assigned_row_groups)
            shard_rows[shard_index] += sum(row_group.num_rows for row_group in assigned_row_groups)
            shard_sources[shard_index].add(source_name)

    plans: list[ShardPlan] = []
    for shard_index, row_groups in enumerate(shard_row_groups):
        if not row_groups:
            continue
        plans.append(
            ShardPlan(
                shard_index=shard_index,
                row_groups=tuple(row_groups),
                total_rows=shard_rows[shard_index],
                source_names=tuple(sorted(shard_sources[shard_index])),
            )
        )

    return plans


def assign_shards_to_workers(shard_plans: list[ShardPlan], *, num_workers: int) -> list[list[ShardPlan]]:
    """Assign shard plans to workers in round-robin order."""
    assignments = [[] for _ in range(num_workers)]
    for plan in shard_plans:
        assignments[plan.shard_index % num_workers].append(plan)
    return assignments


def shuffle_batch_rows(batch: pa.RecordBatch, *, seed: int) -> pa.RecordBatch:
    """Shuffle one bounded record batch."""
    if batch.num_rows <= 1:
        return batch
    indices = list(range(batch.num_rows))
    random.Random(seed).shuffle(indices)
    shuffled_table = pa.Table.from_batches([batch], schema=batch.schema).take(pa.array(indices, type=pa.int64()))
    return shuffled_table.to_batches()[0]


def flush_batches(
    writer: pq.ParquetWriter,
    batches: list[pa.RecordBatch],
    *,
    output_row_group_size: int,
) -> None:
    """Write buffered record batches into the output parquet shard."""
    if not batches:
        return
    writer.write_table(
        pa.Table.from_batches(batches, schema=batches[0].schema),
        row_group_size=output_row_group_size,
    )


def write_shard(
    plan: ShardPlan,
    *,
    output_path: Path,
    output_row_group_size: int,
    shuffle_batch_size: int,
    batch_buffer_size: int,
    compression: str,
    seed: int,
    progress_queue: mp.Queue[ProgressEvent] | None = None,
    worker_id: int | None = None,
) -> None:
    """Stream one shard to disk using bounded shuffled mini-batches."""
    rng = random.Random(seed + plan.shard_index)
    row_groups = list(plan.row_groups)
    rng.shuffle(row_groups)
    parquet_files: dict[Path, pq.ParquetFile] = {}
    buffered_shuffle_batches: list[pa.RecordBatch] = []
    buffered_write_batches: list[pa.RecordBatch] = []
    buffered_write_rows = 0
    writer: pq.ParquetWriter | None = None

    try:
        for row_group in row_groups:
            parquet_file = parquet_files.get(row_group.file_path)
            if parquet_file is None:
                parquet_file = pq.ParquetFile(row_group.file_path)
                parquet_files[row_group.file_path] = parquet_file

            table = parquet_file.read_row_group(row_group.row_group_index, use_threads=True)
            for batch in table.to_batches(max_chunksize=shuffle_batch_size):
                shuffled_batch = shuffle_batch_rows(batch, seed=rng.randrange(1 << 63))
                buffered_shuffle_batches.append(shuffled_batch)
                if progress_queue is not None and worker_id is not None:
                    progress_queue.put(
                        ProgressEvent(
                            kind="advance",
                            worker_id=worker_id,
                            rows=shuffled_batch.num_rows,
                        )
                    )

                if len(buffered_shuffle_batches) >= batch_buffer_size:
                    selected_index = rng.randrange(len(buffered_shuffle_batches))
                    selected_batch = buffered_shuffle_batches.pop(selected_index)
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, selected_batch.schema, compression=compression)
                    buffered_write_batches.append(selected_batch)
                    buffered_write_rows += selected_batch.num_rows
                    if buffered_write_rows >= output_row_group_size:
                        flush_batches(
                            writer,
                            buffered_write_batches,
                            output_row_group_size=output_row_group_size,
                        )
                        buffered_write_batches = []
                        buffered_write_rows = 0

        while buffered_shuffle_batches:
            selected_index = rng.randrange(len(buffered_shuffle_batches))
            selected_batch = buffered_shuffle_batches.pop(selected_index)
            if writer is None:
                writer = pq.ParquetWriter(output_path, selected_batch.schema, compression=compression)
            buffered_write_batches.append(selected_batch)
            buffered_write_rows += selected_batch.num_rows
            if buffered_write_rows >= output_row_group_size:
                flush_batches(
                    writer,
                    buffered_write_batches,
                    output_row_group_size=output_row_group_size,
                )
                buffered_write_batches = []
                buffered_write_rows = 0

        if writer is None:
            raise RuntimeError(f"Shard plan {plan.shard_index} did not materialize any rows")

        flush_batches(
            writer,
            buffered_write_batches,
            output_row_group_size=output_row_group_size,
        )
    except Exception:
        if writer is not None:
            writer.close()
        output_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()


def worker_main(
    worker_id: int,
    plans: list[ShardPlan],
    *,
    output_dir: Path,
    total_shards: int,
    output_row_group_size: int,
    shuffle_batch_size: int,
    batch_buffer_size: int,
    compression: str,
    seed: int,
    progress_queue: mp.Queue[ProgressEvent],
) -> None:
    """Execute one worker's assigned shard plans."""
    try:
        for plan in plans:
            output_path = output_dir / f"train-{plan.shard_index:05d}-of-{total_shards:05d}.parquet"
            write_shard(
                plan,
                output_path=output_path,
                output_row_group_size=output_row_group_size,
                shuffle_batch_size=shuffle_batch_size,
                batch_buffer_size=batch_buffer_size,
                compression=compression,
                seed=seed,
                progress_queue=progress_queue,
                worker_id=worker_id,
            )
    except Exception:
        progress_queue.put(
            ProgressEvent(
                kind="error",
                worker_id=worker_id,
                detail=traceback.format_exc(),
            )
        )
        raise
    else:
        progress_queue.put(ProgressEvent(kind="done", worker_id=worker_id))


def render_progress(
    processes: list[mp.Process],
    worker_row_totals: dict[int, int],
    *,
    total_rows: int,
    progress_queue: mp.Queue[ProgressEvent],
) -> None:
    """Render one global progress bar plus one bar per worker."""
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("{task.completed:>12,.0f}/{task.total:>12,.0f} rows"),
        TimeElapsedColumn(),
        TimeRemainingColumn(compact=True, elapsed_when_finished=True),
    )
    worker_tasks: dict[int, TaskID] = {}

    with progress:
        global_task = progress.add_task("global", total=total_rows)
        for worker_id, worker_total in sorted(worker_row_totals.items()):
            worker_tasks[worker_id] = progress.add_task(f"worker {worker_id}", total=worker_total)

        finished_workers = 0
        while finished_workers < len(processes):
            try:
                event = progress_queue.get(timeout=0.2)
            except queue.Empty:
                event = None

            if event is not None:
                if event.kind == "advance":
                    progress.advance(global_task, event.rows)
                    progress.advance(worker_tasks[event.worker_id], event.rows)
                elif event.kind == "done":
                    finished_workers += 1
                elif event.kind == "error":
                    raise RuntimeError(f"Worker {event.worker_id} failed:\n{event.detail}")

            for process in processes:
                if process.exitcode not in (None, 0):
                    raise RuntimeError(f"Worker process {process.pid} exited with code {process.exitcode}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be > 0")
    if args.target_rows_per_shard <= 0:
        raise ValueError("--target-rows-per-shard must be > 0")
    if args.output_row_group_size <= 0:
        raise ValueError("--output-row-group-size must be > 0")
    if args.shuffle_batch_size <= 0:
        raise ValueError("--shuffle-batch-size must be > 0")
    if args.batch_buffer_size <= 0:
        raise ValueError("--batch-buffer-size must be > 0")

    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    validate_paths(input_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups_by_source, total_files, total_rows = collect_row_groups(input_root)
    shuffled_groups = shuffle_source_groups(groups_by_source, seed=args.seed)
    shard_plans = build_shard_plans(
        shuffled_groups,
        target_rows_per_shard=args.target_rows_per_shard,
        seed=args.seed,
    )
    if not shard_plans:
        raise RuntimeError("No output shard plans were generated")

    worker_count = min(args.num_workers, len(shard_plans))
    assignments = assign_shards_to_workers(shard_plans, num_workers=worker_count)
    worker_row_totals = {
        worker_id: sum(plan.total_rows for plan in plans) for worker_id, plans in enumerate(assignments)
    }

    source_count = len(groups_by_source)
    target_shard_count = max(1, math.ceil(total_rows / args.target_rows_per_shard))
    min_sources_per_shard = min(len(plan.source_names) for plan in shard_plans)
    max_sources_per_shard = max(len(plan.source_names) for plan in shard_plans)
    print(
        f"[plan] files={total_files} sources={source_count} rows={total_rows} "
        f"target_output_shards={target_shard_count} output_shards={len(shard_plans)} "
        f"source_coverage={min_sources_per_shard}-{max_sources_per_shard} workers={worker_count}"
    )

    ctx = mp.get_context("spawn")
    progress_queue: mp.Queue[ProgressEvent] = ctx.Queue()
    processes: list[mp.Process] = []
    try:
        for worker_id, plans in enumerate(assignments):
            process = ctx.Process(
                target=worker_main,
                kwargs={
                    "worker_id": worker_id,
                    "plans": plans,
                    "output_dir": output_dir,
                    "total_shards": len(shard_plans),
                    "output_row_group_size": args.output_row_group_size,
                    "shuffle_batch_size": args.shuffle_batch_size,
                    "batch_buffer_size": args.batch_buffer_size,
                    "compression": args.compression,
                    "seed": args.seed,
                    "progress_queue": progress_queue,
                },
            )
            process.start()
            processes.append(process)

        render_progress(processes, worker_row_totals, total_rows=total_rows, progress_queue=progress_queue)
    except Exception:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise
    finally:
        for process in processes:
            process.join()
        progress_queue.close()
        progress_queue.join_thread()

    print(f"[done] output_dir={output_dir}")
    print(f"[done] shards={len(shard_plans)} rows={total_rows}")


if __name__ == "__main__":
    main()
