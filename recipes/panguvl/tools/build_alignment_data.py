"""Build PanguVL alignment data in mvp-dataset JSONL + TAR format.

This script reconstructs the ``panguvl_stage1_stage2_mm_only`` style corpus
from the downloaded Open-Bee parquet releases:

- all samples from ``Bee-Training-Data-Stage1``
- only multimodal subsets from ``Bee-Training-Data-Stage2``

The output layout is:

.. code-block:: text

    <output_dir>/
      train.jsonl
      images/
        train-00000.tar
        train-00001.tar
        ...

Each JSONL row contains:

- ``messages``: list of ``{"role", "content"}`` chat turns
- ``images``: list of tar reference URIs like
  ``images/train-00000.tar#stage1_000123_0.jpg``

The PanguVL recipe can consume this directly.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_INPUT_ROOT = Path("/mnt/data-alpha-sg-01/team-camera/shared/mvp-engine/data/Open-Bee")
DEFAULT_OUTPUT_ROOT = Path("data/panguvl/panguvl_stage1_stage2_mm_only")
DEFAULT_STAGE2_MM_SUBSETS = (
    "coyo",
    "laion2B",
    "synthdog_en_processed_new",
    "synthdog_zh_processed_new",
    "ureader_tr_processed_new",
)
ROLE_MAP = {
    "assistant": "assistant",
    "gpt": "assistant",
    "human": "user",
    "system": "system",
    "tool": "tool",
    "user": "user",
}
SAFE_KEY_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class SourceSpec:
    """One parquet-backed dataset source."""

    name: str
    parquet_dir: Path


@dataclass
class SampleStats:
    """Running counters for the conversion job."""

    samples_written: int = 0
    images_written: int = 0
    tar_shards_written: int = 0


class TarShardWriter:
    """Write image payloads into sequential tar shards."""

    def __init__(self, output_dir: Path, *, prefix: str, samples_per_shard: int) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.samples_per_shard = samples_per_shard
        self._current_tar: tarfile.TarFile | None = None
        self._current_tar_path: Path | None = None
        self._shard_index = -1
        self._samples_in_shard = 0
        self._closed = False

    def _close_current_tar(self) -> None:
        """Close the currently open tar shard without ending the writer lifecycle."""
        if self._current_tar is not None:
            self._current_tar.close()
            self._current_tar = None
            self._current_tar_path = None

    def close(self) -> None:
        """Close the writer after the final shard is written."""
        self._close_current_tar()
        self._closed = True

    def start_sample(self) -> None:
        """Rotate shards when the current shard is full."""
        if self._closed:
            raise RuntimeError("tar shard writer is already closed")
        if self._current_tar is None or self._samples_in_shard >= self.samples_per_shard:
            self._close_current_tar()
            self._shard_index += 1
            shard_name = f"{self.prefix}-{self._shard_index:05d}.tar"
            self._current_tar_path = self.output_dir / shard_name
            self._current_tar = tarfile.open(self._current_tar_path, mode="w")
            self._samples_in_shard = 0
        self._samples_in_shard += 1

    def add_bytes(self, *, member_name: str, payload: bytes) -> str:
        """Append one image to the current tar and return the relative reference URI."""
        if self._current_tar is None or self._current_tar_path is None:
            raise RuntimeError("tar shard writer has no open shard")

        info = tarfile.TarInfo(name=member_name)
        info.size = len(payload)
        info.mtime = 0
        self._current_tar.addfile(info, io.BytesIO(payload))
        return f"images/{self._current_tar_path.name}#{member_name}"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Downloaded Open-Bee dataset root. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output directory for train.jsonl and image tar shards. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--samples-per-tar",
        type=int,
        default=10_000,
        help="Maximum number of samples per image tar shard.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Parquet rows to decode per batch.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for smoke tests or partial exports.",
    )
    return parser.parse_args()


def build_source_specs(input_root: Path) -> list[SourceSpec]:
    """Return the source directories that make up the alignment dataset."""
    stage1_dir = input_root / "Bee-Training-Data-Stage1" / "data"
    stage2_root = input_root / "Bee-Training-Data-Stage2"

    specs = [SourceSpec(name="stage1", parquet_dir=stage1_dir)]
    specs.extend(SourceSpec(name=subset, parquet_dir=stage2_root / subset) for subset in DEFAULT_STAGE2_MM_SUBSETS)
    return specs


def iter_parquet_rows(parquet_dir: Path, *, batch_size: int) -> Iterator[dict[str, Any]]:
    """Yield Python rows from every parquet file under one directory."""
    shard_paths = sorted(parquet_dir.glob("*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"No parquet shards found in {parquet_dir}")

    columns = ["id", "images", "conversations"]
    for shard_path in shard_paths:
        parquet_file = pq.ParquetFile(shard_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            yield from batch_to_rows(batch)


def batch_to_rows(batch: pa.RecordBatch) -> Iterator[dict[str, Any]]:
    """Convert one Arrow record batch into row dictionaries."""
    columns = {name: batch.column(name).to_pylist() for name in batch.schema.names}
    row_count = batch.num_rows
    for row_index in range(row_count):
        yield {name: values[row_index] for name, values in columns.items()}


def normalize_messages(
    conversations: list[dict[str, Any]], *, source_name: str, sample_id: str
) -> list[dict[str, str]]:
    """Convert Open-Bee conversation format into the recipe's JSONL schema."""
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"{source_name}:{sample_id} has invalid conversations")

    messages: list[dict[str, str]] = []
    for message in conversations:
        role = message.get("from")
        content = message.get("value")
        normalized_role = ROLE_MAP.get(role)
        if normalized_role is None:
            raise ValueError(f"{source_name}:{sample_id} has unsupported role {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"{source_name}:{sample_id} has non-string content")
        messages.append({"role": normalized_role, "content": content})
    return messages


def sanitize_key(raw: str) -> str:
    """Make one sample key tar-member-safe and stable."""
    return SAFE_KEY_RE.sub("_", raw).strip("._") or "sample"


def infer_extension(image_path: str | None) -> str:
    """Infer the tar member extension from the parquet image path."""
    if image_path:
        suffix = Path(image_path).suffix.lower().lstrip(".")
        if suffix:
            return "jpg" if suffix == "jpeg" else suffix
    return "jpg"


def count_image_placeholders(messages: Iterable[dict[str, str]]) -> int:
    """Count ``<image>`` placeholders across the conversation."""
    return sum(message["content"].count("<image>") for message in messages)


def expand_images_for_placeholders(
    images: list[dict[str, Any]],
    *,
    expected_images: int,
    source_name: str,
    sample_id: str,
) -> list[dict[str, Any]]:
    """Match image entries to placeholder count.

    Some Open-Bee samples reuse the same image across multiple conversation
    turns, while storing the underlying image payload only once. In that case,
    expand the single image entry so downstream JSONL rows still satisfy the
    recipe contract: one image reference per ``<image>`` placeholder.
    """
    if expected_images <= 0:
        raise ValueError(f"{source_name}:{sample_id} has images but no <image> placeholders")

    if len(images) == expected_images:
        return images

    if len(images) == 1 and expected_images > 1:
        return images * expected_images

    raise ValueError(
        f"{source_name}:{sample_id} placeholder/image mismatch: placeholders={expected_images} images={len(images)}"
    )


def make_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    """Create and return the JSONL file path plus tar output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "train.jsonl", images_dir


def convert_sources(
    sources: list[SourceSpec],
    *,
    output_jsonl: Path,
    images_dir: Path,
    batch_size: int,
    samples_per_tar: int,
    max_samples: int | None,
) -> SampleStats:
    """Write the merged alignment dataset."""
    stats = SampleStats()
    tar_writer = TarShardWriter(images_dir, prefix="train", samples_per_shard=samples_per_tar)

    try:
        with output_jsonl.open("w", encoding="utf-8") as jsonl_handle:
            for source in sources:
                if not source.parquet_dir.is_dir():
                    raise FileNotFoundError(f"Missing source directory: {source.parquet_dir}")

                for row in iter_parquet_rows(source.parquet_dir, batch_size=batch_size):
                    if max_samples is not None and stats.samples_written >= max_samples:
                        tar_writer.close()
                        stats.tar_shards_written = tar_writer._shard_index + 1 if tar_writer._shard_index >= 0 else 0
                        return stats

                    sample_id = str(row["id"])
                    messages = normalize_messages(
                        row["conversations"],
                        source_name=source.name,
                        sample_id=sample_id,
                    )
                    images = row["images"]
                    if not isinstance(images, list):
                        raise ValueError(f"{source.name}:{sample_id} has invalid images field")
                    if not images:
                        raise ValueError(f"{source.name}:{sample_id} is text-only; alignment data should be multimodal")

                    expected_images = count_image_placeholders(messages)
                    images = expand_images_for_placeholders(
                        images,
                        expected_images=expected_images,
                        source_name=source.name,
                        sample_id=sample_id,
                    )

                    tar_writer.start_sample()
                    sample_key_prefix = sanitize_key(f"{source.name}_{sample_id}")
                    image_refs: list[str] = []
                    for image_index, image in enumerate(images):
                        if not isinstance(image, dict):
                            raise ValueError(f"{source.name}:{sample_id} image entry is not a dictionary")
                        payload = image.get("bytes")
                        if not isinstance(payload, (bytes, bytearray)):
                            raise ValueError(f"{source.name}:{sample_id} image {image_index} is missing raw bytes")
                        member_name = f"{sample_key_prefix}_{image_index}.{infer_extension(image.get('path'))}"
                        image_refs.append(tar_writer.add_bytes(member_name=member_name, payload=bytes(payload)))
                        stats.images_written += 1

                    sample = {
                        "messages": messages,
                        "images": image_refs,
                        "source": source.name,
                        "source_id": sample_id,
                    }
                    jsonl_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    stats.samples_written += 1

                    if stats.samples_written % 10_000 == 0:
                        print(
                            f"[progress] samples={stats.samples_written} "
                            f"images={stats.images_written} tar_shards={tar_writer._shard_index + 1}"
                        )
    finally:
        tar_writer.close()

    stats.tar_shards_written = tar_writer._shard_index + 1 if tar_writer._shard_index >= 0 else 0
    return stats


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if args.samples_per_tar <= 0:
        raise ValueError("--samples-per-tar must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be > 0 when provided")

    output_jsonl, images_dir = make_output_dirs(args.output_dir.expanduser().resolve())
    sources = build_source_specs(args.input_root.expanduser().resolve())
    stats = convert_sources(
        sources,
        output_jsonl=output_jsonl,
        images_dir=images_dir,
        batch_size=args.batch_size,
        samples_per_tar=args.samples_per_tar,
        max_samples=args.max_samples,
    )

    print(f"[done] train_jsonl={output_jsonl}")
    print(f"[done] images_dir={images_dir}")
    print(f"[done] samples={stats.samples_written} images={stats.images_written} tar_shards={stats.tar_shards_written}")


if __name__ == "__main__":
    main()
