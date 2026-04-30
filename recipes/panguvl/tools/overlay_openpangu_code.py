"""Overlay patched OpenPangu-VL runtime code onto a local checkpoint.

This tool copies only Python/runtime source files from the recipe-local
``recipes/panguvl/third_party`` directory into a user-provided OpenPangu-VL
checkpoint directory. It intentionally does not copy model weights, tokenizer
files, chat templates, or generated caches.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

RUNTIME_FILES = (
    "configuration_openpangu_vl.py",
    "imageprocessor_openpangu_vl.py",
    "modeling_openpangu_embedded.py",
    "modeling_openpangu_vl.py",
)

REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "preprocessor_config.json",
)

FORBIDDEN_PATTERNS = (
    "model*.safetensors*",
    "pytorch_model*.bin",
    "tokenizer.model",
    "tokenizer*.json",
    "special_tokens_map.json",
    "chat_template.json",
    "generation_config.json",
)


def default_source_dir() -> Path:
    """Return the repository-local vendored OpenPangu source directory."""
    return Path(__file__).resolve().parents[1] / "third_party"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Local OpenPangu-VL checkpoint directory that already contains weights and tokenizer files.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_source_dir(),
        help="Directory containing the patched OpenPangu runtime .py files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate inputs and report planned copies without writing files.",
    )
    return parser.parse_args()


def validate_source_dir(source_dir: Path) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"OpenPangu source directory does not exist: {source_dir}")

    missing_files = [file_name for file_name in RUNTIME_FILES if not (source_dir / file_name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            "OpenPangu source directory is missing required runtime files: " + ", ".join(missing_files)
        )

    forbidden_matches: list[Path] = []
    for pattern in FORBIDDEN_PATTERNS:
        forbidden_matches.extend(source_dir.glob(pattern))
    if forbidden_matches:
        formatted = ", ".join(path.name for path in sorted(forbidden_matches))
        raise ValueError(f"OpenPangu source directory contains forbidden model/tokenizer artifacts: {formatted}")


def validate_checkpoint_dir(checkpoint_dir: Path) -> None:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    missing_files = [file_name for file_name in REQUIRED_CHECKPOINT_FILES if not (checkpoint_dir / file_name).is_file()]
    if missing_files:
        raise FileNotFoundError("Checkpoint directory is missing expected metadata files: " + ", ".join(missing_files))


def copy_runtime_files(source_dir: Path, checkpoint_dir: Path, *, dry_run: bool) -> None:
    for file_name in RUNTIME_FILES:
        source_path = source_dir / file_name
        target_path = checkpoint_dir / file_name
        action = "would copy" if dry_run else "copy"
        print(f"{action}: {source_path} -> {target_path}")
        if not dry_run:
            shutil.copy2(source_path, target_path)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    dry_run = bool(args.dry_run or args.check)

    validate_source_dir(source_dir)
    validate_checkpoint_dir(checkpoint_dir)
    copy_runtime_files(source_dir, checkpoint_dir, dry_run=dry_run)

    if args.check:
        print("check passed")


if __name__ == "__main__":
    main()
