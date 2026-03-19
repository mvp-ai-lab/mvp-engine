#!/usr/bin/env python3
"""Summarize an mvp-engine run directory into report-friendly JSON."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - depends on local environment
    yaml = None


LEVEL_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| "
    r"(?P<run_id>.+?) \| (?P<level>DEBUG|INFO|WARN|ERROR)\s+\| (?P<message>.*)$"
)
STEP_RE = re.compile(r"(?:Epoch (?P<epoch>\d+)\s*-\s*)?Step\s+(?P<step>\d+)")

SUMMARY_KEYS = (
    "workflow",
    "engine",
    "git_info",
    "project.name",
    "project.run_id",
    "project.output_dir",
    "model.name",
    "data.batch_size",
    "data.num_workers",
    "optim.lr",
    "optim.weight_decay",
    "loop.policy",
    "loop.total_steps",
    "loop.checkpoint.interval",
    "loop.checkpoint.keep_n",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory under outputs/.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--history-limit",
        type=int,
        default=5,
        help="Number of most recent metric points to keep per metric in the JSON tail.",
    )
    parser.add_argument(
        "--include-config",
        action="store_true",
        help="Include the parsed YAML config in the JSON output.",
    )
    return parser.parse_args()


def as_abs(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.resolve())


def sort_checkpoint_key(path: Path) -> tuple[int, int, str]:
    name = path.name
    if "_" not in name:
        return (2, -1, name)
    prefix, suffix = name.split("_", 1)
    try:
        index = int(suffix)
    except ValueError:
        index = -1
    if prefix == "iter":
        return (0, index, name)
    if prefix == "epoch":
        return (1, index, name)
    return (2, index, name)


def select(data: dict[str, Any], dotted_key: str) -> Any:
    current: Any = data
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return value
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"nan", "+nan", "-nan"}:
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number


def discover_files(run_dir: Path) -> dict[str, Any]:
    config_files = sorted(run_dir.glob("config_*.yaml"))
    log_files = sorted(run_dir.glob("log_*.log"))
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_dirs = []
    if checkpoints_dir.exists():
        checkpoint_dirs = sorted([path for path in checkpoints_dir.iterdir() if path.is_dir()], key=sort_checkpoint_key)

    external_patterns = {
        "results_json": "results*.json",
        "metrics_json": "metrics*.json",
        "samples_jsonl": "samples*.jsonl",
        "predictions_jsonl": "predictions*.jsonl",
        "predictions_csv": "predictions*.csv",
    }
    external_artifacts = {
        key: [as_abs(path) for path in sorted(run_dir.rglob(pattern))] for key, pattern in external_patterns.items()
    }

    return {
        "run_dir": as_abs(run_dir),
        "config_path": as_abs(config_files[0]) if config_files else None,
        "log_path": as_abs(log_files[0]) if log_files else None,
        "checkpoints_dir": as_abs(checkpoints_dir) if checkpoints_dir.exists() else None,
        "checkpoints": [
            {
                "name": path.name,
                "path": as_abs(path),
            }
            for path in checkpoint_dirs
        ],
        "external_artifacts": external_artifacts,
    }


def load_config(config_path: Path | None, include_config: bool) -> dict[str, Any]:
    if config_path is None:
        return {"summary": {}, "config": None}
    if yaml is None:
        return {
            "summary": {},
            "config": None,
            "warning": "PyYAML is not available; config parsing was skipped.",
        }

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    summary = {}
    for key in SUMMARY_KEYS:
        value = select(data, key)
        if value is not None:
            summary[key] = value

    return {
        "summary": summary,
        "config": data if include_config else None,
    }


def parse_log(log_path: Path | None, history_limit: int) -> dict[str, Any]:
    if log_path is None:
        return {
            "run_id": None,
            "metric_lines": 0,
            "last_logged_step": None,
            "levels": {"DEBUG": 0, "INFO": 0, "WARN": 0, "ERROR": 0},
            "warnings": [],
            "errors": [],
            "metrics": {},
        }

    levels = {"DEBUG": 0, "INFO": 0, "WARN": 0, "ERROR": 0}
    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    metric_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_id: str | None = None
    last_logged_step: int | None = None
    metric_lines = 0

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue

            if " || " in line:
                metric_lines += 1
                header, metric_payload = line.split(" || ", 1)
                header_parts = [part.strip() for part in header.split(" | ")]
                if len(header_parts) >= 2 and run_id is None:
                    run_id = header_parts[1]

                step_match = STEP_RE.search(header)
                if step_match is None:
                    continue

                step = int(step_match.group("step"))
                epoch = step_match.group("epoch")
                last_logged_step = step

                for token in metric_payload.split(" | "):
                    token = token.strip()
                    if not token or ":" not in token:
                        continue
                    key, value = token.split(":", 1)
                    metric_history[key.strip()].append(
                        {
                            "step": step,
                            "epoch": int(epoch) if epoch is not None else None,
                            "value": parse_scalar(value),
                        }
                    )
                continue

            level_match = LEVEL_LINE_RE.match(line)
            if level_match is None:
                continue

            run_id = run_id or level_match.group("run_id")
            level = level_match.group("level")
            levels[level] += 1
            entry = {
                "timestamp": level_match.group("timestamp"),
                "message": level_match.group("message"),
            }
            if level == "WARN":
                warnings.append(entry)
            elif level == "ERROR":
                errors.append(entry)

    metrics_summary = {}
    for key, entries in metric_history.items():
        latest = entries[-1]
        summary: dict[str, Any] = {
            "count": len(entries),
            "latest": latest["value"],
            "latest_step": latest["step"],
            "tail": entries[-history_limit:] if history_limit > 0 else [],
        }

        numeric_entries = [
            entry
            for entry in entries
            if isinstance(entry["value"], (int, float)) and not isinstance(entry["value"], bool)
        ]
        if numeric_entries:
            min_entry = min(numeric_entries, key=lambda entry: float(entry["value"]))
            max_entry = max(numeric_entries, key=lambda entry: float(entry["value"]))
            summary["min"] = {"value": min_entry["value"], "step": min_entry["step"]}
            summary["max"] = {"value": max_entry["value"], "step": max_entry["step"]}

        metrics_summary[key] = summary

    return {
        "run_id": run_id,
        "metric_lines": metric_lines,
        "last_logged_step": last_logged_step,
        "levels": levels,
        "warnings": warnings[-20:],
        "errors": errors[-20:],
        "metrics": metrics_summary,
    }


def build_summary(run_dir: Path, history_limit: int, include_config: bool) -> dict[str, Any]:
    artifacts = discover_files(run_dir)
    config_path = Path(artifacts["config_path"]) if artifacts["config_path"] is not None else None
    log_path = Path(artifacts["log_path"]) if artifacts["log_path"] is not None else None

    config = load_config(config_path, include_config=include_config)
    log = parse_log(log_path, history_limit=history_limit)

    return {
        "run_dir": artifacts["run_dir"],
        "artifacts": artifacts,
        "config": config,
        "log": log,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    summary = build_summary(
        run_dir=run_dir,
        history_limit=max(args.history_limit, 0),
        include_config=args.include_config,
    )

    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
