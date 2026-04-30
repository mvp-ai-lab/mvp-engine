#!/usr/bin/env python3
"""Parse training logs and plot train/loss curves for one or more runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LINE_PATTERN = re.compile(r"Step\s+(\d+)\s+\|\|\s+train/loss:\s*([0-9eE+\-.]+)")


def extract_steps_and_loss(log_path: Path) -> tuple[list[int], list[float]]:
    """Return parsed step and train/loss values from a training log."""
    steps: list[int] = []
    train_losses: list[float] = []

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = LINE_PATTERN.search(line)
            if match is None:
                continue
            steps.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))

    return steps, train_losses


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    if window > len(values):
        return []

    out: list[float] = []
    current = sum(values[:window])
    out.append(current / window)
    for i in range(window, len(values)):
        current += values[i] - values[i - window]
        out.append(current / window)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/loss curves from one or more training logs.")
    parser.add_argument("logs", nargs="+", type=Path, help="Paths to log files.")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend labels for each log. If omitted, filenames are used.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving-average window size for smoothing (default: 1, no smoothing).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("training_curve_comparison.png"),
        help="Output path for the PNG figure.",
    )
    parser.add_argument(
        "--dump-json",
        type=Path,
        default=None,
        help="Optional output JSON path with parsed `steps` and `train_losses` lists per log.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window in addition to saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.logs):
        raise ValueError("`--labels` must have the same number of values as provided log files.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it with `uv pip install matplotlib` or `pip install matplotlib`."
        ) from exc

    parsed: dict[str, dict[str, list[float] | list[int]]] = {}

    plt.figure(figsize=(10, 6))
    for i, log_path in enumerate(args.logs):
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")

        steps, losses = extract_steps_and_loss(log_path)
        if not steps:
            raise ValueError(f"No `Step ... train/loss` entries found in: {log_path}")

        label = args.labels[i] if args.labels is not None else log_path.stem
        parsed[str(log_path)] = {"steps": steps, "train_losses": losses}

        if args.smooth_window > 1:
            if args.smooth_window > len(losses):
                raise ValueError(
                    f"`--smooth-window` ({args.smooth_window}) is larger than points ({len(losses)}) in: {log_path}"
                )
            smoothed = moving_average(losses, args.smooth_window)
            smoothed_steps = steps[args.smooth_window - 1 :]
            plt.plot(smoothed_steps, smoothed, label=f"{label} (ma={args.smooth_window})")
        else:
            plt.plot(steps, losses, label=label)

        print(f"{log_path}: points={len(steps)}, first_step={steps[0]}, last_step={steps[-1]}")

    plt.xlabel("Step")
    plt.ylabel("train/loss")
    plt.title("Training Curve Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to: {args.out}")

    if args.dump_json is not None:
        args.dump_json.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        print(f"Saved parsed data to: {args.dump_json}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
