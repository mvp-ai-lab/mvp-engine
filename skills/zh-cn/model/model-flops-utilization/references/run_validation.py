"""Generate MFU skill validation template and dry-run execution plan."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

RESULT_FIELDS = [
    "prompt_id",
    "model_name",
    "expected_trigger",
    "actual_trigger",
    "contract_pass",
    "notes",
]


def _load_cases():
    import importlib.util
    import sys

    this_dir = Path(__file__).resolve().parent
    case_file = this_dir / "validation_cases.py"
    spec = importlib.util.spec_from_file_location("validation_cases", case_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load validation cases from {case_file}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.CASES


def _write_template(output_path: Path, cases) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "prompt_id": case.prompt_id,
                    "model_name": case.model_name,
                    "expected_trigger": case.expected_trigger,
                    "actual_trigger": "",
                    "contract_pass": "",
                    "notes": "",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate validation result template for MFU skill checks.")
    parser.add_argument(
        "--output",
        default="validation_results_template.csv",
        help="Output CSV path for result template.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print execution plan only (no external calls); still writes template CSV.",
    )
    args = parser.parse_args()

    cases = _load_cases()
    out_path = Path(args.output).resolve()

    print(f"Total cases: {len(cases)}")
    for case in cases:
        print(f"[{case.prompt_id}] {case.case_type} | {case.model_name} | expected={case.expected_trigger}")

    _write_template(out_path, cases)
    print(f"Result template written: {out_path}")

    if args.dry_run:
        print("Dry-run mode: no external routing/execution performed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
