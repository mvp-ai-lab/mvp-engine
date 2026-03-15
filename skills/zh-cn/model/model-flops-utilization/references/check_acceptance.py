"""Check MFU skill validation acceptance criteria from result CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _load_case_type_map():
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
    return {c.prompt_id: (c.case_type, c.expected_trigger) for c in mod.CASES}


def _as_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "t", "yes", "y", "pass", "passed"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check acceptance criteria for MFU skill validation results.")
    parser.add_argument("--input", required=True, help="Input CSV path from run_validation.py template.")
    args = parser.parse_args()

    case_type_map = _load_case_type_map()
    input_path = Path(args.input).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Result file not found: {input_path}")

    impl_total = 0
    impl_pass = 0
    boundary_total = 0
    boundary_pass = 0

    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_id = (row.get("prompt_id") or "").strip()
            actual_trigger = (row.get("actual_trigger") or "").strip()
            contract_pass = _as_bool(row.get("contract_pass") or "")

            if prompt_id not in case_type_map:
                continue

            case_type, expected_trigger = case_type_map[prompt_id]
            if case_type == "implementation":
                impl_total += 1
                if actual_trigger == expected_trigger and contract_pass:
                    impl_pass += 1
            elif case_type == "boundary":
                boundary_total += 1
                if contract_pass:
                    boundary_pass += 1

    overall_total = impl_total + boundary_total
    overall_pass = impl_pass + boundary_pass

    print(f"Implementation: {impl_pass}/{impl_total}")
    print(f"Boundary: {boundary_pass}/{boundary_total}")
    print(f"Overall: {overall_pass}/{overall_total}")

    impl_ok = impl_total >= 5 and impl_pass == 5
    boundary_ok = boundary_total >= 5 and boundary_pass >= 4
    overall_ok = overall_total >= 10 and overall_pass >= 8

    print(f"Gate implementation (5/5): {'PASS' if impl_ok else 'FAIL'}")
    print(f"Gate boundary (>=4/5): {'PASS' if boundary_ok else 'FAIL'}")
    print(f"Gate overall (>=8/10): {'PASS' if overall_ok else 'FAIL'}")

    final_ok = impl_ok and boundary_ok and overall_ok
    print(f"Acceptance: {'PASS' if final_ok else 'FAIL'}")
    return 0 if final_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
