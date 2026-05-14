from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from mvp_engine.utils import skill_testing_util
from mvp_engine.utils.skill_testing_util import SkillTestSpecError


@dataclass(frozen=True)
class RunResult:
    name: str
    passed: bool
    returncode: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recipe-local skill tests.")
    parser.add_argument(
        "--recipe",
        required=True,
        help="Recipe name under recipes/ or an explicit recipe directory path.",
    )
    parser.add_argument("--skill", required=True, help="Run one recipe-local skill test set.")
    parser.add_argument(
        "--layer",
        choices=skill_testing_util.LAYER_ORDER,
        help="Run only one validation layer for the selected skill.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        recipe_dir = skill_testing_util.resolve_recipe_dir(args.recipe)
        result = run_one_skill(recipe_dir, args.skill, layer=args.layer)
        print_summary([result])
        return 0 if result.passed else result.returncode

    except SkillTestSpecError as exc:
        print(f"skill-test error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError("unreachable")


def run_one_skill(
    recipe_dir: Path,
    skill_id: str,
    *,
    layer: str | None = None,
) -> RunResult:
    skill_tests = skill_testing_util.find_recipe_skill_tests(recipe_dir, skill_id)
    print(f"[skill] {skill_tests.skill_id} ({skill_tests.recipe_name})")

    required_layers = skill_tests.required_layers()
    if layer is not None and layer not in required_layers:
        raise SkillTestSpecError(f"Skill '{skill_tests.skill_id}' does not define the '{layer}' validation layer.")

    requested_layers = (layer,) if layer is not None else required_layers

    for requested_layer in requested_layers:
        paths = skill_tests.pytest_paths_for_layer(requested_layer)
        relative_paths = ", ".join(_format_repo_relative_path(path) for path in paths)
        print(f"  - {requested_layer}: {relative_paths}")
        returncode = _run_pytest(paths, skill_tests.skill_id)
        if returncode != 0:
            result_name = f"{skill_tests.skill_id}:{requested_layer}" if layer is not None else skill_tests.skill_id
            return RunResult(name=result_name, passed=False, returncode=returncode)

    if layer is None:
        skill_testing_util.record_manifest_skill_installed(recipe_dir, skill_tests.skill_id)

    result_name = f"{skill_tests.skill_id}:{layer}" if layer is not None else skill_tests.skill_id
    return RunResult(name=result_name, passed=True, returncode=0)


def print_summary(results: list[RunResult]) -> None:
    print("\nSummary:")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  {status} {result.name}")


def _run_pytest(paths: tuple[Path, ...], skill_id: str) -> int:
    """Run recipe-local skill validation as direct pytest file execution."""
    command = [sys.executable, "-m", "pytest", "-q", *[str(path) for path in paths]]
    env = os.environ.copy()
    env[skill_testing_util.CURRENT_SKILL_ENV] = skill_id
    completed = subprocess.run(command, cwd=skill_testing_util.find_repo_root(paths[0]), env=env, check=False)
    return completed.returncode


def _format_repo_relative_path(path: Path) -> str:
    return str(path.relative_to(skill_testing_util.find_repo_root(path)))


if __name__ == "__main__":
    raise SystemExit(main())
