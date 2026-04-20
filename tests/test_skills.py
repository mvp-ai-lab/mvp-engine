from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recipe-local skill tests.")
    parser.add_argument(
        "--recipe",
        required=True,
        help="Recipe name under recipes/ or an explicit recipe directory path.",
    )
    parser.add_argument(
        "--language",
        choices=("en", "zh-cn"),
        help="Optional skill language to record in the recipe skill manifest.",
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--skill", help="Run only one recipe-local skill test set.")
    selector.add_argument("--all", action="store_true", help="Run all recipe-local skill test sets.")
    selector.add_argument("--init-manifest", action="store_true", help="Create or sync the recipe skill manifest.")
    selector.add_argument(
        "--mark-not-applicable",
        metavar="SKILL_ID",
        help="Mark one manifest skill as not applicable for this recipe.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        recipe_dir = skill_testing_util.resolve_recipe_dir(args.recipe)
        manifest_relative_path = skill_testing_util.get_recipe_skill_manifest_path(recipe_dir).relative_to(recipe_dir)
        if args.init_manifest:
            skill_testing_util.initialize_recipe_skill_manifest(recipe_dir)
            print(f"initialized {recipe_dir.name}/{manifest_relative_path}")
            return 0

        if args.mark_not_applicable:
            skill_testing_util.mark_manifest_skill_not_applicable(recipe_dir, skill_id=args.mark_not_applicable)
            print(f"marked {args.mark_not_applicable} as not_applicable in {recipe_dir.name}/{manifest_relative_path}")
            return 0

        if args.skill:
            result = run_one_skill(recipe_dir, args.skill, language=args.language)
            print_summary([result])
            return 0 if result.passed else result.returncode

        results = run_all_skills(recipe_dir, language=args.language)
        print_summary(results)
        return 0 if all(result.passed for result in results) else 1
    except SkillTestSpecError as exc:
        print(f"skill-test error: {exc}", file=sys.stderr)
        return 2


def run_one_skill(recipe_dir: Path, skill_id: str, *, language: str | None = None) -> RunResult:
    spec = skill_testing_util.find_recipe_skill_spec(recipe_dir, skill_id)
    language = language or skill_testing_util.detect_skill_language(spec.skill_id)
    skill_testing_util.set_manifest_skill_status(recipe_dir, spec.skill_id, status="applied", language=language)
    print(f"[skill] {spec.skill_id} ({spec.recipe_name})")
    _print_real_env_hint_if_needed(spec, language=language, all_skills=False)

    layer_results: dict[str, bool] = {}
    for layer, paths in spec.required_pytest_paths().items():
        relative_paths = ", ".join(_format_repo_relative_path(path) for path in paths)
        print(f"  - {layer}: {relative_paths}")
        returncode = _run_pytest(paths)
        layer_results[layer] = returncode == 0
        if returncode != 0:
            skill_testing_util.set_manifest_skill_status(
                recipe_dir,
                spec.skill_id,
                status="failed",
                language=language,
                layer_results=layer_results,
            )
            return RunResult(name=spec.skill_id, passed=False, returncode=returncode)

    skill_testing_util.set_manifest_skill_status(
        recipe_dir,
        spec.skill_id,
        status="applied",
        language=language,
        layer_results=layer_results,
    )
    return RunResult(name=spec.skill_id, passed=True, returncode=0)


def run_all_skills(recipe_dir: Path, *, language: str | None = None) -> list[RunResult]:
    specs = skill_testing_util.discover_recipe_skill_specs(recipe_dir)
    if not specs:
        raise SkillTestSpecError(f"Recipe '{recipe_dir.name}' does not define any recipe-local skill tests.")

    if _cuda_unavailable() and any(spec.requirements.gpu_preferred for spec in specs):
        command = skill_testing_util.get_default_skill_test_command(recipe_dir.name, language=language, skill_id=None)
        print(
            "[all-skills] GPU-preferred checks are defined for this recipe. "
            "If local smoke tests fail due to environment limits, rerun in a real environment with:\n"
            f"  {command}"
        )

    results: list[RunResult] = []
    for spec in specs:
        results.append(run_one_skill(recipe_dir, spec.skill_id, language=language))

    all_skills_smoke = skill_testing_util.get_all_skills_smoke_test(recipe_dir)
    if all_skills_smoke is not None:
        print(f"[all-skills] {_format_repo_relative_path(all_skills_smoke)}")
        returncode = _run_pytest((all_skills_smoke,))
        results.append(RunResult(name="all-skills", passed=returncode == 0, returncode=returncode))
        skill_testing_util.set_manifest_all_skills_status(recipe_dir, passed=returncode == 0)
    else:
        skill_testing_util.set_manifest_all_skills_status(recipe_dir, passed=all(result.passed for result in results))

    return results


def print_summary(results: list[RunResult]) -> None:
    print("\nSummary:")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  {status} {result.name}")


def _run_pytest(paths: tuple[Path, ...]) -> int:
    command = [sys.executable, "-m", "pytest", "-q", *[str(path) for path in paths]]
    completed = subprocess.run(command, cwd=skill_testing_util.find_repo_root(paths[0]), check=False)
    return completed.returncode


def _format_repo_relative_path(path: Path) -> str:
    return str(path.relative_to(skill_testing_util.find_repo_root(path)))


def _print_real_env_hint_if_needed(spec, *, language: str | None, all_skills: bool) -> None:
    if not spec.requirements.gpu_preferred:
        return
    if not _cuda_unavailable():
        return

    command = skill_testing_util.get_real_env_command(spec, language=language, all_skills=all_skills)
    print(
        f"[skill] {spec.skill_id} declares gpu_preferred=true. "
        "If local runtime/smoke tests fail because this environment has no usable GPU, "
        "run them in a real environment with:\n"
        f"  {command}"
    )


def _cuda_unavailable() -> bool:
    try:
        import torch
    except Exception:
        return True
    return not torch.cuda.is_available()


if __name__ == "__main__":
    raise SystemExit(main())
