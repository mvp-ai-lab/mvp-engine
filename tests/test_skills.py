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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    selector.add_argument("--init-manifest", action="store_true", help="Create or sync the recipe skill manifest.")
    selector.add_argument(
        "--mark-not-applicable",
        metavar="SKILL_ID",
        help="Mark one manifest skill as not applicable for this recipe.",
    )
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
        manifest_relative_path = skill_testing_util.get_recipe_skill_manifest_path(recipe_dir).relative_to(recipe_dir)
        if args.init_manifest:
            skill_testing_util.initialize_recipe_skill_manifest(recipe_dir)
            print(f"initialized {recipe_dir.name}/{manifest_relative_path}")
            return 0

        if args.mark_not_applicable:
            skill_testing_util.mark_manifest_skill_not_applicable(recipe_dir, skill_id=args.mark_not_applicable)
            print(f"marked {args.mark_not_applicable} as not_applicable in {recipe_dir.name}/{manifest_relative_path}")
            return 0

        if args.layer and not args.skill:
            raise SkillTestSpecError("--layer can only be used together with --skill.")

        if args.skill:
            result = run_one_skill(recipe_dir, args.skill, language=args.language, layer=args.layer)
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
    language: str | None = None,
    layer: str | None = None,
) -> RunResult:
    spec = skill_testing_util.find_recipe_skill_spec(recipe_dir, skill_id)
    language = language or skill_testing_util.detect_skill_language(spec.skill_id)
    print(f"[skill] {spec.skill_id} ({spec.recipe_name})")
    _print_real_env_hint_if_needed(spec, language=language, layer=layer)

    required_layers = spec.requirements.required_layers()
    if layer is not None and layer not in required_layers:
        raise SkillTestSpecError(f"Skill '{spec.skill_id}' does not require the '{layer}' validation layer.")

    requested_layers = (layer,) if layer is not None else required_layers
    manifest = skill_testing_util.load_recipe_skill_manifest(recipe_dir)
    entry = skill_testing_util.ensure_manifest_skill_entry(manifest, spec.skill_id)
    layer_statuses = dict(entry["last_validated"])

    for requested_layer in requested_layers:
        paths = spec.pytest_paths_for_layer(requested_layer)
        relative_paths = ", ".join(_format_repo_relative_path(path) for path in paths)
        print(f"  - {requested_layer}: {relative_paths}")
        returncode = _run_pytest(paths)
        layer_statuses[requested_layer] = "passed" if returncode == 0 else "failed"
        status = skill_testing_util.resolve_manifest_skill_status(
            layer_statuses=layer_statuses,
            required_layers=required_layers,
        )
        layer_results = {
            layer_name: (layer_status == "passed")
            for layer_name, layer_status in layer_statuses.items()
            if layer_name in skill_testing_util.LAYER_ORDER and layer_status != "not_run"
        }
        skill_testing_util.set_manifest_skill_status(
            recipe_dir,
            spec.skill_id,
            status=status,
            language=language,
            layer_results=layer_results,
        )
        if returncode != 0:
            result_name = f"{spec.skill_id}:{requested_layer}" if layer is not None else spec.skill_id
            return RunResult(name=result_name, passed=False, returncode=returncode)

    result_name = f"{spec.skill_id}:{layer}" if layer is not None else spec.skill_id
    return RunResult(name=result_name, passed=True, returncode=0)


def print_summary(results: list[RunResult]) -> None:
    print("\nSummary:")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  {status} {result.name}")


def _run_pytest(paths: tuple[Path, ...]) -> int:
    """Run recipe-local skill validation as direct pytest file execution."""
    command = [sys.executable, "-m", "pytest", "-q", *[str(path) for path in paths]]
    completed = subprocess.run(command, cwd=skill_testing_util.find_repo_root(paths[0]), check=False)
    return completed.returncode


def _format_repo_relative_path(path: Path) -> str:
    return str(path.relative_to(skill_testing_util.find_repo_root(path)))


def _print_real_env_hint_if_needed(spec, *, language: str | None, layer: str | None = None) -> None:
    if not spec.requirements.gpu_preferred:
        return
    if not _cuda_unavailable():
        return

    command = skill_testing_util.get_real_env_command(spec, language=language, layer=layer)
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
