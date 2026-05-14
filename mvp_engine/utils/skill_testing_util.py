"""Recipe-local skill test discovery helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import yaml

SKILL_TESTS_DIRNAME = "skill_tests"
MANIFEST_FILENAME = "skill_manifest.yaml"
ASSERTS_FILENAME = "asserts.py"
CURRENT_SKILL_ENV = "MVP_ENGINE_CURRENT_SKILL"
LAYER_ORDER = ("structure", "smoke", "effectiveness")
RECIPE_LAYER_FILES = {
    "structure": ("test_structure.py",),
    "smoke": ("test_smoke.py",),
}
SKILL_LAYER_FILES = {
    "effectiveness": ("test_effectiveness.py",),
}
DEFAULT_LAYER_FILES = {**RECIPE_LAYER_FILES, **SKILL_LAYER_FILES}


class SkillTestSpecError(ValueError):
    """Raised when recipe-local skill tests are invalid."""


def find_repo_root(start: Path | None = None) -> Path:
    """Locate the repository root by walking upward to ``pyproject.toml``."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent.resolve()

    raise SkillTestSpecError(f"Could not determine repository root from: {current}")


@dataclass(frozen=True)
class RecipeSkillTests:
    """Recipe-local test files for one applied skill."""

    skill_id: str
    recipe_name: str
    recipe_dir: Path
    skill_dir: Path

    def required_layers(self) -> tuple[str, ...]:
        """Return validation layers that have recipe-local test files."""
        return tuple(layer for layer in LAYER_ORDER if layer in RECIPE_LAYER_FILES or self._layer_exists(layer))

    def pytest_paths_for_layer(self, layer: str) -> tuple[Path, ...]:
        """Return pytest files for one validation layer."""
        if layer not in LAYER_ORDER:
            raise SkillTestSpecError(f"Unknown test layer '{layer}'.")

        base_dir = self.recipe_dir / SKILL_TESTS_DIRNAME if layer in RECIPE_LAYER_FILES else self.skill_dir
        paths = tuple((base_dir / relative_path).resolve() for relative_path in DEFAULT_LAYER_FILES[layer])
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise SkillTestSpecError(
                f"Recipe '{self.recipe_name}' skill '{self.skill_id}' is missing {layer} test files: {missing}"
            )
        return paths

    def _layer_exists(self, layer: str) -> bool:
        base_dir = self.recipe_dir / SKILL_TESTS_DIRNAME if layer in RECIPE_LAYER_FILES else self.skill_dir
        return all((base_dir / relative_path).is_file() for relative_path in DEFAULT_LAYER_FILES[layer])


def resolve_recipe_dir(recipe: str, repo_root: Path | None = None) -> Path:
    """Resolve a recipe argument from either a recipe name or a directory path."""
    repo_root = repo_root or find_repo_root()
    recipe_path = Path(recipe)
    if recipe_path.is_absolute() or recipe_path.exists():
        resolved = recipe_path.resolve()
    else:
        resolved = (repo_root / "recipes" / recipe).resolve()

    if not resolved.exists():
        raise SkillTestSpecError(f"Recipe path does not exist: {resolved}")
    if not resolved.is_dir():
        raise SkillTestSpecError(f"Recipe path is not a directory: {resolved}")
    return resolved


def discover_recipe_skill_tests(recipe_dir: Path) -> list[RecipeSkillTests]:
    """Discover recipe-local skill test directories for a recipe."""
    skill_tests_dir = recipe_dir / SKILL_TESTS_DIRNAME
    if not skill_tests_dir.exists():
        return []

    discovered = [
        _build_recipe_skill_tests(skill_dir)
        for skill_dir in sorted(path for path in skill_tests_dir.iterdir() if path.is_dir())
        if (skill_dir / ASSERTS_FILENAME).is_file()
    ]
    return sorted(discovered, key=lambda skill_tests: skill_tests.skill_id)


def find_recipe_skill_tests(recipe_dir: Path, skill_id: str) -> RecipeSkillTests:
    """Load one recipe-local skill test directory by skill id."""
    skill_dir = (recipe_dir / SKILL_TESTS_DIRNAME / skill_id).resolve()
    if not skill_dir.exists():
        raise SkillTestSpecError(f"Recipe '{recipe_dir.name}' does not define skill tests for '{skill_id}'.")
    if not skill_dir.is_dir():
        raise SkillTestSpecError(f"Recipe '{recipe_dir.name}' skill tests path is not a directory: {skill_dir}")
    legacy_files = [
        str(skill_dir / file_name)
        for files in RECIPE_LAYER_FILES.values()
        for file_name in files
        if (skill_dir / file_name).is_file()
    ]
    if legacy_files:
        raise SkillTestSpecError(
            f"Recipe '{recipe_dir.name}' skill '{skill_id}' has recipe-level tests in the skill directory. "
            f"Move these files to '{recipe_dir / SKILL_TESTS_DIRNAME}': {legacy_files}"
        )
    asserts_path = skill_dir / ASSERTS_FILENAME
    if not asserts_path.is_file():
        raise SkillTestSpecError(f"Recipe '{recipe_dir.name}' skill '{skill_id}' is missing {asserts_path.name}.")

    skill_tests = _build_recipe_skill_tests(skill_dir)
    if not skill_tests.required_layers():
        raise SkillTestSpecError(f"Recipe '{recipe_dir.name}' skill '{skill_id}' does not define any test files.")
    return skill_tests


def get_recipe_skill_manifest_path(recipe_dir: Path) -> Path:
    """Return the canonical manifest path for a recipe."""
    return (recipe_dir / SKILL_TESTS_DIRNAME / MANIFEST_FILENAME).resolve()


def get_ordered_skill_asserts(
    recipe_dir: Path,
    *,
    current_skill_id: str | None = None,
) -> tuple[tuple[str, Path], ...]:
    """Return installed skill asserts in manifest order, with the current skill last."""
    current_skill_id = current_skill_id or os.environ.get(CURRENT_SKILL_ENV)
    manifest_path = get_recipe_skill_manifest_path(recipe_dir)
    if manifest_path.exists():
        manifest = load_recipe_skill_manifest(recipe_dir, create_if_missing=False)
        skill_ids = list(manifest["skills"])
    else:
        skill_ids = []

    if current_skill_id:
        skill_ids = [skill_id for skill_id in skill_ids if skill_id != current_skill_id]
        skill_ids.append(current_skill_id)

    return tuple((skill_id, get_skill_asserts_path(recipe_dir, skill_id)) for skill_id in skill_ids)


def get_skill_asserts_path(recipe_dir: Path, skill_id: str) -> Path:
    """Return and validate one skill's assertion module path."""
    asserts_path = (recipe_dir / SKILL_TESTS_DIRNAME / skill_id / ASSERTS_FILENAME).resolve()
    if not asserts_path.is_file():
        raise SkillTestSpecError(f"Recipe '{recipe_dir.name}' skill '{skill_id}' is missing {ASSERTS_FILENAME}.")
    return asserts_path


def initialize_recipe_skill_manifest(recipe_dir: Path, repo_root: Path | None = None) -> dict:
    """Create an empty recipe skill manifest and return its contents."""
    del repo_root
    manifest = {"skills": []}
    save_recipe_skill_manifest(recipe_dir, manifest)
    return manifest


def load_recipe_skill_manifest(
    recipe_dir: Path,
    repo_root: Path | None = None,
    *,
    create_if_missing: bool = True,
) -> dict:
    """Load the recipe skill manifest, optionally creating it when absent."""
    del repo_root
    manifest_path = get_recipe_skill_manifest_path(recipe_dir)
    if not manifest_path.exists():
        if not create_if_missing:
            relative_path = get_recipe_skill_manifest_path(recipe_dir).relative_to(recipe_dir)
            raise SkillTestSpecError(f"Recipe '{recipe_dir.name}' does not have {relative_path}.")
        return initialize_recipe_skill_manifest(recipe_dir)

    manifest = _load_manifest_file(manifest_path)
    _validate_installed_skill_manifest(manifest, manifest_path)
    return manifest


def save_recipe_skill_manifest(recipe_dir: Path, manifest: dict) -> None:
    """Persist a recipe skill manifest."""
    manifest_path = get_recipe_skill_manifest_path(recipe_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def record_manifest_skill_installed(
    recipe_dir: Path,
    skill_id: str,
    *,
    repo_root: Path | None = None,
) -> dict:
    """Record a skill as installed after all required recipe-local tests pass."""
    manifest = load_recipe_skill_manifest(recipe_dir, repo_root)
    skills = manifest["skills"]
    if skill_id not in skills:
        skills.append(skill_id)

    save_recipe_skill_manifest(recipe_dir, manifest)
    return manifest


def get_default_skill_test_command(
    recipe_name: str,
    *,
    skill_id: str,
    layer: str | None = None,
) -> str:
    """Build the default CLI command for recipe-local skill tests."""
    if not skill_id.strip():
        raise SkillTestSpecError("A skill test command requires a non-empty skill_id.")
    if layer is not None and layer not in LAYER_ORDER:
        raise SkillTestSpecError(f"Unknown test layer '{layer}'.")

    parts = ["python", "-m", "tests.test_skills", "--recipe", recipe_name]
    parts.extend(["--skill", skill_id])
    if layer is not None:
        parts.extend(["--layer", layer])
    return " ".join(parts)


def build_real_env_required_message(*, command: str, reason: str) -> str:
    """Build a consistent actionable message for tests that need a real environment."""
    return f"{reason}\nRun this test in a real environment with:\n{command}"


def raise_real_env_required(*, command: str, reason: str) -> NoReturn:
    """Fail with an actionable real-environment command instead of skipping."""
    raise AssertionError(build_real_env_required_message(command=command, reason=reason))


def require_cuda_or_real_env(*, command: str, reason: str) -> None:
    """Require CUDA for a test, failing with a real-environment command when unavailable."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - defensive import guard
        raise_real_env_required(
            command=command,
            reason=f"{reason} CUDA check failed because torch could not be imported: {exc}",
        )

    if not torch.cuda.is_available():
        raise_real_env_required(command=command, reason=reason)


def _build_recipe_skill_tests(skill_dir: Path) -> RecipeSkillTests:
    recipe_dir = skill_dir.parents[1]
    return RecipeSkillTests(
        skill_id=skill_dir.name,
        recipe_name=recipe_dir.name,
        recipe_dir=recipe_dir.resolve(),
        skill_dir=skill_dir.resolve(),
    )


def _load_manifest_file(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise SkillTestSpecError(f"Manifest must be a mapping: {manifest_path}")
    return payload


def _validate_installed_skill_manifest(manifest: dict, manifest_path: Path) -> None:
    if set(manifest) != {"skills"}:
        raise SkillTestSpecError(f"Manifest must contain only a 'skills' list of skill names: {manifest_path}")

    skills = manifest.get("skills")
    if not isinstance(skills, list) or not all(isinstance(skill_id, str) and skill_id for skill_id in skills):
        raise SkillTestSpecError(f"Manifest must contain only a 'skills' list of skill names: {manifest_path}")
    if len(skills) != len(set(skills)):
        raise SkillTestSpecError(f"Manifest contains duplicate skill names: {manifest_path}")
