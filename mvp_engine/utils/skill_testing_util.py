"""Recipe-local skill test discovery helpers.

This module keeps the skill-test convention intentionally small:

- Tests live under ``recipes/<recipe>/skill_tests/<skill_id>/``.
- Each applied skill owns a ``test_spec.yaml`` plus its pytest files.
- A recipe-local ``recipes/<recipe>/skill_tests/skill_manifest.yaml`` tracks which training-related
  skills are pending, applied, failed, or not applicable.
- The global CLI runs one skill at a time, dispatches pytest for the required
  layers, and updates the manifest with per-skill validation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

import yaml

SKILL_TESTS_DIRNAME = "skill_tests"
SPEC_FILENAME = "test_spec.yaml"
MANIFEST_FILENAME = "skill_manifest.yaml"
LAYER_ORDER = ("structure", "runtime", "smoke")
MANIFEST_LAYER_STATUSES = ("not_run", "passed", "failed")
MANIFEST_SKILL_STATUSES = ("pending", "applied", "failed", "not_applicable")
MANAGED_SKILL_CATEGORIES = ("training", "parallel", "model")
DEFAULT_LAYER_FILES = {
    "structure": ("test_structure.py",),
    "runtime": ("test_runtime.py",),
    "smoke": ("test_smoke.py",),
}


class SkillTestSpecError(ValueError):
    """Raised when a recipe-local skill test spec is invalid."""


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
class SkillTestRequirements:
    structure: bool = True
    runtime: bool = False
    smoke: bool = False
    gpu_preferred: bool = False

    @classmethod
    def from_raw(cls, raw: object) -> "SkillTestRequirements":
        raw_dict = raw or {}
        if not isinstance(raw_dict, dict):
            raise SkillTestSpecError("'requires' must be a mapping.")

        defaults = cls()
        normalized = {}
        for key in ("structure", "runtime", "smoke", "gpu_preferred"):
            normalized[key] = bool(raw_dict.get(key, getattr(defaults, key)))
        return cls(**normalized)

    def required_layers(self) -> tuple[str, ...]:
        return tuple(layer for layer in LAYER_ORDER if getattr(self, layer))


@dataclass(frozen=True)
class RecipeSkillTestSpec:
    skill_id: str
    recipe_name: str
    recipe_dir: Path
    spec_path: Path
    requirements: SkillTestRequirements
    allowed_files: tuple[str, ...] = ()
    forbidden_patterns: tuple[str, ...] = ()
    test_files: dict[str, tuple[str, ...]] = field(default_factory=dict)
    real_env_commands: dict[str, str] = field(default_factory=dict)

    @property
    def skill_dir(self) -> Path:
        return self.spec_path.parent

    def pytest_paths_for_layer(self, layer: str) -> tuple[Path, ...]:
        if layer not in LAYER_ORDER:
            raise SkillTestSpecError(f"Unknown test layer '{layer}'.")

        raw_paths = self.test_files.get(layer, DEFAULT_LAYER_FILES[layer])
        resolved = tuple((self.skill_dir / relative_path).resolve() for relative_path in raw_paths)
        missing = [str(path) for path in resolved if not path.exists()]
        if missing:
            raise SkillTestSpecError(
                f"Recipe '{self.recipe_name}' skill '{self.skill_id}' is missing {layer} test files: {missing}"
            )
        return resolved

    def required_pytest_paths(self) -> dict[str, tuple[Path, ...]]:
        return {layer: self.pytest_paths_for_layer(layer) for layer in self.requirements.required_layers()}


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


def discover_recipe_skill_specs(recipe_dir: Path) -> list[RecipeSkillTestSpec]:
    """Discover all recipe-local skill specs for a recipe."""
    skill_tests_dir = recipe_dir / SKILL_TESTS_DIRNAME
    if not skill_tests_dir.exists():
        return []

    specs = [
        load_recipe_skill_spec(spec_path)
        for spec_path in sorted(skill_tests_dir.glob(f"*/{SPEC_FILENAME}"))
        if spec_path.is_file()
    ]
    return sorted(specs, key=lambda spec: spec.skill_id)


def load_recipe_skill_spec(spec_path: Path) -> RecipeSkillTestSpec:
    """Load one recipe-local skill test spec from YAML."""
    with spec_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise SkillTestSpecError(f"Spec must be a mapping: {spec_path}")

    skill_id = payload.get("skill_id")
    if not isinstance(skill_id, str) or not skill_id.strip():
        raise SkillTestSpecError(f"'skill_id' must be a non-empty string: {spec_path}")

    recipe_dir = spec_path.parents[2]
    recipe_name = recipe_dir.name

    explicit_skill_dir = spec_path.parent.name
    if explicit_skill_dir != skill_id:
        raise SkillTestSpecError(
            f"Spec directory '{explicit_skill_dir}' does not match skill_id '{skill_id}' in {spec_path}"
        )

    requirements = SkillTestRequirements.from_raw(payload.get("requires"))
    test_files = _normalize_test_files(payload.get("test_files", {}), spec_path)

    return RecipeSkillTestSpec(
        skill_id=skill_id,
        recipe_name=recipe_name,
        recipe_dir=recipe_dir.resolve(),
        spec_path=spec_path.resolve(),
        requirements=requirements,
        allowed_files=_normalize_string_list(payload.get("allowed_files")),
        forbidden_patterns=_normalize_string_list(payload.get("forbidden_patterns")),
        test_files=test_files,
        real_env_commands=_normalize_command_map(payload.get("real_env_commands"), spec_path),
    )


def find_recipe_skill_spec(recipe_dir: Path, skill_id: str) -> RecipeSkillTestSpec:
    """Load one recipe-local skill spec by skill id."""
    spec_path = recipe_dir / SKILL_TESTS_DIRNAME / skill_id / SPEC_FILENAME
    if not spec_path.exists():
        raise SkillTestSpecError(
            f"Recipe '{recipe_dir.name}' does not define skill tests for '{skill_id}' at {spec_path}"
        )
    return load_recipe_skill_spec(spec_path)


def get_recipe_skill_manifest_path(recipe_dir: Path) -> Path:
    """Return the canonical manifest path for a recipe."""
    return (recipe_dir / SKILL_TESTS_DIRNAME / MANIFEST_FILENAME).resolve()


def discover_managed_skill_ids(repo_root: Path | None = None) -> list[str]:
    """Return skill ids that should appear in recipe skill manifests by default."""
    repo_root = repo_root or find_repo_root()
    skills_root = repo_root / "skills" / "en"
    skill_ids: set[str] = set()

    for category in MANAGED_SKILL_CATEGORIES:
        category_dir = skills_root / category
        if not category_dir.exists():
            continue

        for skill_doc in category_dir.glob("*/SKILL.md"):
            skill_ids.add(skill_doc.parent.name)

    return sorted(skill_ids)


def initialize_recipe_skill_manifest(recipe_dir: Path, repo_root: Path | None = None) -> dict:
    """Create or sync the recipe skill manifest and return its contents."""
    repo_root = repo_root or find_repo_root(recipe_dir)
    manifest_path = get_recipe_skill_manifest_path(recipe_dir)
    existing = _load_manifest_file(manifest_path) if manifest_path.exists() else None
    manifest = existing or {
        "schema_version": 1,
        "recipe": recipe_dir.name,
        "skills": {},
    }

    manifest["recipe"] = recipe_dir.name
    manifest["schema_version"] = 1
    manifest.setdefault("skills", {})

    for skill_id in discover_managed_skill_ids(repo_root):
        ensure_manifest_skill_entry(manifest, skill_id)

    save_recipe_skill_manifest(recipe_dir, manifest)
    return manifest


def load_recipe_skill_manifest(
    recipe_dir: Path,
    repo_root: Path | None = None,
    *,
    create_if_missing: bool = True,
) -> dict:
    """Load the recipe skill manifest, optionally creating it when absent."""
    repo_root = repo_root or find_repo_root(recipe_dir)
    manifest_path = get_recipe_skill_manifest_path(recipe_dir)
    if not manifest_path.exists():
        if not create_if_missing:
            relative_path = get_recipe_skill_manifest_path(recipe_dir).relative_to(recipe_dir)
            raise SkillTestSpecError(f"Recipe '{recipe_dir.name}' does not have {relative_path}.")
        return initialize_recipe_skill_manifest(recipe_dir, repo_root)

    manifest = _load_manifest_file(manifest_path)
    manifest["schema_version"] = 1
    manifest["recipe"] = recipe_dir.name
    manifest.setdefault("skills", {})

    for skill_id in discover_managed_skill_ids(repo_root):
        ensure_manifest_skill_entry(manifest, skill_id)

    save_recipe_skill_manifest(recipe_dir, manifest)
    return manifest


def save_recipe_skill_manifest(recipe_dir: Path, manifest: dict) -> None:
    """Persist a recipe skill manifest."""
    manifest_path = get_recipe_skill_manifest_path(recipe_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def ensure_manifest_skill_entry(manifest: dict, skill_id: str) -> dict:
    """Ensure a manifest contains a normalized entry for one skill."""
    skills = manifest.setdefault("skills", {})
    entry = skills.setdefault(skill_id, {})
    entry.setdefault("status", "pending")
    if entry["status"] not in MANIFEST_SKILL_STATUSES:
        raise SkillTestSpecError(f"Invalid manifest skill status for '{skill_id}': {entry['status']}")

    entry.setdefault("language", None)
    last_validated = entry.setdefault("last_validated", {})
    for layer in LAYER_ORDER:
        last_validated.setdefault(layer, "not_run")
        if last_validated[layer] not in MANIFEST_LAYER_STATUSES:
            raise SkillTestSpecError(
                f"Invalid manifest validation status for '{skill_id}.{layer}': {last_validated[layer]}"
            )

    return entry


def set_manifest_skill_status(
    recipe_dir: Path,
    skill_id: str,
    *,
    status: str,
    language: str | None = None,
    layer_results: dict[str, bool] | None = None,
    repo_root: Path | None = None,
) -> dict:
    """Update one skill entry in the recipe manifest."""
    if status not in MANIFEST_SKILL_STATUSES:
        raise SkillTestSpecError(f"Unsupported manifest skill status '{status}' for '{skill_id}'.")

    repo_root = repo_root or find_repo_root(recipe_dir)
    manifest = load_recipe_skill_manifest(recipe_dir, repo_root)
    entry = ensure_manifest_skill_entry(manifest, skill_id)
    entry["status"] = status
    if language is not None:
        entry["language"] = language

    if layer_results:
        for layer, passed in layer_results.items():
            if layer not in LAYER_ORDER:
                raise SkillTestSpecError(f"Unsupported validation layer '{layer}' for '{skill_id}'.")
            entry["last_validated"][layer] = "passed" if passed else "failed"

    save_recipe_skill_manifest(recipe_dir, manifest)
    return manifest


def detect_skill_language(skill_id: str, repo_root: Path | None = None) -> str | None:
    """Infer the preferred language for a skill id from repo docs."""
    repo_root = repo_root or find_repo_root()
    zh_path = list((repo_root / "skills" / "zh-cn").glob(f"*/{skill_id}/SKILL.md"))
    en_path = list((repo_root / "skills" / "en").glob(f"*/{skill_id}/SKILL.md"))
    if zh_path and not en_path:
        return "zh-cn"
    if en_path and not zh_path:
        return "en"
    return None


def mark_manifest_skill_not_applicable(
    recipe_dir: Path,
    skill_id: str,
    *,
    repo_root: Path | None = None,
) -> dict:
    """Mark a manifest skill as not applicable for a recipe."""
    repo_root = repo_root or find_repo_root(recipe_dir)
    return set_manifest_skill_status(recipe_dir, skill_id, status="not_applicable", repo_root=repo_root)


def get_default_skill_test_command(
    recipe_name: str,
    *,
    skill_id: str,
    language: str | None = None,
    layer: str | None = None,
) -> str:
    """Build the default CLI command for recipe-local skill tests."""
    if not skill_id.strip():
        raise SkillTestSpecError("A skill test command requires a non-empty skill_id.")
    if layer is not None and layer not in LAYER_ORDER:
        raise SkillTestSpecError(f"Unknown test layer '{layer}'.")

    parts = ["python", "-m", "tests.test_skills", "--recipe", recipe_name]
    if language:
        parts.extend(["--language", language])
    parts.extend(["--skill", skill_id])
    if layer is not None:
        parts.extend(["--layer", layer])
    return " ".join(parts)


def get_real_env_command(
    spec: RecipeSkillTestSpec,
    *,
    language: str | None = None,
    layer: str | None = None,
) -> str:
    """Resolve the best available real-environment command for a spec."""
    command = spec.real_env_commands.get("skill")
    if command:
        return command
    return get_default_skill_test_command(
        spec.recipe_name,
        skill_id=spec.skill_id,
        language=language,
        layer=layer,
    )


def resolve_manifest_skill_status(*, layer_statuses: dict[str, str], required_layers: tuple[str, ...]) -> str:
    """Derive the manifest skill status from per-layer validation states."""
    if any(layer_statuses.get(layer) == "failed" for layer in required_layers):
        return "failed"
    if all(layer_statuses.get(layer) == "passed" for layer in required_layers):
        return "applied"
    return "pending"


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


def _load_manifest_file(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise SkillTestSpecError(f"Manifest must be a mapping: {manifest_path}")
    return payload


def _normalize_string_list(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise SkillTestSpecError("List fields must be string lists.")
    return tuple(raw)


def _normalize_command_map(raw: object, spec_path: Path) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SkillTestSpecError(f"'real_env_commands' must be a mapping: {spec_path}")

    normalized: dict[str, str] = {}
    for key, value in raw.items():
        if key != "skill":
            raise SkillTestSpecError(f"Unknown real_env_commands key '{key}' in {spec_path}")
        if not isinstance(value, str) or not value.strip():
            raise SkillTestSpecError(f"'real_env_commands.{key}' must be a non-empty string in {spec_path}")
        normalized[key] = value.strip()
    return normalized


def _normalize_test_files(raw: object, spec_path: Path) -> dict[str, tuple[str, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SkillTestSpecError(f"'test_files' must be a mapping: {spec_path}")

    normalized: dict[str, tuple[str, ...]] = {}
    for layer, value in raw.items():
        if layer not in LAYER_ORDER:
            raise SkillTestSpecError(f"Unknown test layer '{layer}' in {spec_path}")
        if isinstance(value, str):
            normalized[layer] = (value,)
            continue
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            normalized[layer] = tuple(value)
            continue
        raise SkillTestSpecError(f"'test_files.{layer}' must be a string or string list in {spec_path}")
    return normalized
