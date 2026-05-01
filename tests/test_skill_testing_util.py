from __future__ import annotations

from pathlib import Path

from mvp_engine.utils import skill_testing_util


def _write_minimal_repo(root: Path, *, skill_body: str) -> Path:
    (root / "pyproject.toml").write_text("[project]\nname = 'tmp-skill-tests'\n", encoding="utf-8")
    skill_dir = root / "skills" / "training" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(skill_body, encoding="utf-8")

    recipe_dir = root / "recipes" / "demo_recipe"
    recipe_skill_dir = recipe_dir / "skill_tests" / "demo-skill"
    recipe_skill_dir.mkdir(parents=True)
    (recipe_skill_dir / "test_spec.yaml").write_text(
        "skill_id: demo-skill\nrequires:\n  structure: true\n  runtime: true\n  smoke: true\n  effectiveness: false\n",
        encoding="utf-8",
    )
    return recipe_dir


def test_manifest_marks_effectiveness_not_applicable_when_skill_does_not_declare_it(tmp_path: Path) -> None:
    recipe_dir = _write_minimal_repo(
        tmp_path, skill_body="# Demo skill\n\nThis mentions effectiveness only as prose.\n"
    )

    spec = skill_testing_util.find_recipe_skill_spec(recipe_dir, "demo-skill")
    manifest = skill_testing_util.initialize_recipe_skill_manifest(recipe_dir, repo_root=tmp_path)

    assert not skill_testing_util.skill_declares_effectiveness("demo-skill", repo_root=tmp_path)
    assert spec.requirements.required_layers() == ("structure", "runtime", "smoke")
    assert manifest["skills"]["demo-skill"]["last_validated"]["effectiveness"] == "not_applicable"


def test_explicit_effectiveness_requirement_enables_fourth_layer(tmp_path: Path) -> None:
    recipe_dir = _write_minimal_repo(
        tmp_path,
        skill_body="# Demo skill\n\nAdd `test_effectiveness.py` for the effectiveness layer.\n",
    )
    spec_path = recipe_dir / "skill_tests" / "demo-skill" / "test_spec.yaml"
    spec_path.write_text(
        "skill_id: demo-skill\nrequires:\n  structure: true\n  runtime: true\n  smoke: true\n  effectiveness: true\n",
        encoding="utf-8",
    )

    spec = skill_testing_util.find_recipe_skill_spec(recipe_dir, "demo-skill")
    manifest = skill_testing_util.initialize_recipe_skill_manifest(recipe_dir, repo_root=tmp_path)

    assert skill_testing_util.skill_declares_effectiveness("demo-skill", repo_root=tmp_path)
    assert spec.requirements.required_layers() == ("structure", "runtime", "smoke", "effectiveness")
    assert manifest["skills"]["demo-skill"]["last_validated"]["effectiveness"] == "not_run"
    assert (
        skill_testing_util.resolve_manifest_skill_status(
            layer_statuses={
                "structure": "passed",
                "runtime": "passed",
                "smoke": "passed",
                "effectiveness": "not_run",
            },
            required_layers=spec.requirements.required_layers(),
        )
        == "pending"
    )
