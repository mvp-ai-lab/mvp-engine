import importlib
import subprocess

from mvp_engine import launch


def test_import_recipe_modules_respects_nested_gitignore(tmp_path, monkeypatch):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)

    recipes_root = tmp_path / "recipes"
    recipe_dir = recipes_root / "demo"
    (tmp_path / "mvp_engine").mkdir()
    recipes_root.mkdir()
    recipe_dir.mkdir()
    (tmp_path / "mvp_engine" / "__init__.py").write_text("", encoding="utf-8")
    (recipes_root / "__init__.py").write_text("", encoding="utf-8")
    (recipe_dir / "__init__.py").write_text("", encoding="utf-8")
    (recipe_dir / "keep.py").write_text("", encoding="utf-8")
    (recipe_dir / "root_ignored.py").write_text("", encoding="utf-8")
    (recipe_dir / "nested").mkdir()
    (recipe_dir / "nested" / "keep_nested.py").write_text("", encoding="utf-8")
    (recipe_dir / "nested" / "ignored_nested.py").write_text("", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("recipes/demo/root_ignored.py\n", encoding="utf-8")
    (recipe_dir / ".gitignore").write_text("nested/ignored_nested.py\n", encoding="utf-8")

    monkeypatch.setattr(launch, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(launch, "RECIPES_ROOT", recipes_root)

    imported_modules = []

    def fake_import_module(module_name: str):
        imported_modules.append(module_name)
        return None

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    launch._import_recipe_modules(recipe_dir)

    assert imported_modules == ["recipes.demo.keep", "recipes.demo.nested.keep_nested"]


def test_apply_runtime_patches_is_idempotent(monkeypatch):
    calls = []

    def fake_apply_all_patches():
        calls.append("called")
        return []

    monkeypatch.setattr(launch, "apply_all_patches", fake_apply_all_patches)
    monkeypatch.setattr(launch, "_RUNTIME_PATCHES_APPLIED", False)

    launch._apply_runtime_patches()
    launch._apply_runtime_patches()

    assert calls == ["called"]
