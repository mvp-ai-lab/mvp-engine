"""Recipe-local assertions for the new-recipe-template skill.

Copy this file to:
recipes/<recipe>/tests/skills/new-recipe-template/asserts.py
"""

import inspect
from pathlib import Path
from typing import Any


def test_file_structure(recipe_root: Path) -> None:
    """Verify the scaffold follows the repo's baseline recipe layout."""
    expected = [
        "README.md",
        "__init__.py",
        "configs/__init__.py",
        "configs/schema.py",
        "dataset/__init__.py",
        "engine/__init__.py",
        "model/__init__.py",
        "tests/test_structure.py",
        "tests/test_smoke.py",
    ]
    for relative_path in expected:
        assert (recipe_root / relative_path).exists(), f"Missing scaffold file: {relative_path}"

    engine_files = sorted((recipe_root / "engine").glob("*_engine.py"))
    assert engine_files, "Scaffold must include engine/<recipe>_engine.py."


def test_config_structure(config: Any) -> None:
    """Verify scaffold config keeps required base sections."""
    for section in ("project", "runtime", "log", "parallel", "optim", "loop", "checkpoint"):
        assert hasattr(config, section), f"Config missing BaseEngineConfig section: {section}."


def test_engine_structure(engine_class: type) -> None:
    """Verify scaffold engine is explicit about unimplemented task behavior."""
    source = inspect.getsource(engine_class)
    assert "NotImplementedError" in source, (
        "New recipe scaffold should keep task-specific behavior explicit until implemented."
    )
