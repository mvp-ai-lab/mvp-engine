"""Pytest configuration for ViT recipe tests."""

import sys
from pathlib import Path


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find the repository root from the archived TP reference test path.")


# Make the archived reference package and repository root importable.
reference_root = Path(__file__).resolve().parents[1]
project_root = _find_project_root()
sys.path.insert(0, str(reference_root))
sys.path.insert(0, str(project_root))
