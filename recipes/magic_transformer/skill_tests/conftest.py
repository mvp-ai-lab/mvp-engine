"""Pytest configuration for magic_transformer recipe-local skill tests."""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
recipes_root = repo_root / "recipes"

sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(recipes_root))
