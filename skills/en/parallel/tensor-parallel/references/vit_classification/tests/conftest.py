"""Pytest configuration for ViT recipe tests."""

import sys
from pathlib import Path

# Make the repository root importable for `recipes.*` test imports.
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
