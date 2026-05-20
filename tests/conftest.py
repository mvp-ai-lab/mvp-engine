"""Pytest configuration for mvp_engine tests."""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
recipes_root = project_root / "recipes"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(recipes_root))
