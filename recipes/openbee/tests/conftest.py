"""Pytest configuration for OpenBee recipe-local tests."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
