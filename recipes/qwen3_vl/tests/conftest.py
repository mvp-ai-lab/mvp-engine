"""Pytest configuration for Qwen3-VL recipe-local tests."""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register command-line options consumed by Qwen3-VL smoke tests."""
    parser.addoption(
        "--run-smoke",
        action="store_true",
        default=False,
        help="Run the GPU/NPU Qwen3-VL smoke test.",
    )
    parser.addoption(
        "--world-size",
        type=int,
        default=8,
        help="Distributed world size for smoke tests.",
    )
    parser.addoption(
        "--config-override",
        action="append",
        default=[],
        help="OmegaConf dotlist override for smoke tests.",
    )


@pytest.fixture
def run_smoke(request: pytest.FixtureRequest) -> bool:
    """Return whether the heavyweight smoke test was explicitly requested."""
    return bool(request.config.getoption("--run-smoke"))


@pytest.fixture
def world_size(request: pytest.FixtureRequest) -> int:
    """Return the number of distributed worker processes for smoke tests."""
    return int(request.config.getoption("--world-size"))


@pytest.fixture
def config_overrides(request: pytest.FixtureRequest) -> tuple[str, ...]:
    """Return command-line OmegaConf dotlist overrides for smoke configs."""
    return tuple(request.config.getoption("--config-override"))
