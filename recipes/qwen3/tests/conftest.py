"""Pytest configuration for Qwen3 pretrain stage-local tests.

The options here are intentionally recipe-local. They let developers and agents
run the same pytest entrypoints while selecting a config file, distributed world
size, and smoke-test config overrides without adding test-only branches to the
training code.
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register command-line options consumed by Qwen3 pretrain tests.

    ``--config-override`` may be passed multiple times. Each value uses
    OmegaConf dotlist syntax, for example ``parallel.mesh.tensor=2`` or
    ``model.compile.enabled=false``.
    """
    parser.addoption(
        "--world-size",
        type=int,
        default=1,
        help="Distributed world size for smoke tests.",
    )
    parser.addoption(
        "--config-name",
        default="train",
        help="Config name for smoke tests, without the .yaml suffix.",
    )
    parser.addoption(
        "--config-override",
        action="append",
        default=[],
        help="OmegaConf dotlist override for smoke tests, for example parallel.mesh.tensor=2.",
    )


@pytest.fixture
def world_size(request: pytest.FixtureRequest) -> int:
    """Return the number of distributed worker processes for smoke tests."""
    return request.config.getoption("--world-size")


@pytest.fixture
def config_name(request: pytest.FixtureRequest) -> str:
    """Return the recipe config name to load, without the ``.yaml`` suffix."""
    return request.config.getoption("--config-name")


@pytest.fixture
def config_overrides(request: pytest.FixtureRequest) -> tuple[str, ...]:
    """Return command-line OmegaConf dotlist overrides for smoke configs."""
    return tuple(request.config.getoption("--config-override"))
