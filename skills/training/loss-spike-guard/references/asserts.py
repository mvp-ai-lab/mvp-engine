"""Recipe-local assertions for the loss-spike-guard skill.

Copy this file to:
recipes/<recipe>/tests/skills/loss-spike-guard/asserts.py
"""

import ast
import inspect
import textwrap
from pathlib import Path


def test_file_structure(recipe_root: Path) -> None:
    """Verify the recipe contains loss-spike guard implementation wiring."""
    config_files = sorted((recipe_root / "configs").glob("*.py")) + sorted((recipe_root / "configs").glob("*.yaml"))
    engine_files = sorted((recipe_root / "engine").glob("*.py"))
    guard_files = [
        path
        for path in sorted(recipe_root.rglob("*.py"))
        if "tests" not in path.parts and "guards" in path.parts and "loss" in path.stem
    ]
    config_source = "\n".join(path.read_text(encoding="utf-8") for path in config_files)
    guard_source = "\n".join(path.read_text(encoding="utf-8") for path in guard_files)
    engine_source = "\n".join(path.read_text(encoding="utf-8") for path in engine_files)

    assert "loss_spike_skip_multiplier" in config_source, "Config must expose optim.loss_spike_skip_multiplier."
    assert "loss_spike_skip_window_size" in config_source, "Config must expose optim.loss_spike_skip_window_size."
    assert "loss_spike_skip_min_history" in config_source, "Config must expose optim.loss_spike_skip_min_history."
    assert guard_files, "Loss spike guard must live in a recipe-local guard file such as guards/loss.py."
    assert "LossGuard" in guard_source or "PerTokenLossGuard" in guard_source, (
        "Recipe guard file must define LossGuard or PerTokenLossGuard."
    )
    assert "loss_guard" in engine_source, "Engine must store the guard as recipe-local loss_guard state."


def test_config_structure(config) -> None:
    """Verify loss-spike guard config fields are well formed."""
    optim = config.optim
    multiplier = optim.loss_spike_skip_multiplier
    assert multiplier is None or multiplier > 0.0, "optim.loss_spike_skip_multiplier must be null or > 0."
    assert int(optim.loss_spike_skip_window_size) >= 1, "optim.loss_spike_skip_window_size must be >= 1."
    assert int(optim.loss_spike_skip_min_history) >= 1, "optim.loss_spike_skip_min_history must be >= 1."
    assert int(optim.loss_spike_skip_min_history) <= int(optim.loss_spike_skip_window_size), (
        "optim.loss_spike_skip_min_history must be <= optim.loss_spike_skip_window_size."
    )


def test_engine_structure(engine_class: type) -> None:
    """Verify the engine checks the guard before backward."""
    prepare_optimizer = getattr(engine_class, "prepare_optimizer", None)
    backward_step = getattr(engine_class, "backward_step", None)
    assert backward_step is not None, "Engine must define backward_step for loss guard wiring."

    prepare_source = textwrap.dedent(inspect.getsource(prepare_optimizer)) if prepare_optimizer else ""
    backward_source = textwrap.dedent(inspect.getsource(backward_step))
    source = f"{prepare_source}\n{backward_source}"

    assert "loss_guard" in source, "Engine must create and use self.loss_guard."
    assert ".check(" in backward_source, "backward_step must call the loss guard check before backward."

    tree = ast.parse(backward_source)
    check_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "check"
    ]
    backward_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "backward"
    ]
    assert check_lines, "backward_step must call loss_guard.check(...)."
    assert backward_lines, "backward_step must call backward()."
    assert min(check_lines) < min(backward_lines), "loss_guard.check(...) must run before backward()."
    assert "* 0.0" in backward_source or "zero_" in backward_source, (
        "Skipped micro-batches must zero the backward loss or local loss contribution."
    )


def assert_before_train_end(engine) -> None:
    """After setup, verify enabled loss-spike guard config reaches runtime state."""
    multiplier = engine.config.optim.loss_spike_skip_multiplier
    guard = getattr(engine, "loss_guard", None)
    if multiplier is None:
        return

    assert guard is not None, "Loss spike guard is enabled in config, but engine.loss_guard is missing."
    assert getattr(guard, "spike_multiplier", None) == multiplier, (
        "engine.loss_guard.spike_multiplier must match optim.loss_spike_skip_multiplier."
    )
    assert hasattr(guard, "loss_history"), "Loss guard must keep non-spike loss history."
