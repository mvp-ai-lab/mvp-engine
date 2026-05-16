"""Recipe-local assertions for the gradient-checkpointing skill.

Copy this file to:
recipes/<recipe>/tests/skills/gradient-checkpointing/asserts.py
"""

import ast
import inspect
import textwrap
from pathlib import Path

from mvp_engine.testing.utils import read_recipe_source


def test_file_structure(recipe_root: Path) -> None:
    """Verify the recipe contains gradient-checkpointing test artifacts and code."""
    source = read_recipe_source(recipe_root)

    assert "gradient_checkpointing_enable" in source or "_gradient_checkpointing_func" in source, (
        "Recipe code must enable native checkpointing or expose a model-side checkpoint function."
    )


def test_config_structure(config) -> None:
    """Verify config exposes the standard gradient-checkpointing shape."""
    gc_config = config.model.gradient_checkpointing
    assert isinstance(gc_config.enabled, bool), "model.gradient_checkpointing.enabled must be a bool."
    assert isinstance(gc_config.use_reentrant, bool), "model.gradient_checkpointing.use_reentrant must be a bool."


def test_engine_structure(engine_class: type) -> None:
    """Verify prepare_model preserves a build-before-wrap boundary for checkpointing."""
    source = textwrap.dedent(inspect.getsource(engine_class.prepare_model))
    tree = ast.parse(source)
    calls = [(node.lineno, ast.unparse(node.func)) for node in ast.walk(tree) if isinstance(node, ast.Call)]

    gc_lines = [lineno for lineno, name in calls if name.endswith(".gradient_checkpointing_enable")]
    wrap_lines = [
        lineno
        for lineno, name in calls
        if name.endswith("parallelize_model") or name.endswith("DDP") or name.endswith("FullyShardedDataParallel")
    ]
    if gc_lines and wrap_lines:
        assert min(gc_lines) < min(wrap_lines), "Gradient checkpointing must be enabled before distributed wrapping."

    if wrap_lines:
        pre_wrap_calls = [name for lineno, name in calls if lineno < min(wrap_lines)]
        assert pre_wrap_calls, "prepare_model must build and configure the model before distributed wrapping."


def assert_before_train_end(engine) -> None:
    """After model setup, verify enabled config reaches the runtime model."""
    gc_config = engine.config.model.gradient_checkpointing
    if not gc_config.enabled:
        return

    model = engine.model.module if hasattr(engine.model, "module") else engine.model
    modules = list(model.modules()) if hasattr(model, "modules") else [model]
    enabled = any(getattr(module, "gradient_checkpointing", False) for module in modules)
    assert enabled, "Gradient checkpointing config is enabled, but no runtime module has gradient_checkpointing=True."

    state = {"calls": 0}
    for module in modules:
        checkpoint_func = getattr(module, "_gradient_checkpointing_func", None)
        if checkpoint_func is None:
            continue

        def counted_checkpoint(*args, _checkpoint_func=checkpoint_func, **kwargs):
            state["calls"] += 1
            return _checkpoint_func(*args, **kwargs)

        module._gradient_checkpointing_func = counted_checkpoint

    engine._gradient_checkpointing_assert_state = state


def assert_backward_step_end(engine, ctx) -> None:
    """After backward, verify the checkpoint function was used when observable."""
    gc_config = engine.config.model.gradient_checkpointing
    if not gc_config.enabled:
        return

    state = getattr(engine, "_gradient_checkpointing_assert_state", None)
    if state is not None:
        assert state["calls"] > 0, "No checkpoint function call was observed during the training step."
