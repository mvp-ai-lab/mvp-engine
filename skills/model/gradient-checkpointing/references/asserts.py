"""Recipe-local assertions for the kit-aware gradient-checkpointing skill.

Copy this file to:
recipes/<recipe>/tests/skills/gradient-checkpointing/asserts.py
"""

import ast
import inspect
import textwrap
from typing import Any


def test_config_structure(config: Any) -> None:
    """Verify config exposes the standard gradient-checkpointing shape."""
    gc_config = config.model.gradient_checkpointing
    assert isinstance(gc_config.enabled, bool), "model.gradient_checkpointing.enabled must be a bool."
    assert isinstance(gc_config.use_reentrant, bool), "model.gradient_checkpointing.use_reentrant must be a bool."


def test_engine_structure(engine_class: type) -> None:
    """Verify prepare_model routes checkpointing through MLLMModelKit before wrapping."""
    source = textwrap.dedent(inspect.getsource(engine_class.prepare_model))
    tree = ast.parse(source)
    calls = [(node.lineno, ast.unparse(node.func)) for node in ast.walk(tree) if isinstance(node, ast.Call)]

    kit_gc_lines = [lineno for lineno, name in calls if name.endswith(".apply_gradient_checkpointing")]
    native_gc_lines = [lineno for lineno, name in calls if name.endswith(".gradient_checkpointing_enable")]
    gc_lines = kit_gc_lines or native_gc_lines
    wrap_lines = [
        lineno
        for lineno, name in calls
        if name.endswith("parallelize_model") or name.endswith("DDP") or name.endswith("FullyShardedDataParallel")
    ]

    assert gc_lines or "_gradient_checkpointing_func" in source, (
        "prepare_model must use MLLMModelKit checkpointing or a documented checkpointing fallback."
    )
    assert "self.config.model.gradient_checkpointing.enabled" in source
    assert "self.config.model.gradient_checkpointing.use_reentrant" in source
    if wrap_lines:
        assert min(gc_lines) < min(wrap_lines), "Gradient checkpointing must run before distributed wrapping."


def assert_before_train_end(engine) -> None:
    """After setup, verify enabled config reaches the runtime model."""
    if not engine.config.model.gradient_checkpointing.enabled:
        return

    model = engine.model.module if hasattr(engine.model, "module") else engine.model
    modules = list(model.modules()) if hasattr(model, "modules") else [model]
    enabled = any(getattr(module, "gradient_checkpointing", False) for module in modules)
    assert enabled, "Gradient checkpointing config is enabled, but no runtime module has gradient_checkpointing=True."
