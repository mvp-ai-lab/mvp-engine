"""Recipe-local assertions for the model-compile skill.

Copy this file to:
recipes/<recipe>/tests/skills/model-compile/asserts.py
"""

import ast
import inspect
import textwrap
from pathlib import Path

from mvp_engine.testing.utils import read_recipe_source

ALLOWED_COMPILE_BACKENDS = {"inductor", "aot_eager", "eager"}
ALLOWED_COMPILE_MODES = {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}


def test_file_structure(recipe_root: Path) -> None:
    """Verify the recipe contains compile config and implementation wiring."""
    source = read_recipe_source(recipe_root)

    assert ".compile(" in source or "torch.compile" in source, (
        "Recipe code must call model.compile(...), submodule.compile(...), or torch.compile(...)."
    )
    assert "compile_backend" in source, "Recipe code must expose or read model.compile_backend."
    assert "compile_mode" in source, "Recipe code must expose or read model.compile_mode."


def test_config_structure(config) -> None:
    """Verify config exposes the standard model.compile shape."""
    assert isinstance(config.model.compile, bool), "model.compile must be a bool."
    assert config.model.compile_backend in ALLOWED_COMPILE_BACKENDS, (
        f"model.compile_backend must be one of {sorted(ALLOWED_COMPILE_BACKENDS)}. "
        "Adapt this assertion only for a documented custom backend."
    )
    assert config.model.compile_mode in ALLOWED_COMPILE_MODES, (
        f"model.compile_mode must be one of {sorted(ALLOWED_COMPILE_MODES)}."
    )


def test_engine_structure(engine_class: type) -> None:
    """Verify prepare_model gates compile by config and places it before wrapping."""
    source = textwrap.dedent(inspect.getsource(engine_class.prepare_model))
    tree = ast.parse(source)
    calls = [(node.lineno, ast.unparse(node.func), node) for node in ast.walk(tree) if isinstance(node, ast.Call)]

    compile_calls = [
        (lineno, node) for lineno, name, node in calls if name == "torch.compile" or name.endswith(".compile")
    ]
    wrap_lines = [
        lineno
        for lineno, name, _ in calls
        if name.endswith("parallelize_model") or name.endswith("DDP") or name.endswith("FullyShardedDataParallel")
    ]

    assert compile_calls, "prepare_model must call model.compile(...), submodule.compile(...), or torch.compile(...)."
    assert "self.config.model.compile" in source, "compile must be gated by self.config.model.compile."
    assert "self.config.model.compile_backend" in source, "compile must use self.config.model.compile_backend."
    assert "self.config.model.compile_mode" in source, "compile must use self.config.model.compile_mode."

    if wrap_lines:
        assert min(lineno for lineno, _ in compile_calls) < min(wrap_lines), (
            "Compile must happen before distributed wrapping unless this recipe adapts the assertion."
        )


def assert_before_train_end(engine) -> None:
    """After setup, verify compile-enabled smoke tests produced a compiled marker."""
    if not engine.config.model.compile:
        return

    model = engine.model.module if hasattr(engine.model, "module") else engine.model
    modules = list(model.modules()) if hasattr(model, "modules") else [model]
    compiled = any(
        getattr(module, "_compiled_call_impl", None) is not None
        or hasattr(module, "_orig_mod")
        or module.__class__.__name__ in {"OptimizedModule", "CompiledFunction"}
        for module in modules
    )
    assert compiled, "model.compile is enabled, but no compiled module marker was found after model setup."
