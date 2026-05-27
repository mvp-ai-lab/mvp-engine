"""Recipe-local assertions for the kit-aware model-compile skill.

Copy this file to:
recipes/<recipe>/tests/skills/model-compile/asserts.py
"""

import ast
import inspect
import textwrap
from typing import Any

from mvp_engine.kit.mllm import MLLMModelKit

ALLOWED_COMPILE_BACKENDS = {"inductor", "aot_eager", "eager"}
ALLOWED_COMPILE_MODES = {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}


def test_config_structure(config: Any) -> None:
    """Verify config exposes the standard nested model.compile shape."""
    compile_config = config.model.compile
    assert isinstance(compile_config.enabled, bool), "model.compile.enabled must be a bool."
    assert compile_config.backend in ALLOWED_COMPILE_BACKENDS, (
        f"model.compile.backend must be one of {sorted(ALLOWED_COMPILE_BACKENDS)}."
    )
    assert compile_config.mode in ALLOWED_COMPILE_MODES, (
        f"model.compile.mode must be one of {sorted(ALLOWED_COMPILE_MODES)}."
    )


def test_engine_structure(engine_class: type) -> None:
    """Verify prepare_model routes compile through MLLMModelKit before wrapping."""
    source = textwrap.dedent(inspect.getsource(engine_class.prepare_model))
    tree = ast.parse(source)
    calls = [(node.lineno, ast.unparse(node.func), node) for node in ast.walk(tree) if isinstance(node, ast.Call)]

    kit_compile_lines = [lineno for lineno, name, _ in calls if name.endswith(".apply_model_compile")]
    direct_compile_lines = [lineno for lineno, name, _ in calls if name == "torch.compile" or name.endswith(".compile")]
    compile_lines = kit_compile_lines or direct_compile_lines
    wrap_lines = [
        lineno
        for lineno, name, _ in calls
        if name.endswith("parallelize_model") or name.endswith("DDP") or name.endswith("FullyShardedDataParallel")
    ]

    assert compile_lines, "prepare_model must call MLLMModelKit compile or a documented compile fallback."
    assert "self.config.model.compile.enabled" in source, "compile must be gated by model.compile.enabled."
    assert "self.config.model.compile.backend" in source, "compile must use model.compile.backend."
    assert "self.config.model.compile.mode" in source, "compile must use model.compile.mode."
    if wrap_lines:
        assert min(compile_lines) < min(wrap_lines), "Compile must happen before distributed wrapping."

    if kit_compile_lines:
        kit_source = textwrap.dedent(inspect.getsource(MLLMModelKit.apply_model_compile))
        assert ".compile(" in kit_source, "MLLMModelKit.apply_model_compile must call model.compile(...)."


def assert_before_train_end(engine) -> None:
    """After setup, verify compile-enabled smoke tests produced a compiled marker."""
    if not engine.config.model.compile.enabled:
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
