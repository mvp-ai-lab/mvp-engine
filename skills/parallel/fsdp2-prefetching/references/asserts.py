"""Recipe-local assertions for the fsdp2-prefetching skill.

Copy this file to:
recipes/<recipe>/tests/skills/fsdp2-prefetching/asserts.py
"""

import ast
import inspect
import textwrap
from pathlib import Path


def test_file_structure(recipe_root: Path) -> None:
    """Verify model-local FSDP2 prefetch wiring exists."""
    asserts_path = recipe_root / "tests" / "skills" / "fsdp2-prefetching" / "asserts.py"
    model_files = sorted((recipe_root / "model").rglob("*.py"))
    assert asserts_path.exists(), f"Missing recipe-local skill assertions: {asserts_path}"
    assert model_files, "Recipe must have model/*.py files, or this assertion should be adapted."

    model_source = "\n".join(path.read_text(encoding="utf-8") for path in model_files)
    config_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((recipe_root / "configs").glob("*"))
        if path.suffix in {".py", ".yaml", ".yml"}
    )

    assert "APPLY_FSDP2_CUSTOM_PREFETCHING" in model_source, (
        "Top-level model class must bind APPLY_FSDP2_CUSTOM_PREFETCHING."
    )
    assert "set_modules_to_forward_prefetch" in model_source or "set_modules_to_backward_prefetch" in model_source, (
        "FSDP2 prefetch hook must install forward or backward prefetch edges."
    )
    assert "_fsdp2_prefetching_configured" in model_source, "FSDP2 prefetch hook must be idempotent."
    assert "custom_prefetch" not in config_source and "fsdp2_prefetch" not in config_source, (
        "Do not add YAML/config toggles for FSDP2 custom prefetching."
    )


def test_config_structure(config) -> None:
    """Verify configs expose the standard FSDP2 config shape."""
    assert hasattr(config.parallel, "backend_kwargs"), "config.parallel.backend_kwargs is required."
    assert hasattr(config.parallel.backend_kwargs, "fsdp2"), "config.parallel.backend_kwargs.fsdp2 is required."
    assert hasattr(config.parallel.backend_kwargs.fsdp2, "target_classes"), (
        "config.parallel.backend_kwargs.fsdp2.target_classes must exist, even when _no_split_modules supplies targets."
    )


def test_engine_structure(engine_class: type) -> None:
    """Verify the engine reaches the shared FSDP2 runtime path."""
    source = textwrap.dedent(inspect.getsource(engine_class.prepare_model))
    tree = ast.parse(source)
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert any(name.endswith("parallelize_model") for name in calls), (
        "prepare_model must call parallelize_model(...) so the FSDP2 runtime can discover the prefetch hook."
    )
    assert "config.parallel.backend_kwargs" in source or "self.config.parallel.backend_kwargs" in source, (
        "prepare_model must pass config.parallel.backend_kwargs into parallelize_model(...)."
    )


def assert_before_train_end(engine) -> None:
    """After setup, verify an FSDP2 smoke run applied the custom prefetch hook."""
    if getattr(engine.config.parallel.mesh, "shard", 1) == 1:
        return

    model = engine.unwrapped_model if hasattr(engine, "unwrapped_model") else engine.model
    hook = getattr(model.__class__, "APPLY_FSDP2_CUSTOM_PREFETCHING", None)

    assert callable(hook), "FSDP2-active smoke run must use a model class with a callable prefetch hook."
    assert getattr(model, "_fsdp2_prefetching_configured", False), (
        "FSDP2 prefetch hook did not mark the wrapped model as configured."
    )
