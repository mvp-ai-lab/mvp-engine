"""Recipe-local assertions for the tensor-parallel skill.

Copy this file to:
recipes/<recipe>/tests/skills/tensor-parallel/asserts.py
"""

import ast
import inspect
import textwrap
from pathlib import Path

ALLOWED_TP_MODES = {"col", "row"}


def test_file_structure(recipe_root: Path) -> None:
    """Verify model-local TP plan wiring exists."""
    asserts_path = recipe_root / "tests" / "skills" / "tensor-parallel" / "asserts.py"
    model_files = sorted((recipe_root / "model").rglob("*.py"))
    assert asserts_path.exists(), f"Missing recipe-local skill assertions: {asserts_path}"
    assert model_files, "Recipe must have model/*.py files, or this assertion should be adapted."

    model_source = "\n".join(path.read_text(encoding="utf-8") for path in model_files)
    config_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((recipe_root / "configs").glob("*"))
        if path.suffix in {".py", ".yaml", ".yml"}
    )

    assert "TP_MODULE_CONFIG" in model_source, "Top-level model class must bind TP_MODULE_CONFIG."
    assert '"col"' in model_source or "'col'" in model_source, "TP plan must include at least one col-sharded linear."
    assert '"row"' in model_source or "'row'" in model_source, "TP plan must include at least one row-sharded linear."
    assert "tensor:" in config_source or "tensor =" in config_source, "Config must expose parallel.mesh.tensor."
    assert "tp_module_config" not in config_source.lower(), "Do not move TP module plans into YAML/config."


def test_config_structure(config) -> None:
    """Verify mesh config can activate TP safely."""
    assert isinstance(config.parallel.mesh.tensor, int), "parallel.mesh.tensor must be an int."
    assert config.parallel.mesh.tensor >= 1 or config.parallel.mesh.tensor == -1, (
        "parallel.mesh.tensor must be >= 1 or -1."
    )
    assert isinstance(config.parallel.mesh.shard, int), "parallel.mesh.shard must be an int."
    assert config.parallel.mesh.shard >= 1 or config.parallel.mesh.shard == -1, (
        "parallel.mesh.shard must be >= 1 or -1."
    )

    if config.parallel.mesh.tensor > 1 or config.parallel.mesh.tensor == -1:
        assert config.parallel.mesh.shard != 1, "This repo requires FSDP2 shard > 1 when tensor > 1."


def test_engine_structure(engine_class: type) -> None:
    """Verify the engine reaches the shared TP runtime path."""
    source = textwrap.dedent(inspect.getsource(engine_class.prepare_model))
    tree = ast.parse(source)
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert any(name.endswith("parallelize_model") for name in calls), (
        "prepare_model must call parallelize_model(...) so the TP runtime can discover TP_MODULE_CONFIG."
    )
    assert "config.parallel.backend_kwargs" in source or "self.config.parallel.backend_kwargs" in source, (
        "prepare_model must pass config.parallel.backend_kwargs into parallelize_model(...)."
    )


def assert_before_train_end(engine) -> None:
    """After setup, verify a TP-active smoke run produced DTensor parameters."""
    if hasattr(engine, "device_mesh"):
        tp_size = engine.device_mesh["tensor"].size()
    else:
        tp_size = engine.config.parallel.mesh.tensor
    if tp_size == 1:
        return

    model = engine.unwrapped_model if hasattr(engine, "unwrapped_model") else engine.model
    tp_config = getattr(model.__class__, "TP_MODULE_CONFIG", None)

    assert isinstance(tp_config, dict) and tp_config, "TP-active smoke run must use a non-empty TP_MODULE_CONFIG."
    for module_name, plan in tp_config.items():
        assert isinstance(module_name, str) and module_name, "TP_MODULE_CONFIG keys must be runtime class names."
        assert isinstance(plan, dict) and plan, f"TP plan for {module_name} must be a non-empty dict."
        assert set(plan.values()) <= ALLOWED_TP_MODES, (
            f"TP plan for {module_name} must use only {sorted(ALLOWED_TP_MODES)} unless this assertion is adapted."
        )

    has_dtensor_param = any(hasattr(param, "to_local") for param in model.parameters())
    assert has_dtensor_param, "TP-active smoke run did not produce any DTensor parameters."
