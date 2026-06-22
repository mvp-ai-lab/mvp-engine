"""Recipe-local assertions for the sequence-parallel skill.

Copy this file to:
recipes/<recipe>/tests/skills/sequence-parallel/asserts.py
"""

import ast
import inspect
import textwrap
from pathlib import Path

from mvp_engine.distributed.utils import (
    MESH_DIM_SHARD,
    MESH_DIM_TENSOR,
    get_mesh_dim_size,
)

ALLOWED_TP_MODES = {"col", "row"}
ALLOWED_SP_MODES = {"sequence"}


def test_file_structure(recipe_root: Path) -> None:
    """Verify model-local SP plan wiring can be discovered by the runtime."""
    asserts_path = recipe_root / "tests" / "skills" / "sequence-parallel" / "asserts.py"
    model_files = sorted((recipe_root / "model").rglob("*.py"))
    assert asserts_path.exists(), f"Missing recipe-local skill assertions: {asserts_path}"
    assert model_files, "Recipe must have model/*.py files, or this assertion should be adapted."

    model_source = "\n".join(path.read_text(encoding="utf-8") for path in model_files)
    config_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((recipe_root / "configs").glob("*"))
        if path.suffix in {".py", ".yaml", ".yml"}
    )

    assert "TP_MODULE_CONFIG" in model_source, "Sequence parallel requires an existing TP_MODULE_CONFIG."
    assert "SEQUENCE_PARALLEL" in model_source, (
        "Bind SEQUENCE_PARALLEL_MODULE_CONFIG or SEQUENCE_PARALLEL_SEQUENCE_DIM on the top-level model class."
    )
    assert "sequence_parallel" in config_source, (
        "Config must expose parallel.backend_kwargs.sequence_parallel for SP activation."
    )
    assert "mesh.sequence" not in config_source, (
        "Do not add a parallel.mesh.sequence dimension; SP must reuse parallel.mesh.tensor."
    )
    assert "sequence_parallel_module_config" not in config_source.lower(), (
        "Do not move sequence-parallel module plans into YAML/config."
    )


def test_config_structure(config) -> None:
    """Verify mesh and backend config can activate SP safely."""
    backend_kwargs = config.parallel.backend_kwargs
    assert hasattr(backend_kwargs, "sequence_parallel"), (
        "parallel.backend_kwargs.sequence_parallel must be available in the config schema."
    )
    assert isinstance(backend_kwargs.sequence_parallel, bool), "sequence_parallel must be a bool."
    assert isinstance(config.parallel.mesh.tensor, int), "parallel.mesh.tensor must be an int."
    assert isinstance(config.parallel.mesh.shard, int), "parallel.mesh.shard must be an int."

    if backend_kwargs.sequence_parallel:
        assert config.parallel.mesh.tensor > 1 or config.parallel.mesh.tensor == -1, (
            "sequence_parallel=true requires parallel.mesh.tensor > 1 or -1."
        )
        assert config.parallel.mesh.shard != 1, "This repo requires FSDP2 shard > 1 when sequence_parallel=true."


def test_engine_structure(engine_class: type) -> None:
    """Verify the engine reaches the shared TP/SP runtime path."""
    source = textwrap.dedent(inspect.getsource(engine_class.prepare_model))
    tree = ast.parse(source)
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert any(name.endswith("parallelize_model") for name in calls), (
        "prepare_model must call parallelize_model(...) so the runtime can discover TP/SP plans."
    )
    assert "config.parallel.backend_kwargs" in source or "self.config.parallel.backend_kwargs" in source, (
        "prepare_model must pass config.parallel.backend_kwargs into parallelize_model(...)."
    )


def assert_before_train_end(engine) -> None:
    """After setup, verify an SP-active smoke run produced TP/SP-compatible state."""
    if not engine.config.parallel.backend_kwargs.sequence_parallel:
        return

    if hasattr(engine, "device_mesh"):
        tp_size = get_mesh_dim_size(engine.device_mesh, MESH_DIM_TENSOR)
        shard_size = get_mesh_dim_size(engine.device_mesh, MESH_DIM_SHARD)
    else:
        tp_size = engine.config.parallel.mesh.tensor
        shard_size = engine.config.parallel.mesh.shard

    assert tp_size > 1, "SP-active smoke run must use tensor mesh size > 1."
    assert shard_size != 1, "SP-active smoke run must use FSDP2 shard > 1."

    model = engine.unwrapped_model if hasattr(engine, "unwrapped_model") else engine.model
    tp_config = getattr(model.__class__, "TP_MODULE_CONFIG", None)
    sp_config = getattr(model.__class__, "SEQUENCE_PARALLEL_MODULE_CONFIG", {})
    sequence_dim = getattr(model.__class__, "SEQUENCE_PARALLEL_SEQUENCE_DIM", 1)

    assert isinstance(tp_config, dict) and tp_config, "SP-active smoke run must use a non-empty TP_MODULE_CONFIG."
    assert isinstance(sp_config, dict), "SEQUENCE_PARALLEL_MODULE_CONFIG must be a dict when present."
    assert isinstance(sequence_dim, int), "SEQUENCE_PARALLEL_SEQUENCE_DIM must be an int."

    for module_name, plan in tp_config.items():
        assert isinstance(module_name, str) and module_name, "TP_MODULE_CONFIG keys must be runtime class names."
        assert isinstance(plan, dict) and plan, f"TP plan for {module_name} must be a non-empty dict."
        assert set(plan.values()) <= ALLOWED_TP_MODES, (
            f"TP plan for {module_name} must use only {sorted(ALLOWED_TP_MODES)} unless this assertion is adapted."
        )

    for module_name, plan in sp_config.items():
        assert isinstance(module_name, str) and module_name, "SP config keys must be runtime class names."
        assert isinstance(plan, dict) and plan, f"SP plan for {module_name} must be a non-empty dict."
        assert set(plan.values()) <= ALLOWED_SP_MODES, (
            f"SP plan for {module_name} must use only {sorted(ALLOWED_SP_MODES)} unless this assertion is adapted."
        )

    has_dtensor_param = any(hasattr(param, "to_local") for param in model.parameters())
    assert has_dtensor_param, "SP-active smoke run did not produce any DTensor parameters."
