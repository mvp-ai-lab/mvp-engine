"""Recipe-local assertions for the context-parallel skill.

Copy this file to:
recipes/<recipe>/tests/skills/context-parallel/asserts.py
"""

import ast
import inspect
import textwrap
from pathlib import Path


def test_file_structure(recipe_root: Path) -> None:
    """Verify context-parallel recipe wiring can be discovered."""
    asserts_path = recipe_root / "tests" / "skills" / "context-parallel" / "asserts.py"
    model_files = sorted((recipe_root / "model").rglob("*.py"))
    engine_files = sorted((recipe_root / "engine").rglob("*.py"))
    config_files = sorted((recipe_root / "configs").glob("*"))
    assert asserts_path.exists(), f"Missing recipe-local skill assertions: {asserts_path}"
    assert model_files, "Recipe must have model/*.py files, or this assertion should be adapted."

    model_source = _source_for(model_files)
    engine_source = _source_for(engine_files)
    config_source = _source_for(path for path in config_files if path.suffix in {".py", ".yaml", ".yml"})

    assert "CP_MODULE_CONFIG" in model_source, "Top-level model class must bind CP_MODULE_CONFIG."
    assert "CPKit" in engine_source, "Engine should use CPKit for context-parallel batch helpers."
    assert "slice_sequence_batch" in engine_source or "QwenVLCPKit" in engine_source, (
        "Engine should use CPKit or a model-family CPKit extension to shard dense token tensors."
    )
    assert "sync_cp_grads" in engine_source, "optimizer_step must sync CP gradients before clipping."
    assert "labels" in engine_source, "Engine must pass CP-shifted labels to local loss."
    assert "cp" in config_source, "Config must expose parallel.backend_kwargs.cp."
    assert "context" in config_source, "Config must expose parallel.mesh.context."


def test_config_structure(config) -> None:
    """Verify mesh and backend config can activate context parallel safely."""
    mesh = config.parallel.mesh
    backend_kwargs = config.parallel.backend_kwargs
    cp_config = backend_kwargs.cp
    assert isinstance(mesh.context, int), "parallel.mesh.context must be an int."
    assert mesh.context >= 1 or mesh.context == -1, "parallel.mesh.context must be >= 1 or -1."
    assert isinstance(backend_kwargs.tp.builtin_sequence_parallel, bool), "tp.builtin_sequence_parallel must be a bool."
    assert cp_config.implementation == "ulysses", "Only cp.implementation='ulysses' is currently supported."
    assert cp_config.attn_implementation in {"sdpa", "flash_attention_2"}, (
        "cp.attn_implementation must be sdpa or flash_attention_2."
    )
    assert isinstance(cp_config.grad_sync, bool), "cp.grad_sync must be a bool."
    assert cp_config.grad_reduce_dtype in {"same", "float32"}, "cp.grad_reduce_dtype must be same or float32."

    if mesh.context > 1 or mesh.context == -1:
        assert mesh.shard != 1, "This repo requires FSDP2 shard > 1 when context mesh is active."
        if backend_kwargs.tp.builtin_sequence_parallel:
            assert mesh.tensor > 1 or mesh.tensor == -1, (
                "CP+TP built-in sequence parallel requires parallel.mesh.tensor > 1 or -1."
            )


def test_engine_structure(engine_class: type) -> None:
    """Verify the engine reaches the shared CP runtime path."""
    prepare_model_source = textwrap.dedent(inspect.getsource(engine_class.prepare_model))
    optimizer_step_source = textwrap.dedent(inspect.getsource(engine_class.optimizer_step))
    tree = ast.parse(prepare_model_source)
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert any(name.endswith("parallelize_model") for name in calls), (
        "prepare_model must call parallelize_model(...) so the runtime can apply context parallel."
    )
    assert "config.parallel.backend_kwargs" in prepare_model_source or "self.config.parallel.backend_kwargs" in (
        prepare_model_source
    ), "prepare_model must pass config.parallel.backend_kwargs into parallelize_model(...)."
    assert "sync_cp_grads" in optimizer_step_source, "optimizer_step must call sync_cp_grads(...) before clipping."


def assert_before_train_end(engine) -> None:
    """After setup, verify a context-parallel smoke run produced expected runtime state."""
    cp_config = engine.config.parallel.backend_kwargs.cp
    if hasattr(engine, "parallel_mesh"):
        context_size = engine.parallel_mesh.cp.world_size
    else:
        context_size = engine.config.parallel.mesh.context
    if context_size <= 1:
        return

    model = engine.model
    if bool(cp_config.grad_sync):
        assert getattr(model, "_cp_grad_sync", None) is not None, "CP grad sync was not attached to the model."


def assert_train_pre_step_end(engine, ctx) -> None:
    """Verify token batches are sharded along sequence over context ranks."""
    if hasattr(engine, "parallel_mesh"):
        context_size = engine.parallel_mesh.cp.world_size
    else:
        context_size = engine.config.parallel.mesh.context
    if context_size <= 1:
        return
    if not isinstance(ctx.data, dict) or "input_ids" not in ctx.data:
        return

    local_seq_len = int(ctx.data["input_ids"].shape[1])
    assert local_seq_len > 0, "Context-local input_ids sequence length must be positive."


def _source_for(paths) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)
