"""Recipe-local assertions for the context-parallel skill.

Copy this file to:
recipes/<recipe>/tests/skills/context-parallel/asserts.py
"""

import ast
import inspect
import textwrap
from pathlib import Path

from mvp_engine.distributed.utils import get_context_parallel_size

SUPPORTED_RING_IMPL_TYPES = {"zigzag", "basic"}


def test_file_structure(recipe_root: Path) -> None:
    """Verify long-context recipe wiring can be discovered."""
    asserts_path = recipe_root / "tests" / "skills" / "context-parallel" / "asserts.py"
    model_files = sorted((recipe_root / "model").rglob("*.py"))
    engine_files = sorted((recipe_root / "engine").rglob("*.py"))
    config_files = sorted((recipe_root / "configs").glob("*"))
    assert asserts_path.exists(), f"Missing recipe-local skill assertions: {asserts_path}"
    assert model_files, "Recipe must have model/*.py files, or this assertion should be adapted."

    model_source = _source_for(model_files)
    engine_source = _source_for(engine_files)
    config_source = _source_for(path for path in config_files if path.suffix in {".py", ".yaml", ".yml"})

    assert "APPLY_LONG_CONTEXT_ATTENTION" in model_source, (
        "Top-level model class must bind APPLY_LONG_CONTEXT_ATTENTION."
    )
    assert "build_long_context_attention" in model_source, "Attention adapter must use the shared builder."
    assert "CPKit" in engine_source, "Engine should use CPKit for context-parallel batch and loss helpers."
    assert "prepare_causal_batch" in engine_source, "Engine should use CPKit to shard token tensors."
    assert "compute_cross_entropy_loss" in engine_source, "Engine should use CPKit for CP loss semantics."
    assert "labels" in engine_source, "Engine must pass CP-shifted labels to local loss."
    if "get_basic_sequence_offset" in engine_source:
        assert "ring_impl_type" in engine_source and "basic" in engine_source, (
            "get_basic_sequence_offset is valid only behind an explicit basic ring layout branch."
        )
    assert "long_context" in config_source, "Config must expose parallel.backend_kwargs.long_context."
    assert "context" in config_source, "Config must expose parallel.mesh.context."
    assert "ring_impl_type" in config_source, "Config must expose long_context.ring_impl_type."


def test_config_structure(config) -> None:
    """Verify mesh and backend config can activate long-context safely."""
    mesh = config.parallel.mesh
    backend_kwargs = config.parallel.backend_kwargs
    long_context = backend_kwargs.long_context
    assert isinstance(mesh.context, int), "parallel.mesh.context must be an int."
    assert mesh.context >= 1 or mesh.context == -1, "parallel.mesh.context must be >= 1 or -1."
    assert hasattr(backend_kwargs, "sequence_parallel"), "sequence_parallel must remain available."
    assert isinstance(long_context.enabled, bool), "long_context.enabled must be a bool."
    assert int(long_context.ulysses_degree) >= 1, "ulysses_degree must be >= 1."
    assert int(long_context.ring_degree) >= 1, "ring_degree must be >= 1."
    assert str(long_context.ring_impl_type) in SUPPORTED_RING_IMPL_TYPES, (
        f"ring_impl_type must be one of {sorted(SUPPORTED_RING_IMPL_TYPES)}."
    )

    if long_context.enabled:
        assert not backend_kwargs.sequence_parallel, "long_context and sequence_parallel must not both be enabled."
        assert mesh.context > 1 or mesh.context == -1, "long_context requires parallel.mesh.context > 1 or -1."
        assert mesh.shard != 1, "This repo requires FSDP2 shard > 1 when long_context is enabled."
        if str(long_context.ring_impl_type) == "zigzag" and getattr(mesh, "tensor", 1) > 1:
            raise AssertionError("Current shared runtime supports zigzag long-context only with mesh.tensor == 1.")
        if mesh.context > 1:
            assert int(long_context.ulysses_degree) * int(long_context.ring_degree) == int(mesh.context), (
                "ulysses_degree * ring_degree must equal parallel.mesh.context."
            )


def test_engine_structure(engine_class: type) -> None:
    """Verify the engine reaches the shared parallel runtime path."""
    prepare_model_source = textwrap.dedent(inspect.getsource(engine_class.prepare_model))
    train_pre_step_source = textwrap.dedent(inspect.getsource(engine_class.train_pre_step))
    tree = ast.parse(prepare_model_source)
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert any(name.endswith("parallelize_model") for name in calls), (
        "prepare_model must call parallelize_model(...) so the runtime can apply long-context hooks."
    )
    assert "config.parallel.backend_kwargs" in prepare_model_source or "self.config.parallel.backend_kwargs" in (
        prepare_model_source
    ), "prepare_model must pass config.parallel.backend_kwargs into parallelize_model(...)."
    assert "long_context" in train_pre_step_source, "train_pre_step must prepare long-context local batches."


def assert_before_train_end(engine) -> None:
    """After setup, verify a long-context smoke run produced expected runtime state."""
    long_context = engine.config.parallel.backend_kwargs.long_context
    if not long_context.enabled:
        return

    context_size = get_context_parallel_size(engine.device_mesh)
    assert context_size > 1, "Long-context smoke run must use context mesh size > 1."

    model = engine.unwrapped_model if hasattr(engine, "unwrapped_model") else engine.model
    assert getattr(model, "_long_context_attention_configured", False), (
        "Long-context hook did not mark the model as configured."
    )
    if bool(long_context.grad_sync):
        assert getattr(model, "_long_context_grad_sync_configured", False), (
            "Long-context grad sync hooks were not installed."
        )


def assert_train_pre_step_end(engine, ctx) -> None:
    """Verify simple token batches are sharded along sequence over context ranks."""
    long_context = engine.config.parallel.backend_kwargs.long_context
    if not long_context.enabled:
        return
    if not isinstance(ctx.data, dict) or "input_ids" not in ctx.data:
        return
    if not hasattr(engine.config, "data") or not hasattr(engine.config.data, "seq_len"):
        return

    context_size = get_context_parallel_size(engine.device_mesh)
    local_seq_len = int(ctx.data["input_ids"].shape[1])
    global_seq_len = int(getattr(engine, "_long_context_global_seq_len", local_seq_len * context_size))
    assert global_seq_len == local_seq_len * context_size, (
        "input_ids local sequence length must multiply back to the padded global sequence length."
    )
    assert global_seq_len >= int(engine.config.data.seq_len), (
        "Padded global sequence length must cover the configured sequence length."
    )
    if str(long_context.ring_impl_type) == "zigzag":
        layout_multiple = 2 * int(long_context.ring_degree) * int(long_context.ulysses_degree)
        assert global_seq_len % layout_multiple == 0, (
            "Zigzag global sequence length must be padded to 2 * ring_degree * ulysses_degree."
        )
        assert "position_ids" in ctx.data, "Zigzag batches must carry extracted global position_ids."


def _source_for(paths) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)
