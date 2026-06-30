"""Recipe-local assertions for the context-parallel skill.

Copy this file to:
recipes/<recipe>/tests/skills/context-parallel/asserts.py

Keep this file as the canonical recipe-local CP validation surface. Add
recipe-local knobs and assertions incrementally; do not replace it with
round-local checks or demo-specific expected diffs.
"""

import ast
import json
from pathlib import Path

TOKEN_SEQUENCE_FIELDS = {
    "input_ids",
    "attention_mask",
    "labels",
    "shift_labels",
    "pack_segment_ids",
    "position_ids",
}
MODEL_FAMILY_FIELD_MARKERS = ("pixel", "image", "video", "audio", "vision", "visual", "media")
CP_HELPER_NAMES = {
    "gather_sequence",
    "gather_seq_scatter_hidden",
    "scatter_seq_gather_hidden",
}
DOWNSTREAM_CALL_MARKERS = (
    "attention",
    "embed",
    "forward",
    "language",
    "media",
    "merge",
    "model",
    "placeholder",
    "select",
    "vision",
    "visual",
)
PACKED_TOPOLOGY_MARKERS = ("cu_seq_lens", "cu_seqlens", "max_seqlen")
RUNTIME_HOOK_FUNCTIONS = {
    "assert_before_train_end",
    "assert_train_pre_step_end",
    "assert_forward_step_end",
    "assert_optimizer_step_end",
}
REQUIRED_PARITY_METRICS = (
    "loss_cp_off",
    "loss_cp_on",
    "loss_abs_diff",
    "grad_max_abs_diff",
    "grad_mean_abs_diff",
)

# Adapt these in recipe-local copies when the generic inference is not enough.
MODEL_FILE_GLOBS: tuple[str, ...] = ("model/**/*.py",)
ENGINE_FILE_GLOBS: tuple[str, ...] = ("engine/**/*.py",)
CONFIG_FILE_GLOBS: tuple[str, ...] = ("configs/*",)
MODEL_FAMILY_SEQUENCE_FIELDS: tuple[str, ...] = ()
MODEL_FAMILY_HELPER_NAMES: tuple[str, ...] = ()
MODEL_FAMILY_NATIVE_LOCAL_FORWARD = False
CP_ATTENTION_CLASS_NAMES: tuple[str, ...] = ()
AUXILIARY_HIDDEN_NAMES: tuple[str, ...] = ()
PARITY_ARTIFACT_PATHS: tuple[str, ...] = ()


def test_file_structure(recipe_root: Path) -> None:
    """Verify context-parallel recipe validation and code locations exist."""
    asserts_path = recipe_root / "tests" / "skills" / "context-parallel" / "asserts.py"
    model_files = _files(recipe_root, MODEL_FILE_GLOBS)
    engine_files = _files(recipe_root, ENGINE_FILE_GLOBS)
    config_files = _files(recipe_root, CONFIG_FILE_GLOBS)

    assert asserts_path.exists(), f"Missing recipe-local skill assertions: {asserts_path}"
    assert model_files, "Recipe must expose model files, or MODEL_FILE_GLOBS should be adapted."
    assert engine_files, "Recipe must expose engine files, or ENGINE_FILE_GLOBS should be adapted."
    assert config_files, "Recipe must expose configs, or CONFIG_FILE_GLOBS should be adapted."


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
        assert not backend_kwargs.tp.builtin_sequence_parallel, (
            "TP built-in sequence parallel is not compatible with active context parallel."
        )


def test_engine_structure(engine_class: type) -> None:
    """Verify the engine exposes the methods CP contract tests inspect."""
    assert hasattr(engine_class, "prepare_model"), "Engine class must define prepare_model."
    assert hasattr(engine_class, "optimizer_step"), "Engine class must define optimizer_step."


def test_contract(recipe_root: Path) -> None:
    """Verify cheap semantic CP invariants without running training."""
    model_files = _files(recipe_root, MODEL_FILE_GLOBS)
    engine_files = _files(recipe_root, ENGINE_FILE_GLOBS)
    config_files = _files(recipe_root, CONFIG_FILE_GLOBS)
    test_files = _recipe_validation_files(recipe_root)

    assert_sequence_specs_contract(engine_files)
    assert_model_family_dataflow_contract(engine_files, model_files, test_files)
    assert_cp_helper_outputs_drive_dataflow(model_files + engine_files)
    assert_attention_dispatch_contract(model_files, test_files)
    assert_packed_topology_contract(engine_files + model_files, test_files)
    assert_auxiliary_hidden_contract(model_files, test_files)
    assert_optimizer_order_contract(engine_files)


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


def assert_packed_topology_matches_flattened_qkv(cu_seqlens, qkv_length: int, *, name: str = "cu_seqlens") -> None:
    """Verify packed/varlen attention topology matches the Q/K/V tensor passed to attention."""
    values = _as_int_list(cu_seqlens)
    assert len(values) >= 2, f"{name} must contain at least two entries."
    assert values[0] == 0, f"{name} must start with 0."
    assert all(curr >= prev for prev, curr in zip(values, values[1:])), f"{name} must be monotonically increasing."
    assert values[-1] == int(qkv_length), (
        f"{name}[-1] must match the flattened Q/K/V length consumed by this attention call. "
        "Validate post-gather length when Ulysses or another CP wrapper gathers Q/K/V before attention."
    )


def assert_attention_dispatch_bound(model, *, class_names=None, probe=None) -> None:
    """Verify attention dispatch reaches a CP-compatible path.

    Prefer a runtime probe when the model wraps external attention calls and
    static `CP_MODULE_CONFIG` metadata is not enough to prove executable dispatch.
    """
    if probe is not None:
        assert callable(probe), "Attention dispatch probe must be callable and observe runtime binding."
        result = probe(model)
        assert isinstance(result, dict), "Attention dispatch probe must return structured evidence."
        observed = bool(result.get("cp_dispatch_observed")) or int(result.get("bound_module_count", 0)) > 0
        assert observed, "Attention dispatch runtime probe did not observe CP dispatch or bound modules."
        observed_classes = set(result.get("class_names", ()))
        required = set(class_names or CP_ATTENTION_CLASS_NAMES)
        missing = sorted(required - observed_classes)
        assert not missing, f"Attention dispatch probe did not observe configured classes: {missing}."
        return

    required = set(class_names or CP_ATTENTION_CLASS_NAMES)
    config = getattr(model, "CP_MODULE_CONFIG", None)
    assert isinstance(config, dict) and config, "Model must expose a non-empty CP_MODULE_CONFIG."
    if required:
        missing = sorted(required - set(config))
        assert not missing, f"CP_MODULE_CONFIG is missing attention classes: {missing}."


def assert_auxiliary_hidden_layout(inputs_embeds, auxiliary_tensors: dict[str, object]) -> None:
    """Verify auxiliary hidden tensors entering the LLM share inputs_embeds sequence ownership."""
    expected_seq = int(inputs_embeds.shape[1])
    for name, tensor in auxiliary_tensors.items():
        actual_seq = int(tensor.shape[1])
        assert actual_seq == expected_seq, (
            f"{name} sequence length {actual_seq} must match inputs_embeds sequence length {expected_seq}."
        )


def assert_native_local_forward(model, *, media_fields: tuple[str, ...], probe) -> None:
    """Verify a model advertised as native-local really consumes CP-local media."""
    assert callable(probe), "Native-local forward proof must use a callable runtime probe."
    result = probe(model)
    assert isinstance(result, dict), "Native-local forward probe must return structured evidence."
    assert result.get("native_local_forward") is True, "Native-local forward probe did not pass."
    observed_fields = set(result.get("media_fields", ()))
    missing = sorted(set(media_fields) - observed_fields)
    assert not missing, f"Native-local forward probe did not observe media fields: {missing}."


def assert_cp_parity_artifact(path: str | Path, *, allow_blocked: bool = False) -> None:
    """Verify a CP parity/impact artifact has real metrics or an explicit unresolved blocked status."""
    artifact_path = Path(path)
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    status = data.get("status")
    metrics = data.get("metrics")
    assert status in {"passed", "failed", "blocked"}, "Parity artifact status must be passed, failed, or blocked."
    assert isinstance(metrics, dict), "Parity artifact metrics must be a dict."

    if status == "blocked":
        assert data.get("reason"), "Blocked parity artifacts must include a concrete reason."
        assert allow_blocked, "Blocked hard validation is unresolved, not a correctness pass."
        return

    assert status == "passed", f"Parity artifact did not pass: {status}."
    missing = [name for name in REQUIRED_PARITY_METRICS if name not in metrics]
    assert not missing, f"Parity artifact is missing metrics: {missing}."


def assert_sequence_specs_contract(engine_files: list[Path]) -> None:
    fields = _cp_sequence_spec_fields(engine_files)
    assert "input_ids" in fields, "CPSequenceSpec must include input_ids."
    assert fields & {"labels", "shift_labels"}, "CPSequenceSpec should include labels or shift_labels."


def assert_model_family_dataflow_contract(engine_files: list[Path], model_files: list[Path], test_files: list[Path]) -> None:
    fields = _cp_sequence_spec_fields(engine_files)
    inferred_model_fields = {field for field in fields if _looks_like_model_family_field(field)}
    model_family_fields = inferred_model_fields | set(MODEL_FAMILY_SEQUENCE_FIELDS)
    if not model_family_fields:
        return
    if MODEL_FAMILY_NATIVE_LOCAL_FORWARD:
        has_native_probe = _files_call_executable_assertion(test_files, "assert_native_local_forward")
        has_parity_validation = bool(PARITY_ARTIFACT_PATHS) and (
            _files_call_executable_assertion(test_files, "assert_parity_artifacts_contract")
            or _files_call_executable_assertion(test_files, "assert_cp_parity_artifact")
        )
        assert has_native_probe or has_parity_validation, (
            "MODEL_FAMILY_NATIVE_LOCAL_FORWARD=True requires a runtime proof via assert_native_local_forward(...) "
            "or an executable parity/impact artifact validation."
        )
        return

    model_source = _source_for(model_files)
    assert not _source_discards_cp_kit(model_source), (
        "Model-family fields are CP-sliced, but the model patch discards cp_kit. "
        "Feed CP helper outputs into media, placeholder, attention, or LLM dataflow."
    )
    assert _has_meaningful_cp_helper_dataflow(model_files), (
        "Model-family fields are CP-sliced, but model files do not use CP helper outputs. "
        "Add model-family CP dataflow or set MODEL_FAMILY_NATIVE_LOCAL_FORWARD=True with runtime validation."
    )


def assert_cp_helper_outputs_drive_dataflow(paths: list[Path]) -> None:
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _attach_parents(tree)
        assigned_helpers = _assigned_cp_helpers(tree)
        for target_name, lineno in assigned_helpers:
            loads = _loads_after(tree, target_name, lineno)
            assert any(_is_meaningful_helper_output_use(load) for load in loads), (
                f"{path}: CP helper output {target_name!r} is assigned but not used in downstream dataflow."
            )

        for node in ast.walk(tree):
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call_name = _call_name(node.value.func).rsplit(".", 1)[-1]
            assert call_name not in _cp_helper_names(), (
                f"{path}: CP helper {call_name} is called as a marker. Assign and use its output."
            )


def assert_attention_dispatch_contract(model_files: list[Path], test_files: list[Path]) -> None:
    config_names = _cp_module_config_names(model_files)
    has_dispatch_probe = _files_call_executable_assertion(test_files, "assert_attention_dispatch_bound", "probe")
    assert config_names or has_dispatch_probe, (
        "Model files must expose inspectable CP_MODULE_CONFIG names, or public tests must call "
        "assert_attention_dispatch_bound(...) with a runtime dispatch probe."
    )
    if CP_ATTENTION_CLASS_NAMES:
        missing = sorted(set(CP_ATTENTION_CLASS_NAMES) - config_names)
        assert not missing or has_dispatch_probe, (
            f"CP_MODULE_CONFIG is missing configured attention classes: {missing}. "
            "If dispatch is dynamic, add a recipe-local smoke probe that calls assert_attention_dispatch_bound(...)."
        )


def assert_packed_topology_contract(paths: list[Path], test_files: list[Path]) -> None:
    if not any(_source_uses_name_like(path, PACKED_TOPOLOGY_MARKERS) for path in paths):
        return
    has_runtime_check = _files_call_executable_assertion(test_files, "assert_packed_topology_matches_flattened_qkv")
    assert has_runtime_check, (
        "Recipe builds packed/varlen attention metadata. Add a contract/smoke call to "
        "assert_packed_topology_matches_flattened_qkv(...) using metadata and the actual attention Q/K/V length."
    )


def assert_auxiliary_hidden_contract(model_files: list[Path], test_files: list[Path]) -> None:
    if not AUXILIARY_HIDDEN_NAMES:
        return
    missing = [name for name in AUXILIARY_HIDDEN_NAMES if not _source_defines_or_uses_name_like(model_files, name)]
    assert not missing, f"Configured auxiliary hidden tensors are missing from model source: {missing}."
    has_runtime_check = _files_call_executable_assertion(test_files, "assert_auxiliary_hidden_layout")
    assert has_runtime_check, (
        "Auxiliary hidden tensors require a smoke or contract hook calling "
        "assert_auxiliary_hidden_layout(...) at the LLM boundary."
    )


def assert_optimizer_order_contract(engine_files: list[Path]) -> None:
    checked = False
    for path in engine_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for function in ast.walk(tree):
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) and function.name == "optimizer_step":
                checked = True
                _assert_optimizer_step_function_order(path, function)
    assert checked, "Engine must define optimizer_step so CP gradient sync order can be validated."


def assert_parity_artifacts_contract(recipe_root: Path) -> None:
    for relative_path in PARITY_ARTIFACT_PATHS:
        assert_cp_parity_artifact(recipe_root / relative_path)


def _assert_optimizer_step_function_order(path: Path, function: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    calls = _call_events(function)
    sync_positions = [pos for pos, name in calls if name == "sync_cp_grads"]
    assert sync_positions, f"{path}: optimizer_step must call sync_cp_grads(...)."
    sync_pos = min(sync_positions)
    unscale_positions = [pos for pos, name in calls if name in {"unscale", "unscale_"}]

    for pos, name in calls:
        if name in {"rescale_grads", "reduce_window"}:
            assert pos < sync_pos, f"{path}: token/global gradient rescale must run before sync_cp_grads(...)."
            for unscale_pos in unscale_positions:
                assert unscale_pos < pos, f"{path}: AMP unscale must run before token/global gradient rescale."
        if name in {"clip_grad", "clip_grad_norm", "clip_grad_norm_"}:
            assert sync_pos < pos, f"{path}: sync_cp_grads(...) must run before gradient clipping."
        if name in {"optimizer.step", "step"}:
            assert sync_pos < pos, f"{path}: sync_cp_grads(...) must run before optimizer.step()."
    for unscale_pos in unscale_positions:
        assert unscale_pos < sync_pos, f"{path}: AMP unscale must run before sync_cp_grads(...)."


def _call_events(node: ast.AST) -> list[tuple[tuple[int, int], str]]:
    events: list[tuple[tuple[int, int], str]] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = _call_name(child.func)
        short_name = name.rsplit(".", 1)[-1]
        events.append(((getattr(child, "lineno", 0), getattr(child, "col_offset", 0)), short_name))
        if name.endswith(".step"):
            events.append(((getattr(child, "lineno", 0), getattr(child, "col_offset", 0)), "optimizer.step"))
    return sorted(events)


def _cp_sequence_spec_fields(engine_files: list[Path]) -> set[str]:
    fields: set[str] = set()
    for path in engine_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node.func).rsplit(".", 1)[-1] != "CPSequenceSpec":
                continue
            value = node.args[0] if node.args else None
            for keyword in node.keywords:
                if keyword.arg in {"key", "field"}:
                    value = keyword.value
                    break
            field = _literal_string(value)
            if field is not None:
                fields.add(field)
    return fields


def _cp_module_config_names(model_files: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in model_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        dict_assignments = _dict_assignments(tree)
        for node in ast.walk(tree):
            names.update(_cp_module_update_names(node))
            value = _cp_module_config_value(node, dict_assignments)
            if not isinstance(value, ast.Dict):
                continue
            for key in value.keys:
                literal = _literal_string(key)
                if literal is not None:
                    names.add(literal)
    return names


def _cp_module_config_value(node: ast.AST, dict_assignments: dict[str, ast.Dict]) -> ast.Dict | None:
    value = None
    if isinstance(node, ast.Assign) and any(_is_cp_module_config_target(target) for target in node.targets):
        value = node.value
    elif isinstance(node, ast.AnnAssign) and _is_cp_module_config_target(node.target):
        value = node.value
    if isinstance(value, ast.Dict):
        return value
    if isinstance(value, ast.Name):
        return dict_assignments.get(value.id)
    return None


def _dict_assignments(tree: ast.AST) -> dict[str, ast.Dict]:
    assignments: dict[str, ast.Dict] = {}
    for node in ast.walk(tree):
        value = None
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            value = node.value
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = [node.target]
        if not isinstance(value, ast.Dict):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = value
    return assignments


def _is_cp_module_config_target(node: ast.AST) -> bool:
    if isinstance(node, ast.Name) and node.id == "CP_MODULE_CONFIG":
        return True
    return isinstance(node, ast.Attribute) and node.attr == "CP_MODULE_CONFIG"


def _cp_module_update_names(node: ast.AST) -> set[str]:
    if not isinstance(node, ast.Call):
        return set()
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "update":
        return set()
    owner = node.func.value
    if not isinstance(owner, ast.Name) or "cp_module_config" not in owner.id.lower():
        return set()
    names: set[str] = set()
    for arg in node.args:
        if not isinstance(arg, ast.Dict):
            continue
        for key in arg.keys:
            literal = _literal_string(key)
            if literal is not None:
                names.add(literal)
    return names


def _assigned_cp_helpers(tree: ast.AST) -> list[tuple[str, int]]:
    assignments: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        value = None
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            value = node.value
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = [node.target]
        if not isinstance(value, ast.Call):
            continue
        if _call_name(value.func).rsplit(".", 1)[-1] not in _cp_helper_names():
            continue
        for target in targets:
            assignments.extend((name, node.lineno) for name in _target_names(target))
    return assignments


def _has_meaningful_cp_helper_dataflow(paths: list[Path]) -> bool:
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _attach_parents(tree)
        for target_name, lineno in _assigned_cp_helpers(tree):
            if any(_is_meaningful_helper_output_use(load) for load in _loads_after(tree, target_name, lineno)):
                return True
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node.func).rsplit(".", 1)[-1] not in _cp_helper_names():
                continue
            if _call_is_downstream_argument(node):
                return True
    return False


def _call_is_downstream_argument(node: ast.Call) -> bool:
    parent = getattr(node, "_parent", None)
    while parent is not None:
        if isinstance(parent, ast.Call):
            call_name = _call_name(parent.func).rsplit(".", 1)[-1].lower()
            return any(marker in call_name for marker in DOWNSTREAM_CALL_MARKERS)
        if isinstance(parent, ast.Subscript):
            parent = getattr(parent, "_parent", None)
            continue
        if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr, ast.Assert, ast.Compare)):
            return False
        parent = getattr(parent, "_parent", None)
    return False


def _loads_after(tree: ast.AST, name: str, lineno: int) -> list[ast.Name]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == name and node.lineno > lineno
    ]


def _is_meaningful_helper_output_use(node: ast.Name) -> bool:
    parent = getattr(node, "_parent", None)
    while parent is not None:
        if isinstance(parent, (ast.Assert, ast.Compare, ast.Expr)):
            return False
        if isinstance(parent, ast.Call):
            call_name = _call_name(parent.func).rsplit(".", 1)[-1].lower()
            return any(marker in call_name for marker in DOWNSTREAM_CALL_MARKERS)
        if isinstance(parent, ast.Return):
            return True
        if isinstance(parent, ast.Subscript):
            parent = getattr(parent, "_parent", None)
            continue
        if isinstance(parent, (ast.Assign, ast.AnnAssign)):
            return False
        parent = getattr(parent, "_parent", None)
    return False


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in node.elts:
            names.extend(_target_names(element))
        return names
    return []


def _looks_like_model_family_field(field: str) -> bool:
    lowered = field.lower()
    return any(marker in lowered for marker in MODEL_FAMILY_FIELD_MARKERS) and field not in TOKEN_SEQUENCE_FIELDS


def _source_discards_cp_kit(source: str) -> bool:
    return "del cp_kit" in source or "cp_kit = None" in source


def _source_uses_name_like(path: Path, markers) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and any(marker in node.id for marker in markers):
            return True
        if isinstance(node, ast.Attribute) and any(marker in node.attr for marker in markers):
            return True
    return False


def _files_call_function(paths: list[Path], function_name: str) -> bool:
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node.func).rsplit(".", 1)[-1] == function_name:
                return True
    return False


def _files_call_executable_assertion(paths: list[Path], function_name: str, keyword_name: str | None = None) -> bool:
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _attach_parents(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node.func).rsplit(".", 1)[-1] != function_name:
                continue
            if keyword_name is not None and not any(keyword.arg == keyword_name for keyword in node.keywords):
                continue
            if _call_is_in_executable_validation_function(node):
                return True
    return False


def _call_is_in_executable_validation_function(node: ast.AST) -> bool:
    parent = getattr(node, "_parent", None)
    while parent is not None:
        if isinstance(parent, ast.If) and isinstance(parent.test, ast.Constant) and parent.test.value is False:
            return False
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _function_is_skipped_test(parent):
                return False
            return parent.name.startswith("test_") or parent.name in RUNTIME_HOOK_FUNCTIONS
        parent = getattr(parent, "_parent", None)
    return False


def _function_is_skipped_test(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in function.decorator_list:
        name = _call_name(decorator.func if isinstance(decorator, ast.Call) else decorator)
        if name.endswith(".skip") or name.endswith(".skipif") or name in {"skip", "skipif"}:
            return True
    return False


def _source_defines_or_uses_name_like(paths: list[Path], name: str) -> bool:
    markers = (name,)
    return any(_source_uses_name_like(path, markers) for path in paths)


def _recipe_test_files(recipe_root: Path) -> list[Path]:
    return sorted((recipe_root / "tests").rglob("test_*.py"))


def _recipe_validation_files(recipe_root: Path) -> list[Path]:
    assert_files = sorted((recipe_root / "tests" / "skills" / "context-parallel").glob("*.py"))
    return sorted(set(_recipe_test_files(recipe_root) + assert_files))


def _cp_helper_names() -> set[str]:
    return set(CP_HELPER_NAMES) | set(MODEL_FAMILY_HELPER_NAMES)


def _files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(files))


def _as_int_list(value) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


def _literal_string(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        prefix = _call_name(func.value)
        return f"{prefix}.{func.attr}" if prefix else func.attr
    return ""


def _attach_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent


def _source_for(paths) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)
