"""Recipe-local hard-validation assertions for the model-flops-utilization skill.

Copy this file to:

    recipes/<recipe>/tests/skills/model-flops-utilization/asserts.py

Adapt the string checks only if the target recipe uses unusual names. These
assertions verify MFU wiring without encoding a recipe-specific FLOPs formula.
"""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
from pathlib import Path
from typing import Any

from mvp_engine.testing.utils import read_recipe_source


def test_file_structure(recipe_root: Path) -> None:
    """Verify the recipe contains MFU implementation wiring outside tests."""
    source = read_recipe_source(recipe_root)

    assert "calculate_model_flops" in source, (
        "MFU implementation must define or inject calculate_model_flops(...) outside recipe tests."
    )
    assert "perf/mfu" in source, "MFU implementation must log the standard metric key 'perf/mfu' outside recipe tests."


def test_config_structure(config: Any) -> None:
    """Verify the config exposes the fields needed to interpret MFU scope."""
    optim = config.get("optim") if isinstance(config, dict) else getattr(config, "optim", None)
    parallel = config.get("parallel") if isinstance(config, dict) else getattr(config, "parallel", None)

    assert optim is not None, "Config must expose optim for peak FLOPs lookup."
    assert getattr(optim, "mixed_precision", None) is not None, "Config must expose optim.mixed_precision."
    assert parallel is not None, "Config must expose parallel for distributed MFU scope."
    assert getattr(parallel, "mesh", None) is not None, "Config must expose parallel.mesh."


def test_engine_structure(engine_class: type) -> None:
    """Verify the engine carries per-step FLOPs into the MFU logging path."""
    method_names = ("forward_step", "backward_step", "optimizer_step", "train_post_step")
    source = "\n".join(
        textwrap.dedent(inspect.getsource(method))
        for name in method_names
        if (method := getattr(engine_class, name, None)) is not None
    )
    tree = ast.parse(source or "pass")
    call_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute | ast.Name)
    }

    assert "calculate_model_flops" in call_names, (
        "Engine must call calculate_model_flops(...) on the real prepared batch."
    )
    assert "model_flops" in source or "flops_per_step" in source, (
        "Engine must carry model FLOPs from micro-batches to optimizer-step logging."
    )
    assert call_names & {"build_mfu_log", "calculate_mfu", "log_metrics"}, (
        "Engine must call an MFU logging helper or log_metrics(...) after computing FLOPs."
    )
    assert "timer" in source or "step_time_seconds" in source or "progress_time_latest" in source, (
        "Engine MFU logging must use real optimizer-step timing."
    )


def assert_train_post_step_end(engine: Any, ctx: Any) -> None:
    """Verify a real training step produced a valid MFU log entry."""
    outputs = getattr(ctx, "outputs", None)
    assert isinstance(outputs, dict), "TrainStepContext outputs must be a dict after train_post_step."
    logs = outputs.get("logs")
    assert isinstance(logs, dict), "TrainStepContext outputs must contain a logs dict."

    assert "perf/mfu" in logs, "MFU smoke validation requires logs['perf/mfu']."
    mfu = logs["perf/mfu"]
    assert isinstance(mfu, float), "logs['perf/mfu'] must be a float."
    assert math.isfinite(mfu), "logs['perf/mfu'] must be finite."
    assert mfu >= 0.0, "logs['perf/mfu'] must be non-negative."

    for key in ("perf/model_flops_per_step", "perf/step_time_seconds", "perf/peak_tflops", "perf/num_training_gpus"):
        if key in logs:
            assert isinstance(logs[key], int | float), f"{key} must be numeric when logged."
            assert math.isfinite(float(logs[key])), f"{key} must be finite when logged."
            assert float(logs[key]) > 0.0, f"{key} must be positive when logged."
