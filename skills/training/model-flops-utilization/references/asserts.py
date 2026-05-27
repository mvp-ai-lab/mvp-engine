"""Recipe-local assertions for the kit-aware model-flops-utilization skill.

Copy this file to:
recipes/<recipe>/tests/skills/model-flops-utilization/asserts.py
"""

import ast
import inspect
import math
import textwrap
from typing import Any


def test_config_structure(config: Any) -> None:
    """Verify the config exposes fields needed to interpret MFU scope."""
    optim = config.get("optim") if isinstance(config, dict) else getattr(config, "optim", None)
    parallel = config.get("parallel") if isinstance(config, dict) else getattr(config, "parallel", None)

    assert optim is not None, "Config must expose optim for precision lookup."
    assert getattr(optim, "mixed_precision", None) is not None, "Config must expose optim.mixed_precision."
    assert parallel is not None, "Config must expose parallel for distributed MFU scope."
    assert getattr(parallel, "mesh", None) is not None, "Config must expose parallel.mesh."


def test_engine_structure(engine_class: type) -> None:
    """Verify the engine routes MFU through MFUKit."""
    source = "\n".join(
        textwrap.dedent(inspect.getsource(method))
        for name in ("forward_step", "train_post_step")
        if (method := getattr(engine_class, name, None)) is not None
    )
    tree = ast.parse(source or "pass")
    call_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute | ast.Name)
    }

    assert "accumulate_microbatch" in call_names, "forward_step must call MFUKit.accumulate_microbatch(...)."
    assert "build_log" in call_names, "train_post_step must call MFUKit.build_log(...)."
    assert "calculate_model_flops" in source or "model=" in source, (
        "MFUKit must receive model FLOPs or a model that can calculate them."
    )
    assert "step_time_seconds" in source or "progress_time_latest" in source, (
        "MFU logging must use real optimizer-step timing."
    )


def assert_train_post_step_end(engine: Any, ctx: Any) -> None:
    """Verify a real training step produced a valid MFU log entry when peak is known."""
    outputs = getattr(ctx, "outputs", None)
    assert isinstance(outputs, dict), "TrainStepContext outputs must be a dict after train_post_step."
    logs = outputs.get("logs")
    assert isinstance(logs, dict), "TrainStepContext outputs must contain a logs dict."

    if "perf/mfu" not in logs:
        return
    mfu = logs["perf/mfu"]
    assert isinstance(mfu, float), "logs['perf/mfu'] must be a float."
    assert math.isfinite(mfu), "logs['perf/mfu'] must be finite."
    assert mfu >= 0.0, "logs['perf/mfu'] must be non-negative."
