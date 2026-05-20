"""Recipe-local assertions for the token-normalized-loss skill.

Copy this file to:
recipes/<recipe>/tests/skills/token-normalized-loss/asserts.py
"""

import inspect
import math
import textwrap
from pathlib import Path

from mvp_engine.testing.utils import read_recipe_source


def test_file_structure(recipe_root: Path) -> None:
    """Verify recipe code contains token-normalized loss wiring."""
    source = read_recipe_source(recipe_root)

    assert 'reduction="none"' in source or "reduction='none'" in source, (
        "Model loss path must use unreduced per-token loss."
    )
    assert "effective_token" in source, "Recipe must count effective supervised tokens."
    assert "loss_sum" in source, "Recipe must accumulate unreduced loss_sum."
    assert "backward_loss_divisor" in source, "Recipe must store a provisional backward denominator."
    assert "gradient_scale" in source, "Optimizer step must rescale gradients by global token count."
    assert "tokens/effective" in source, "Recipe must log tokens/effective."


def test_engine_structure(engine_class: type) -> None:
    """Verify engine phases contain token-normalized loss accounting."""
    method_names = ("train_pre_step", "forward_step", "backward_step", "optimizer_step", "train_post_step")
    source_by_method = {
        name: textwrap.dedent(inspect.getsource(method))
        for name in method_names
        if (method := getattr(engine_class, name, None)) is not None
    }

    assert "effective_token" in source_by_method.get("train_pre_step", ""), (
        "train_pre_step must compute effective supervised token counts."
    )
    backward_source = source_by_method.get("backward_step", "")
    optimizer_source = source_by_method.get("optimizer_step", "")
    post_source = source_by_method.get("train_post_step", "")

    assert ".sum()" in backward_source and "loss_sum" in backward_source, (
        "backward_step must sum unreduced per-token loss."
    )
    assert "backward_loss_divisor" in backward_source, "backward_step must store the provisional backward_loss_divisor."
    assert "reduce_all" in optimizer_source or "all_reduce" in optimizer_source, (
        "optimizer_step must reduce loss/token metrics before gradient rescale."
    )
    assert "gradient_scale" in optimizer_source and ".grad" in optimizer_source, (
        "optimizer_step must rescale gradients by global effective token count."
    )
    assert "clip_grad" in optimizer_source, "Gradient clipping must remain after token rescale."
    assert "tokens/effective" in post_source and "perf/toks_per_sec" in post_source, (
        "train_post_step must log token metrics from global reduced values."
    )

    _assert_order(optimizer_source, "unscale_", "gradient_scale")
    _assert_order(optimizer_source, "gradient_scale", "clip_grad")


def assert_train_post_step_end(engine, ctx) -> None:
    """Verify one real optimizer step produced valid token-normalized logs."""
    outputs = getattr(ctx, "outputs", None)
    if not getattr(ctx, "optimizer_step_completed", False):
        return

    assert isinstance(outputs, dict), "TrainStepContext outputs must be a dict after train_post_step."
    logs = outputs.get("logs")
    assert isinstance(logs, dict), "TrainStepContext outputs must contain logs."

    for key in ("train/loss", "tokens/total", "tokens/effective", "perf/toks_per_sec"):
        assert key in logs, f"Token-normalized training must log {key}."
        assert isinstance(logs[key], int | float), f"{key} must be numeric."
        assert math.isfinite(float(logs[key])), f"{key} must be finite."

    assert int(logs["tokens/effective"]) > 0, "tokens/effective must be positive for a completed optimizer step."


def _assert_order(source: str, before: str, after: str) -> None:
    before_index = source.find(before)
    after_index = source.find(after)
    assert before_index >= 0, f"optimizer_step must contain {before}."
    assert after_index >= 0, f"optimizer_step must contain {after}."
    assert before_index < after_index, f"{before} must appear before {after}."
