"""Recipe-local assertions for the kit-aware token-normalized-loss skill.

Copy this file to:
recipes/<recipe>/tests/skills/token-normalized-loss/asserts.py
"""

import inspect
import math
import textwrap


def test_engine_structure(engine_class: type) -> None:
    """Verify engine phases use TokenNormedLossKit for accounting."""
    method_names = ("prepare_model", "backward_step", "optimizer_step", "train_post_step")
    source_by_method = {
        name: textwrap.dedent(inspect.getsource(method))
        for name in method_names
        if (method := getattr(engine_class, name, None)) is not None
    }

    prepare_source = source_by_method.get("prepare_model", "")
    backward_source = source_by_method.get("backward_step", "")
    optimizer_source = source_by_method.get("optimizer_step", "")
    post_source = source_by_method.get("train_post_step", "")

    assert "token_loss_kit" in "\n".join(source_by_method.values()), "Engine must own a TokenNormedLossKit instance."
    assert "apply_chunked_token_loss_patch" in prepare_source or 'reduction="none"' in prepare_source, (
        "Model must return unreduced per-token loss."
    )
    assert ".sum()" in backward_source and "accumulate_microbatch" in backward_source, (
        "backward_step must sum unreduced loss and call TokenNormedLossKit.accumulate_microbatch(...)."
    )
    assert "effective_tokens" in backward_source and "total_tokens" in backward_source, (
        "backward_step must pass total/effective token counts."
    )
    assert "reduce_window" in optimizer_source, "optimizer_step must call TokenNormedLossKit.reduce_window()."
    assert "rescale_gradients" in optimizer_source, "optimizer_step must call TokenNormedLossKit.rescale_gradients()."
    assert "clip_grad" in optimizer_source, "Gradient clipping must remain after token rescale."
    assert "tokens/effective" in post_source and "perf/toks_per_sec" in post_source, (
        "train_post_step must log token metrics from global reduced values."
    )

    _assert_order(optimizer_source, "unscale_", "rescale_gradients")
    _assert_order(optimizer_source, "rescale_gradients", "clip_grad")


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
