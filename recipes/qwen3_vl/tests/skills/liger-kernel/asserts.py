"""Recipe-local assertions for the Liger Kernel skill.

Copy this file to:
recipes/<recipe>/tests/skills/liger-kernel/asserts.py
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

ALLOWED_STAGES = {"pre_build", "post_build"}


def test_config_structure(config: Any) -> None:
    """Verify config exposes the standard nested model.liger_kernel shape."""
    liger_config = config.model.liger_kernel

    assert isinstance(liger_config.enabled, bool), "model.liger_kernel.enabled must be a bool."
    assert liger_config.stage in ALLOWED_STAGES, f"model.liger_kernel.stage must be one of {sorted(ALLOWED_STAGES)}."
    assert liger_config.modules == "auto" or isinstance(liger_config.modules, dict), (
        "model.liger_kernel.modules must be 'auto' or a mapping of module names to booleans."
    )
    if isinstance(liger_config.modules, dict):
        for name, enabled in liger_config.modules.items():
            assert isinstance(name, str) and name, "Liger module names must be non-empty strings."
            assert isinstance(enabled, bool), f"model.liger_kernel.modules.{name} must be a bool."


def test_engine_structure(engine_class: type) -> None:
    """Verify prepare_model wires Liger at the correct pre/post build stages."""
    source = textwrap.dedent(inspect.getsource(engine_class.prepare_model))
    tree = ast.parse(source)
    calls = [(node.lineno, ast.unparse(node.func), node) for node in ast.walk(tree) if isinstance(node, ast.Call)]

    build_lines = [
        lineno for lineno, name, _ in calls if name.endswith(".build_model") or name.endswith(".from_pretrained")
    ]
    model_patch_lines = [lineno for lineno, name, _ in calls if name.endswith(".apply_model_patches")]
    token_loss_lines = [
        lineno
        for lineno, name, _ in calls
        if "apply_chunked_token_loss_patch" in name or name.endswith(".apply_token_loss_patch")
    ]
    wrap_lines = [
        lineno
        for lineno, name, _ in calls
        if name.endswith("parallelize_model") or name.endswith("DDP") or name.endswith("FullyShardedDataParallel")
    ]

    assert "liger_kernel" in source, "prepare_model must read model.liger_kernel."
    assert "pre_build" in source, "prepare_model must handle model.liger_kernel.stage == 'pre_build'."
    assert "post_build" in source, "prepare_model must handle model.liger_kernel.stage == 'post_build'."
    assert build_lines, "prepare_model must build the model through the recipe model-loading path."
    assert model_patch_lines, "post-build Liger must be able to run through MLLMModelKit.apply_model_patches(...)."

    pre_build_liger_lines = [
        lineno for lineno, name, _ in calls if "liger" in name.lower() and "pre_build" in name.lower()
    ]
    post_build_liger_text = "patch_liger_kernel_post_build" in source or (
        "liger" in source.lower() and "post_build" in source
    )
    assert pre_build_liger_lines, "prepare_model must call a Liger pre-build helper before model construction."
    assert min(pre_build_liger_lines) < min(build_lines), "Liger pre-build patching must run before model loading."
    assert post_build_liger_text, "prepare_model must route post-build Liger through a recipe model patch."
    assert min(model_patch_lines) > min(build_lines), "Model patches must run after model loading."

    if token_loss_lines:
        assert min(model_patch_lines) < min(token_loss_lines), (
            "Liger post-build patches must run before token-loss patching."
        )
    if wrap_lines:
        assert min(model_patch_lines) < min(wrap_lines), (
            "Liger post-build patches must run before distributed wrapping."
        )
