"""Recipe-local assertions for the liger-kernel skill.

Copy this file to:
recipes/<recipe>/tests/skills/liger-kernel/asserts.py
"""

import inspect
import textwrap
from pathlib import Path


def test_file_structure(recipe_root: Path) -> None:
    """Verify the recipe contains Liger kernel wiring."""
    config_files = sorted((recipe_root / "configs").glob("*.py")) + sorted((recipe_root / "configs").glob("*.yaml"))
    engine_files = sorted((recipe_root / "engine").glob("*.py"))
    config_source = "\n".join(path.read_text(encoding="utf-8") for path in config_files)
    engine_source = "\n".join(path.read_text(encoding="utf-8") for path in engine_files)

    assert "liger_kernel" in config_source, "Config must expose model.liger_kernel."
    assert "LigerKernelKit" in engine_source, "Engine must use LigerKernelKit."


def test_config_structure(config) -> None:
    """Verify liger config fields are well formed."""
    liger = getattr(config.model, "liger_kernel", None)
    assert liger is not None, "Config must define model.liger_kernel."
    modules = getattr(liger, "modules", "auto")
    assert modules == "auto" or hasattr(modules, "items"), "model.liger_kernel.modules must be 'auto' or a mapping."


def test_engine_structure(engine_class: type) -> None:
    """Verify the engine applies Liger before building the model."""
    prepare_model = getattr(engine_class, "prepare_model", None)
    assert prepare_model is not None, "Engine must define prepare_model."
    source = textwrap.dedent(inspect.getsource(prepare_model))

    apply_index = source.find("liger_kit.apply(")
    build_index = source.find("build_model(")
    assert apply_index != -1, "prepare_model must call LigerKernelKit.apply(...)."
    assert build_index != -1, "prepare_model must build the model."
    assert apply_index < build_index, "LigerKernelKit.apply(...) must run before the model is built."


def assert_before_train_end(engine) -> None:
    """After setup, verify enabled Liger config reaches the built model.

    Module-class scanning covers norm/MLP swaps (present in every official family
    and typical custom maps); function-only swaps such as rope are validated by
    the smoke loss comparison instead.
    """
    liger = getattr(engine.config.model, "liger_kernel", None)
    if liger is None or not liger.enabled:
        return
    liger_modules = [
        name for name, module in engine.model.named_modules() if module.__class__.__module__.startswith("liger_kernel")
    ]
    assert liger_modules, "Liger is enabled but no liger_kernel modules are present in the built model."
