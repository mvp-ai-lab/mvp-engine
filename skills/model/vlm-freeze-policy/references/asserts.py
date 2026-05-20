"""Recipe-local assertions for the vlm-freeze-policy skill.

Copy this file to:
recipes/<recipe>/tests/skills/vlm-freeze-policy/asserts.py
"""

import ast
import inspect
import textwrap
from pathlib import Path

from mvp_engine.testing.utils import read_recipe_source

VISION_FREEZE_FIELDS = ("freeze_vit", "freeze_vision", "freeze_vision_encoder")
CONNECTOR_FREEZE_FIELDS = ("freeze_merger", "freeze_projector", "freeze_connector", "freeze_resampler")
LANGUAGE_FREEZE_FIELDS = ("freeze_llm", "freeze_language_model", "freeze_language")

VISION_NAME_HINTS = ("visual.patch_embed", "visual.blocks", "vision", "vit", "image_encoder")
CONNECTOR_NAME_HINTS = ("merger", "projector", "connector", "resampler", "adapter", "q_former")
LANGUAGE_NAME_HINTS = ("language_model", "llm", "text_model", "lm_head", "embed_tokens")


def test_file_structure(recipe_root: Path) -> None:
    """Verify the recipe contains freeze-policy implementation wiring."""
    source = read_recipe_source(recipe_root)

    assert "requires_grad" in source, "Freeze policy must update or consume parameter.requires_grad."
    assert "named_parameters" in source, "Freeze policy must derive groups from model.named_parameters()."
    assert "freeze_" in source, "Recipe code must expose recipe-local freeze flags."


def test_config_structure(config) -> None:
    """Verify config exposes freeze flags for vision, connector, and language groups."""
    model_config = config.model
    field_names = set(model_config.keys()) if hasattr(model_config, "keys") else set(vars(model_config))

    for group_name, candidates in {
        "vision": VISION_FREEZE_FIELDS,
        "connector": CONNECTOR_FREEZE_FIELDS,
        "language": LANGUAGE_FREEZE_FIELDS,
    }.items():
        matches = field_names & set(candidates)
        assert matches, (
            f"model config must expose a {group_name} freeze flag; adapt candidates for recipe-specific names."
        )
        for field_name in matches:
            if hasattr(model_config, "__getitem__"):
                value = model_config[field_name]
            else:
                value = getattr(model_config, field_name)
            assert isinstance(value, bool), f"model.{field_name} must be a bool."


def test_engine_structure(engine_class: type) -> None:
    """Verify optimizer construction respects requires_grad and model is built before wrapping."""
    prepare_model_source = textwrap.dedent(inspect.getsource(engine_class.prepare_model))
    prepare_optimizer = getattr(engine_class, "prepare_optimizer", None)
    prepare_optimizer_source = textwrap.dedent(inspect.getsource(prepare_optimizer)) if prepare_optimizer else ""

    tree = ast.parse(prepare_model_source)
    calls = [(node.lineno, ast.unparse(node.func)) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    wrap_lines = [
        lineno
        for lineno, name in calls
        if name.endswith("parallelize_model") or name.endswith("DDP") or name.endswith("FullyShardedDataParallel")
    ]
    if wrap_lines:
        pre_wrap_source = "\n".join(
            line for index, line in enumerate(prepare_model_source.splitlines(), start=1) if index < min(wrap_lines)
        )
        assert "freeze" in pre_wrap_source or "build_" in pre_wrap_source, (
            "Freeze policy must run in the model build path before distributed wrapping."
        )

    assert "requires_grad" in prepare_optimizer_source, (
        "prepare_optimizer must collect only trainable parameters or explicitly validate trainability."
    )


def assert_before_train_end(engine) -> None:
    """After setup, verify freeze flags match runtime trainability for common VLM groups."""
    model_config = engine.config.model
    model = engine.model.module if hasattr(engine.model, "module") else engine.model
    named_parameters = list(model.named_parameters())
    trainable = [name for name, parameter in named_parameters if parameter.requires_grad]
    assert trainable, "Freeze policy left the recipe with no trainable parameters."

    for field_names, name_hints in (
        (VISION_FREEZE_FIELDS, VISION_NAME_HINTS),
        (CONNECTOR_FREEZE_FIELDS, CONNECTOR_NAME_HINTS),
        (LANGUAGE_FREEZE_FIELDS, LANGUAGE_NAME_HINTS),
    ):
        field_name = next((name for name in field_names if hasattr(model_config, name)), None)
        if field_name is None:
            continue
        if hasattr(model_config, "__getitem__"):
            enabled = model_config[field_name]
        else:
            enabled = getattr(model_config, field_name)
        if not enabled:
            continue

        group_parameters = [
            parameter for name, parameter in named_parameters if any(hint in name.lower() for hint in name_hints)
        ]
        assert group_parameters, f"Could not find parameters for enabled freeze flag model.{field_name}."
        assert all(not parameter.requires_grad for parameter in group_parameters), (
            f"model.{field_name}=true, but at least one matching parameter is still trainable."
        )
