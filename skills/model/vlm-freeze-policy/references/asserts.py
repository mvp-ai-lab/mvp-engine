"""Recipe-local assertions for the kit-aware vlm-freeze-policy skill.

Copy this file to:
recipes/<recipe>/tests/skills/vlm-freeze-policy/asserts.py
"""

import ast
import inspect
import textwrap
from typing import Any

VISION_FREEZE_FIELDS = ("freeze_vit", "freeze_vision", "freeze_vision_encoder")
CONNECTOR_FREEZE_FIELDS = ("freeze_projector", "freeze_connector", "freeze_resampler")
LANGUAGE_FREEZE_FIELDS = ("freeze_llm", "freeze_language_model", "freeze_language")

VISION_NAME_HINTS = ("visual.patch_embed", "visual.blocks", "vision", "vit", "image_encoder")
CONNECTOR_NAME_HINTS = ("merger", "projector", "connector", "resampler", "adapter", "q_former")
LANGUAGE_NAME_HINTS = ("language_model", "llm", "text_model", "lm_head", "embed_tokens")


def test_config_structure(config: Any) -> None:
    """Verify config exposes freeze flags for vision, connector, and language groups."""
    model_config = config.model
    field_names = set(model_config.keys()) if hasattr(model_config, "keys") else set(vars(model_config))

    for group_name, candidates in {
        "vision": VISION_FREEZE_FIELDS,
        "connector": CONNECTOR_FREEZE_FIELDS,
        "language": LANGUAGE_FREEZE_FIELDS,
    }.items():
        matches = field_names & set(candidates)
        assert matches, f"model config must expose a {group_name} freeze flag."
        for field_name in matches:
            value = (
                model_config[field_name] if hasattr(model_config, "__getitem__") else getattr(model_config, field_name)
            )
            assert isinstance(value, bool), f"model.{field_name} must be a bool."


def test_engine_structure(engine_class: type) -> None:
    """Verify freeze policy runs through MLLMModelKit before wrapping and optimizer."""
    prepare_model_source = textwrap.dedent(inspect.getsource(engine_class.prepare_model))
    prepare_optimizer = getattr(engine_class, "prepare_optimizer", None)
    prepare_optimizer_source = textwrap.dedent(inspect.getsource(prepare_optimizer)) if prepare_optimizer else ""

    tree = ast.parse(prepare_model_source)
    calls = [(node.lineno, ast.unparse(node.func)) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    kit_freeze_lines = [lineno for lineno, name in calls if name.endswith(".apply_freeze_policy")]
    fallback_freeze = "requires_grad" in prepare_model_source and (
        "named_parameters" in prepare_model_source or ".parameters()" in prepare_model_source
    )
    freeze_lines = kit_freeze_lines
    wrap_lines = [
        lineno
        for lineno, name in calls
        if name.endswith("parallelize_model") or name.endswith("DDP") or name.endswith("FullyShardedDataParallel")
    ]

    assert freeze_lines or fallback_freeze, (
        "prepare_model must use MLLMModelKit freeze policy or a documented freeze fallback."
    )
    if wrap_lines and freeze_lines:
        assert min(freeze_lines) < min(wrap_lines), "Freeze policy must run before distributed wrapping."
    assert "build_optimizer" in prepare_optimizer_source or "requires_grad" in prepare_optimizer_source, (
        "prepare_optimizer must use OptimKit or otherwise collect only trainable parameters."
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
        enabled = (
            model_config[field_name] if hasattr(model_config, "__getitem__") else getattr(model_config, field_name)
        )
        if not enabled:
            continue

        group_parameters = [
            parameter for name, parameter in named_parameters if any(hint in name.lower() for hint in name_hints)
        ]
        assert group_parameters, f"Could not find parameters for enabled freeze flag model.{field_name}."
        assert all(not parameter.requires_grad for parameter in group_parameters), (
            f"model.{field_name}=true, but at least one matching parameter is still trainable."
        )
