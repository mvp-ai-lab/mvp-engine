"""Qwen3-VL model helpers for the minimal VLM recipe."""

from __future__ import annotations

from typing import Any

from transformers import AutoModelForImageTextToText

PROJECTOR_PREFIXES = (
    "model.visual.merger.",
    "model.visual.deepstack_merger_list.",
)


def _is_projector_parameter(name: str) -> bool:
    """Check whether a parameter belongs to the Qwen3-VL projector/merger stack.

    Args:
        name: Fully qualified parameter name from ``model.named_parameters()``.

    Returns:
        ``True`` when the parameter belongs to the visual projector/merger path.
    """
    return any(name.startswith(prefix) for prefix in PROJECTOR_PREFIXES)


def apply_freeze_policy(
    model,
    *,
    freeze_vit: bool = False,
    freeze_projector: bool = False,
    freeze_llm: bool = False,
) -> int:
    """Apply the recipe freeze policy to Qwen3-VL parameter groups.

    Args:
        model: Loaded Qwen3-VL model instance.
        freeze_vit: Whether to freeze the visual encoder stack.
        freeze_projector: Whether to freeze the visual projector and merger stack.
        freeze_llm: Whether to freeze the language model and ``lm_head``.

    Returns:
        The number of parameters that were marked non-trainable.
    """
    frozen_parameters = 0
    for name, parameter in model.named_parameters():
        should_freeze = False
        if freeze_vit and name.startswith("model.visual.") and not _is_projector_parameter(name):
            should_freeze = True
        if freeze_projector and _is_projector_parameter(name):
            should_freeze = True
        if freeze_llm and (name.startswith("model.language_model.") or name.startswith("lm_head.")):
            should_freeze = True

        if should_freeze:
            parameter.requires_grad = False
            frozen_parameters += parameter.numel()

    return frozen_parameters


def build_qwen3_vl_model(model_config: Any):
    """Load the Qwen3-VL model checkpoint and apply the freeze policy.

    Args:
        model_config: Recipe model config with load and freeze settings.

    Returns:
        The initialized Qwen3-VL model.
    """
    freeze_vit = bool(getattr(model_config, "freeze_vit", True))
    freeze_projector = bool(getattr(model_config, "freeze_projector", True))
    freeze_llm = bool(getattr(model_config, "freeze_llm", False))

    model = AutoModelForImageTextToText.from_pretrained(
        model_config.pretrained_model_name_or_path,
        trust_remote_code=bool(getattr(model_config, "trust_remote_code", True)),
        torch_dtype="auto",
    )
    apply_freeze_policy(
        model,
        freeze_vit=freeze_vit,
        freeze_projector=freeze_projector,
        freeze_llm=freeze_llm,
    )
    return model
