"""Qwen3-VL model helpers for the OpenBee recipe."""

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


def apply_default_freeze_policy(model) -> int:
    """Freeze the visual encoder and projector for the default demo setup.

    Args:
        model: Loaded Qwen3-VL model instance.

    Returns:
        The number of parameters that were marked non-trainable.
    """
    frozen_parameters = 0
    for name, parameter in model.named_parameters():
        if name.startswith("model.visual.") or _is_projector_parameter(name):
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
    model = AutoModelForImageTextToText.from_pretrained(
        model_config.pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    apply_default_freeze_policy(model)
    return model
