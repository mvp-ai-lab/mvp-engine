"""Qwen3-VL model helpers for the OpenBee recipe."""

from __future__ import annotations

from typing import Any

from transformers import AutoModelForImageTextToText

# ---------------------------------------------------------------------------
# Parameter-name prefixes for each logical sub-module
# ---------------------------------------------------------------------------

# Visual encoder (ViT): patch embedding + transformer blocks
VIT_PREFIXES = (
    "model.visual.patch_embed.",
    "model.visual.blocks.",
)

# Projector / merger stack that maps visual tokens into the LLM embedding space
MERGER_PREFIXES = (
    "model.visual.merger.",
    "model.visual.deepstack_merger_list.",
)

# Language model backbone and output head
LLM_PREFIXES = (
    "model.language_model.",
    "lm_head.",
)


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(p) for p in prefixes)


def apply_freeze_policy(
    model,
    *,
    freeze_vit: bool = True,
    freeze_merger: bool = False,
    freeze_llm: bool = False,
) -> dict[str, int]:
    """Freeze sub-modules of a Qwen3-VL model according to the given flags.

    Args:
        model: Loaded Qwen3-VL model instance.
        freeze_vit: When ``True``, freeze the visual encoder (ViT blocks +
            patch embedding).
        freeze_merger: When ``True``, freeze the projector / merger modules
            (``model.visual.merger`` and ``model.visual.deepstack_merger_list``).
        freeze_llm: When ``True``, freeze the language model backbone and the
            LM head (``model.language_model`` and ``lm_head``).

    Returns:
        A dict mapping sub-module name to the number of frozen parameters.
    """
    frozen_counts: dict[str, int] = {"vit": 0, "merger": 0, "llm": 0}

    for name, parameter in model.named_parameters():
        if freeze_vit and _matches(name, VIT_PREFIXES):
            parameter.requires_grad = False
            frozen_counts["vit"] += parameter.numel()
        elif freeze_merger and _matches(name, MERGER_PREFIXES):
            parameter.requires_grad = False
            frozen_counts["merger"] += parameter.numel()
        elif freeze_llm and _matches(name, LLM_PREFIXES):
            parameter.requires_grad = False
            frozen_counts["llm"] += parameter.numel()

    return frozen_counts


def build_qwen3_vl_model(model_config: Any):
    """Load the Qwen3-VL model checkpoint and apply the configured freeze policy.

    Args:
        model_config: Recipe model config (``OpenbeeModelConfig``) with load
            and freeze settings.

    Returns:
        The initialized Qwen3-VL model.
    """
    model = AutoModelForImageTextToText.from_pretrained(
        model_config.pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    apply_freeze_policy(
        model,
        freeze_vit=model_config.freeze_vit,
        freeze_merger=model_config.freeze_merger,
        freeze_llm=model_config.freeze_llm,
    )

    return model
