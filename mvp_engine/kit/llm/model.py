"""Reusable text-only LM model utilities."""

from __future__ import annotations

from typing import Any, Callable, Literal

import torch


class LLMModelKit:
    """Group reusable text-LM model build and preparation utilities."""

    def __init__(self) -> None:
        """Disable Transformers progress bars for recipe-level model setup."""
        from transformers.utils.logging import disable_progress_bar

        disable_progress_bar()

    def build_model(
        self,
        pretrained_model_name_or_path: str,
        *,
        load_pretrained_model: bool = True,
        random_init_seed: int = 42,
        trust_remote_code: bool = True,
        torch_dtype: str | torch.dtype = "auto",
        attn_implementation: str | None = None,
        **kwargs: Any,
    ):
        """Load a causal LM from Hugging Face, or randomly initialize one from its config.

        When ``load_pretrained_model`` is False, seed the RNG and build the model
        from its config only, without loading pretrained weights.
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        if load_pretrained_model:
            model_kwargs: dict[str, Any] = {
                "trust_remote_code": trust_remote_code,
                "torch_dtype": torch_dtype,
                **kwargs,
            }
            if attn_implementation is not None:
                model_kwargs["attn_implementation"] = attn_implementation
            return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **model_kwargs)

        from accelerate.utils import set_seed

        set_seed(random_init_seed)
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation
        config_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if torch_dtype != "auto":
            config_kwargs["torch_dtype"] = torch_dtype
        return AutoModelForCausalLM.from_config(config, **config_kwargs)

    def apply_freeze_policy(
        self,
        model: torch.nn.Module,
        freeze_llm: bool = False,
        llm_prefixes: tuple[str, ...] | None = None,
    ) -> torch.nn.Module:
        """Apply a prefix-based language-model freeze policy."""
        if not freeze_llm:
            return model

        llm_prefixes = llm_prefixes or (
            "model.",
            "transformer.",
            "backbone.",
            "decoder.",
            "language_model.",
            "lm_head.",
        )
        for name, parameter in model.named_parameters():
            if _matches(name, llm_prefixes):
                parameter.requires_grad = False
        return model

    def apply_gradient_checkpointing(
        self,
        model: torch.nn.Module,
        use_reentrant: bool = False,
        mode: Literal["hf", "custom", "hf_with_custom"] = "hf",
        target_modules: list[str] | None = None,
    ) -> torch.nn.Module:
        """Enable gradient checkpointing through HF and optional module wrappers."""
        if mode in ("hf", "hf_with_custom"):
            if not hasattr(model, "gradient_checkpointing_enable"):
                raise AttributeError(f"{model.__class__.__name__} does not support gradient checkpointing.")
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
            if hasattr(model, "config"):
                setattr(model.config, "use_cache", False)

            if mode == "hf":
                return model

        if not target_modules:
            raise ValueError("`target_modules` is required when gradient checkpointing mode uses custom wrappers.")

        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            checkpoint_wrapper,
        )

        checkpoint_impl = CheckpointImpl.REENTRANT if use_reentrant else CheckpointImpl.NO_REENTRANT
        target_parent_names, target_names = [], []
        for name in target_modules or []:
            if ":" not in name:
                raise ValueError(f"Invalid target module name '{name}'. Expected format 'parent_module:child_module'.")
            parent_name, child_name = name.split(":")
            target_parent_names.append(parent_name.strip())
            target_names.append(child_name.strip())

        for parent in model.modules():
            if parent.__class__.__name__ not in target_parent_names:
                continue
            for child_name, child in list(parent.named_children()):
                if child_name not in target_names or hasattr(child, "_checkpoint_wrapped_module"):
                    continue
                parent.add_module(
                    child_name,
                    checkpoint_wrapper(child, checkpoint_impl=checkpoint_impl, preserve_rng_state=False),
                )

        return model

    def apply_model_compile(
        self,
        model: torch.nn.Module,
        backend: str,
        mode: str,
    ) -> torch.nn.Module:
        """Compile the text model through ``torch.compile``."""
        model.compile(
            backend=backend,
            mode=mode,
        )
        return model

    def apply_model_patches(
        self,
        model: torch.nn.Module,
        patch_fns: Callable[[torch.nn.Module], torch.nn.Module] | list[Callable[[torch.nn.Module], torch.nn.Module]],
    ) -> torch.nn.Module:
        """Apply one or more recipe-specific patch functions to the model."""
        if not isinstance(patch_fns, list):
            patch_fns = [patch_fns]
        for patch_fn in patch_fns:
            model = patch_fn(model)
        return model


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    """Return whether a parameter name belongs to one configured prefix group."""
    return any(name.startswith(prefix) for prefix in prefixes)
