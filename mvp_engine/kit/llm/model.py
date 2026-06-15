"""Reusable text-only LM model utilities."""

from __future__ import annotations

from typing import Any, Callable

import torch


class LLMModelKit:
    """Group reusable text-LM model build and preparation utilities."""

    def build_model(
        self,
        pretrained_model_name_or_path: str,
        *,
        train_from_scratch: bool = False,
        init_seed: int = 42,
        trust_remote_code: bool = True,
        torch_dtype: str | torch.dtype = "auto",
        attn_implementation: str | None = None,
    ):
        """Load a causal LM from Hugging Face, or randomly initialize one from its config.

        When ``train_from_scratch`` is True we seed the RNG and build the model from
        its config only (no pretrained weights) for from-scratch pretraining.
        Otherwise we load the pretrained weights as usual.
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        if train_from_scratch:
            from accelerate.utils import set_seed

            set_seed(init_seed)
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
            if attn_implementation is not None:
                config._attn_implementation = attn_implementation
            config_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
            if torch_dtype != "auto":
                config_kwargs["torch_dtype"] = torch_dtype
            return AutoModelForCausalLM.from_config(config, **config_kwargs)

        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, "torch_dtype": torch_dtype}
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **model_kwargs)

    def apply_freeze_policy(
        self,
        model: torch.nn.Module,
        freeze_llm: bool = False,
        llm_prefixes: tuple[str, ...] | None = None,
    ) -> torch.nn.Module:
        """Optionally freeze the language model parameters (default: train everything)."""
        if not freeze_llm:
            return model

        llm_prefixes = llm_prefixes or ("model.", "lm_head.")
        for name, parameter in model.named_parameters():
            if _matches(name, llm_prefixes):
                parameter.requires_grad = False
        return model

    def apply_gradient_checkpointing(
        self,
        model: torch.nn.Module,
        use_reentrant: bool = False,
    ) -> torch.nn.Module:
        """Enable Hugging Face native gradient checkpointing."""
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise AttributeError(f"{model.__class__.__name__} does not support gradient checkpointing.")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
        model.config.use_cache = False
        return model

    def apply_model_compile(self, model: torch.nn.Module, backend: str, mode: str) -> torch.nn.Module:
        """Compile the whole model with ``torch.compile``."""
        model.compile(backend=backend, mode=mode)
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
    """Return whether a parameter name starts with one of the prefixes."""
    return any(name.startswith(prefix) for prefix in prefixes)
