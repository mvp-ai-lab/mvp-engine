"""Pydantic schema for the Qwen3 LM recipe."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mvp_engine.config.schema import BaseEngineConfig, BaseLoopConfig, BaseOptimConfig


class Qwen3LMDataConfig(BaseModel):
    """Dataset and batching options for the Qwen3 LM recipe."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    source_type: Literal["jsonl", "mvp_dataset"] = "jsonl"
    source_format: Literal["jsonl", "parquet", "lance"] = "jsonl"
    train_path: str = ""
    thinking_mode: bool | None | Literal["non-empty"] = "non-empty"
    packing: bool = True
    packing_selection_strategy: Literal["random", "best_fit"] = "best_fit"
    packing_open_pack_limit: int = Field(8, ge=1)
    packing_buffer_size: int = Field(64, ge=0)
    max_seq_len: int = Field(8192, ge=1)
    batch_size: int = 1
    num_workers: int = Field(4, ge=0)

    @field_validator("train_path", mode="before")
    @classmethod
    def validate_train_path(cls, value: str | None) -> str | None:
        """Normalize the training dataset path."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("`data.train_path` must be a string or null.")

        normalized = value.strip()
        if not normalized:
            raise ValueError("`data.train_path` must not be empty.")
        return normalized

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: int) -> int:
        """Allow a positive micro batch size or ``-1`` for inference."""
        if value == 0 or value < -1:
            raise ValueError("`data.batch_size` must be positive or exactly -1.")
        return value

    @field_validator("thinking_mode", mode="before")
    @classmethod
    def validate_thinking_mode(cls, value: bool | str | None) -> bool | None | Literal["non-empty"]:
        """Normalize string forms of the assistant thinking-block policy."""
        if value is None or isinstance(value, bool):
            return value
        if not isinstance(value, str):
            raise TypeError("`data.thinking_mode` must be a bool, null, or 'non-empty'.")

        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        if normalized in {"none", "null"}:
            return None
        if normalized == "non-empty":
            return "non-empty"
        raise ValueError("`data.thinking_mode` only accepts true, false, null, or 'non-empty'.")


class Qwen3LMGradientCheckpointingConfig(BaseModel):
    """Gradient checkpointing options passed to the Qwen3 model."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    enabled: bool = False
    use_reentrant: bool = False


class Qwen3LMModelConfig(BaseModel):
    """Model loading, tiny-smoke, and runtime patch options."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    pretrained_model_name_or_path: str = "Qwen/Qwen3-8B"
    tokenizer_name_or_path: str | None = None
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "flash_attention_2"
    gradient_checkpointing: Qwen3LMGradientCheckpointingConfig = Field(
        default_factory=Qwen3LMGradientCheckpointingConfig
    )
    compile: bool = False
    compile_backend: str = "inductor"
    compile_mode: str = "default"
    loss_chunk_size: int = Field(4096, ge=1)
    upcast_trainable_params: bool = True

    tiny_random: bool = False
    tiny_vocab_size: int = Field(256, ge=32)
    tiny_hidden_size: int = Field(64, ge=16)
    tiny_intermediate_size: int = Field(128, ge=16)
    tiny_num_hidden_layers: int = Field(2, ge=1)
    tiny_num_attention_heads: int = Field(4, ge=1)
    tiny_num_key_value_heads: int = Field(2, ge=1)

    @field_validator("pretrained_model_name_or_path", mode="before")
    @classmethod
    def validate_pretrained_model_name_or_path(cls, value: str) -> str:
        """Normalize the model name or local checkpoint path."""
        if not isinstance(value, str):
            raise TypeError("`model.pretrained_model_name_or_path` must be a string.")

        normalized = value.strip()
        if not normalized:
            raise ValueError("`model.pretrained_model_name_or_path` must not be empty.")
        return normalized

    @field_validator("tokenizer_name_or_path", mode="before")
    @classmethod
    def validate_tokenizer_name_or_path(cls, value: str | None) -> str | None:
        """Normalize the optional tokenizer path."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("`model.tokenizer_name_or_path` must be a string or null.")

        normalized = value.strip()
        return normalized or None


class Qwen3LMOptimConfig(BaseOptimConfig):
    """Optimizer options extended with Qwen3 LM batch and loss-guard settings."""

    model_config = ConfigDict(frozen=False)

    gradient_accumulation_steps: int = 1
    global_batch_size: int | None = Field(None, ge=1)
    loss_spike_skip_multiplier: float | None = Field(None, gt=0.0)
    loss_spike_skip_window_size: int = Field(8, ge=1)
    loss_spike_skip_min_history: int = Field(3, ge=1)

    @field_validator("gradient_accumulation_steps")
    @classmethod
    def validate_gradient_accumulation_steps(cls, value: int) -> int:
        """Allow a positive accumulation count or ``-1`` for inference."""
        if value == 0 or value < -1:
            raise ValueError("`optim.gradient_accumulation_steps` must be positive or exactly -1.")
        return value


class Qwen3LMLoopConfig(BaseLoopConfig):
    """Iteration-loop options for Qwen3 LM training."""

    model_config = ConfigDict(frozen=False)

    total_steps: int = 10000

    @field_validator("total_steps")
    @classmethod
    def validate_total_steps(cls, value: int) -> int:
        """Allow a positive step count or ``-1`` for dataset-based inference."""
        if value == 0 or value < -1:
            raise ValueError("`loop.total_steps` must be positive or exactly -1.")
        return value


class Qwen3LMConfig(BaseEngineConfig):
    """Top-level Qwen3 LM recipe config."""

    model_config = ConfigDict(frozen=False, extra="allow")

    data: Qwen3LMDataConfig = Field(default_factory=Qwen3LMDataConfig)
    model: Qwen3LMModelConfig = Field(default_factory=Qwen3LMModelConfig)
    optim: Qwen3LMOptimConfig = Field(default_factory=Qwen3LMOptimConfig)
    loop: Qwen3LMLoopConfig = Field(default_factory=Qwen3LMLoopConfig)
