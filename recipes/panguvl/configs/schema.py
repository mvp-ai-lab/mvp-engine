"""Pydantic schema for the PanguVL recipe."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mvp_engine.config.schema import BaseEngineConfig, BaseLoopConfig, BaseOptimConfig


class PanguvlDataConfig(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")

    train_path: Optional[str] = "./data/panguvl/alignment_demo.jsonl"
    cache_dir: str | None = None
    enable_thinking: bool | None | Literal["non-empty"] = "non-empty"
    cache: bool = False
    shuffle_buffer: int = Field(1000, ge=1)
    packing: bool = False
    shuffle_on_packs: bool = False
    shuffle_on_packs_buffer: int = Field(256, ge=1)
    packing_selection_strategy: Literal["random", "best_fit"] = "best_fit"
    packing_open_pack_limit: int = Field(8, ge=1)
    packing_buffer_size: int = Field(64, ge=0)
    max_seq_len: int = Field(2048, ge=1)
    batch_size: int = 1
    num_workers: int = Field(0, ge=0)

    @field_validator("train_path", mode="before")
    @classmethod
    def validate_train_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("`data.train_path` must be a string or null.")

        normalized = value.strip()
        if not normalized:
            raise ValueError("`data.train_path` must not be empty.")
        return normalized

    @field_validator("cache_dir", mode="before")
    @classmethod
    def validate_cache_dir(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("`data.cache_dir` must be a string or null.")

        normalized = value.strip()
        if not normalized:
            raise ValueError("`data.cache_dir` must not be empty when provided.")
        return normalized

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: int) -> int:
        if value == 0 or value < -1:
            raise ValueError("`data.batch_size` must be positive or exactly -1.")
        return value

    @field_validator("enable_thinking", mode="before")
    @classmethod
    def validate_enable_thinking(cls, value: bool | str | None) -> bool | None | Literal["non-empty"]:
        if value is None or isinstance(value, bool):
            return value
        if not isinstance(value, str):
            raise TypeError("`data.enable_thinking` must be a bool, null, or 'non-empty'.")

        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        if normalized in {"none", "null"}:
            return None
        if normalized == "non-empty":
            return "non-empty"
        raise ValueError("`data.enable_thinking` only accepts true, false, null, or 'non-empty'.")


class PanguvlGradientCheckpointingConfig(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")

    enabled: bool = False
    use_reentrant: bool = False


class PanguvlModelConfig(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")

    pretrained_model_name_or_path: str = "./recipes/panguvl/pretrained/Qwen3-VL-8B-Instruct"
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "flash_attention_2"
    image_max_pixels: int | None = Field(None, ge=1)
    gradient_checkpointing: PanguvlGradientCheckpointingConfig = Field(
        default_factory=PanguvlGradientCheckpointingConfig
    )

    # Freeze flags for each sub-module.  Default follows the alignment-stage
    # convention: train only the merger while keeping ViT and LLM frozen.
    freeze_vit: bool = True
    freeze_merger: bool = False
    freeze_llm: bool = False

    compile: bool = True
    compile_backend: str = "inductor"
    compile_mode: str = "default"

    @field_validator("pretrained_model_name_or_path", mode="before")
    @classmethod
    def validate_pretrained_model_name_or_path(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("`model.pretrained_model_name_or_path` must be a string.")

        normalized = value.strip()
        if not normalized:
            raise ValueError("`model.pretrained_model_name_or_path` must not be empty.")
        return normalized


class PanguvlOptimConfig(BaseOptimConfig):
    model_config = ConfigDict(frozen=False)

    gradient_accumulation_steps: int = 1
    global_batch_size: int | None = Field(None, ge=1)

    @field_validator("gradient_accumulation_steps")
    @classmethod
    def validate_gradient_accumulation_steps(cls, value: int) -> int:
        if value == 0 or value < -1:
            raise ValueError("`optim.gradient_accumulation_steps` must be positive or exactly -1.")
        return value


class PanguvlLoopConfig(BaseLoopConfig):
    model_config = ConfigDict(frozen=False)

    total_steps: int = 10000

    @field_validator("total_steps")
    @classmethod
    def validate_total_steps(cls, value: int) -> int:
        if value == 0 or value < -1:
            raise ValueError("`loop.total_steps` must be positive or exactly -1.")
        return value


class PanguvlConfig(BaseEngineConfig):
    model_config = ConfigDict(frozen=False, extra="allow")

    data: PanguvlDataConfig = Field(default_factory=PanguvlDataConfig)
    model: PanguvlModelConfig = Field(default_factory=PanguvlModelConfig)
    optim: PanguvlOptimConfig = Field(default_factory=PanguvlOptimConfig)
    loop: PanguvlLoopConfig = Field(default_factory=PanguvlLoopConfig)
