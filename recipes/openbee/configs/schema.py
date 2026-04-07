"""Pydantic schema for the OpenBee recipe."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mvp_engine.config.schema import BaseEngineConfig


class OpenbeeDataConfig(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")

    train_path: Optional[str] = "./data/openbee/alignment_demo.jsonl"
    enable_thinking: bool = True
    cache: bool = False
    cache_show_progress: bool = True
    shuffle_buffer: int = Field(1000, ge=1)
    packing: bool = False
    packing_selection_strategy: Literal["random", "best_fit"] = "best_fit"
    packing_open_pack_limit: int = Field(8, ge=1)
    packing_buffer_size: int = Field(64, ge=-1)
    max_seq_len: int = Field(2048, ge=1)
    batch_size: int = Field(1, ge=1)
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


class OpenbeeModelConfig(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")

    pretrained_model_name_or_path: str = "./recipes/openbee/pretrained/Qwen3-VL-8B-Instruct"
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "flash_attention_2"
    image_max_pixels: int | None = Field(None, ge=1)

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


class OpenbeeConfig(BaseEngineConfig):
    model_config = ConfigDict(frozen=False, extra="allow")

    data: OpenbeeDataConfig = Field(default_factory=OpenbeeDataConfig)
    model: OpenbeeModelConfig = Field(default_factory=OpenbeeModelConfig)
