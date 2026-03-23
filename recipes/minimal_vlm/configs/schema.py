"""Pydantic schema for the minimal VLM recipe."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from mvp_engine.config.schema import BaseEngineConfig


class MinimalVLMDataConfig(BaseModel):
    model_config = ConfigDict(frozen=False, extra="allow")

    train_path: Optional[str] = "./data/minimal_vlm/demo.jsonl"
    jsonl_num_shards: Optional[int] = Field(default=None, ge=1)
    shuffle_buffer: int = Field(128, ge=1)
    packing: bool = False
    packing_selection_strategy: Literal["random", "best_fit"] = "best_fit"
    packing_open_pack_limit: int = Field(8, ge=1)
    packing_buffer_size: int = Field(64, ge=-1)
    loader_prefetch_factor: int = Field(2, ge=1)
    max_seq_len: int = Field(2048, ge=1)
    batch_size: int = Field(1, ge=1)
    num_workers: int = Field(0, ge=0)


class MinimalVLMModelConfig(BaseModel):
    model_config = ConfigDict(frozen=False, extra="allow")

    pretrained_model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] | None = None
    trust_remote_code: bool = True
    freeze_vit: bool = True
    freeze_projector: bool = True
    freeze_llm: bool = False


class MinimalVLMConfig(BaseEngineConfig):
    model_config = ConfigDict(frozen=False, extra="allow")

    data: MinimalVLMDataConfig = Field(default_factory=MinimalVLMDataConfig)
    model: MinimalVLMModelConfig = Field(default_factory=MinimalVLMModelConfig)
