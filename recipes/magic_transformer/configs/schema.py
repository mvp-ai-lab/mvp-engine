"""Pydantic schema for the Magic Transformer fake-data recipe."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mvp_engine.config.schema import BaseEngineConfig, BaseOptimConfig


class MagicTransformerDataConfig(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")

    fake_train_size: int = Field(128, ge=1)
    fake_eval_size: int = Field(32, ge=1)
    seq_len: int = Field(16, ge=2)
    vocab_size: int = Field(1024, ge=2)
    batch_size: int = Field(8, ge=1)
    num_workers: int = Field(0, ge=0)


class MagicTransformerModelConfig(BaseModel):
    model_config = ConfigDict(frozen=False, extra="forbid")

    vocab_size: int = Field(1024, ge=2)
    max_seq_len: int = Field(16, ge=2)
    d_model: int = Field(128, ge=1)
    n_heads: int = Field(4, ge=1)
    n_kv_heads: int = Field(2, ge=1)
    n_layers: int = Field(4, ge=2)
    dropout: float = Field(0.0, ge=0.0, le=1.0)
    rope_base: float = Field(10000.0, gt=0.0)
    mod_top_k_ratio: float = Field(0.5, gt=0.0, le=1.0)
    dual_stream: bool = True
    router_ema_decay: float = Field(0.9, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_attention_layout(self) -> "MagicTransformerModelConfig":
        if self.d_model % self.n_heads != 0:
            raise ValueError("`model.d_model` must be divisible by `model.n_heads`.")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("`model.n_heads` must be divisible by `model.n_kv_heads`.")
        if self.n_layers % 2 != 0:
            raise ValueError("`model.n_layers` must be even for the dual-stream block layout.")
        return self


class MagicTransformerConfig(BaseEngineConfig):
    model_config = ConfigDict(frozen=False, extra="forbid")

    data: MagicTransformerDataConfig = Field(default_factory=MagicTransformerDataConfig)
    model: MagicTransformerModelConfig = Field(default_factory=MagicTransformerModelConfig)
    optim: BaseOptimConfig = Field(default_factory=BaseOptimConfig)

    @model_validator(mode="after")
    def validate_shared_vocab_and_length(self) -> "MagicTransformerConfig":
        if self.data.vocab_size != self.model.vocab_size:
            raise ValueError("`data.vocab_size` must match `model.vocab_size`.")
        if self.data.seq_len > self.model.max_seq_len:
            raise ValueError("`data.seq_len` must be <= `model.max_seq_len`.")
        return self
