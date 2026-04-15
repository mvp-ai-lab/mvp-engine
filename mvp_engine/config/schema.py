"""Pydantic v2 schemas for the mvp-engine configuration."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseLogConfig(BaseModel):
    model_config = ConfigDict(frozen=False)

    interval: int = Field(20, ge=1)
    backends: list[Literal["terminal", "file", "wandb"]] = ["terminal", "file"]
    timer_window_size: int = Field(100, ge=1)


class BaseProjectConfig(BaseModel):
    model_config = ConfigDict(frozen=False)

    name: str = "mvp-engine"
    dir: str = "./outputs"


class BaseRuntimeInfo(BaseModel):
    model_config = ConfigDict(frozen=False)

    git_info: str = ""
    run_id: str = ""
    output_dir: str = ""
    world_size: int = 1
    hostname: str = ""
    python_version: str = ""
    torch_version: str = ""


class BaseMeshConfig(BaseModel):
    model_config = ConfigDict(frozen=False)

    replicate: int = 1
    shard: int = 8
    tensor: int = 1

    @field_validator("replicate", "shard", "tensor")
    @classmethod
    def validate_mesh_dim(cls, v: int) -> int:
        if v == 0 or v < -1:
            raise ValueError("Mesh dimensions must be >= 1 or exactly -1")
        return v


class BaseMpPolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=False)

    param_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    reduce_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    output_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    buffer_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"


class BaseFSDP2Config(BaseModel):
    model_config = ConfigDict(frozen=False, extra="allow")

    reshard_after_forward: bool = True
    offload_policy: bool = False
    mp_policy: BaseMpPolicyConfig = Field(default_factory=BaseMpPolicyConfig)
    high_precision_modules: list[str] = []
    target_classes: list[str] = []


class BaseDDPConfig(BaseModel):
    model_config = ConfigDict(frozen=False, extra="allow")


class BaseBackendKwargsConfig(BaseModel):
    model_config = ConfigDict(frozen=False)

    fsdp2: BaseFSDP2Config = Field(default_factory=BaseFSDP2Config)
    ddp: BaseDDPConfig = Field(default_factory=BaseDDPConfig)


class BaseParallelConfig(BaseModel):
    model_config = ConfigDict(frozen=False)

    mesh: BaseMeshConfig = Field(default_factory=BaseMeshConfig)
    backend_kwargs: BaseBackendKwargsConfig = Field(default_factory=BaseBackendKwargsConfig)


class BaseOptimConfig(BaseModel):
    model_config = ConfigDict(frozen=False)

    lr: float = 1e-3
    weight_decay: float = 0.05
    gradient_accumulation_steps: int = Field(1, ge=1)
    mixed_precision: Literal["fp32", "fp16", "bf16"] = "bf16"
    warmup_ratio: float = Field(0.1, ge=0.0, le=1.0)
    clip_grad_norm: Optional[float] = 5.0


class BaseCheckpointConfig(BaseModel):
    model_config = ConfigDict(frozen=False)

    keep_n: int = Field(5, ge=1)
    interval: int = Field(5000, ge=1)
    hf_enable: bool = False


class BaseLoopConfig(BaseModel):
    model_config = ConfigDict(frozen=False)

    policy: Literal["iter", "epoch"] = "iter"
    total_steps: int = Field(10000, ge=1)


class BaseEngineConfig(BaseModel):
    model_config = ConfigDict(frozen=False)

    dev_mode: bool = False
    engine: str = "Engine"
    seed: int = 42
    deterministic: bool = False
    init_from_checkpoint: str | None = None
    project: BaseProjectConfig = Field(default_factory=BaseProjectConfig)
    runtime: BaseRuntimeInfo = Field(default_factory=BaseRuntimeInfo)
    log: BaseLogConfig = Field(default_factory=BaseLogConfig)
    parallel: BaseParallelConfig = Field(default_factory=BaseParallelConfig)
    optim: BaseOptimConfig = Field(default_factory=BaseOptimConfig)
    loop: BaseLoopConfig = Field(default_factory=BaseLoopConfig)
    checkpoint: BaseCheckpointConfig = Field(default_factory=BaseCheckpointConfig)

    @field_validator("init_from_checkpoint", mode="before")
    @classmethod
    def validate_init_from_checkpoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("`init_from_checkpoint` must be a string or null.")

        normalized = value.strip()
        if not normalized:
            raise ValueError("`init_from_checkpoint` must not be empty.")
        return normalized
