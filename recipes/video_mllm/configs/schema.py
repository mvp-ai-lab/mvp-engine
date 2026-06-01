"""Pydantic schema for the video MLLM recipe."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mvp_engine.config.schema import BaseEngineConfig, BaseLoopConfig, BaseOptimConfig


class VideoMLLMDataConfig(BaseModel):
    """Dataset and batching options for the video MLLM recipe.

    Video samples are kept unpacked: uniform frame sampling (the swappable seam
    in ``dataset/sampling.py``) plus decode-then-expand preprocessing. There are
    no packing options.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    train_path: str = "./data/video_mllm/smoke.jsonl"
    source: Literal["jsonl", "parquet"] = "jsonl"
    video_root: str | None = None
    max_seq_len: int = Field(8192, ge=1)
    batch_size: int = 1
    num_workers: int = Field(0, ge=0)
    # Uniform frame-sampling budget; the swappable seam lives in dataset/sampling.py.
    num_frames: int = Field(16, ge=1)

    @field_validator("train_path", mode="before")
    @classmethod
    def validate_train_path(cls, value: str | None) -> str | None:
        """Normalize the optional train dataset path."""
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


class VideoMLLMGradientCheckpointingConfig(BaseModel):
    """Gradient checkpointing options passed to the Qwen3-VL model."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    enabled: bool = False
    use_reentrant: bool = False


class VideoMLLMCompileConfig(BaseModel):
    """Model compile options for the video MLLM recipe."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    enabled: bool = False
    backend: str = "inductor"
    mode: str = "default"


class VideoMLLMModelConfig(BaseModel):
    """Model loading, precision compatibility, and freeze-policy options."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    pretrained_model_name_or_path: str = "./pretrained/Qwen3-VL-8B-Base-woDS-stage0"
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "flash_attention_2"
    image_max_pixels: int | None = Field(None, ge=1)
    gradient_checkpointing: VideoMLLMGradientCheckpointingConfig = Field(
        default_factory=VideoMLLMGradientCheckpointingConfig
    )

    # Video SFT defaults: train the language model, keep ViT and projector frozen.
    freeze_vit: bool = True
    freeze_projector: bool = True
    freeze_llm: bool = False

    compile: VideoMLLMCompileConfig = Field(default_factory=VideoMLLMCompileConfig)

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


class VideoMLLMOptimConfig(BaseOptimConfig):
    """Optimizer options extended with batch and loss-guard settings."""

    model_config = ConfigDict(frozen=False)

    optimizer: str = "AdamW"
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


class VideoMLLMLoopConfig(BaseLoopConfig):
    """Iteration-loop options for video MLLM training."""

    model_config = ConfigDict(frozen=False)

    total_steps: int = 1000

    @field_validator("total_steps")
    @classmethod
    def validate_total_steps(cls, value: int) -> int:
        """Require an explicit positive step budget (this recipe has no total_steps auto-inference)."""
        if value < 1:
            raise ValueError("`loop.total_steps` must be >= 1 (video_mllm does not auto-infer total_steps).")
        return value


class VideoMLLMConfig(BaseEngineConfig):
    """Top-level video MLLM recipe config."""

    model_config = ConfigDict(frozen=False, extra="allow")

    data: VideoMLLMDataConfig = Field(default_factory=VideoMLLMDataConfig)
    model: VideoMLLMModelConfig = Field(default_factory=VideoMLLMModelConfig)
    optim: VideoMLLMOptimConfig = Field(default_factory=VideoMLLMOptimConfig)
    loop: VideoMLLMLoopConfig = Field(default_factory=VideoMLLMLoopConfig)

    def resolve_batching_config(self, *, data_parallel_world_size: int) -> None:
        """Resolve global batch size into micro batch size or accumulation steps."""
        target_global_batch_size = self.optim.global_batch_size
        accumulation_steps = int(self.optim.gradient_accumulation_steps)
        micro_batch_size = int(self.data.batch_size)

        if target_global_batch_size is None:
            if accumulation_steps == -1:
                raise ValueError("`optim.gradient_accumulation_steps=-1` requires `optim.global_batch_size`.")
            if micro_batch_size == -1:
                raise ValueError("`data.batch_size=-1` requires `optim.global_batch_size`.")
            return

        if micro_batch_size == -1 and accumulation_steps == -1:
            raise ValueError(
                "`optim.global_batch_size` cannot infer both `data.batch_size` and "
                "`optim.gradient_accumulation_steps` at the same time."
            )

        if micro_batch_size == -1:
            batch_size_divisor = int(data_parallel_world_size) * accumulation_steps
            if batch_size_divisor <= 0 or target_global_batch_size % batch_size_divisor != 0:
                raise ValueError(
                    "`data.batch_size` cannot be inferred exactly: "
                    "`optim.global_batch_size` must be divisible by "
                    "`data_parallel_world_size * optim.gradient_accumulation_steps`."
                )
            self.data.batch_size = target_global_batch_size // batch_size_divisor
            micro_batch_size = int(self.data.batch_size)
        elif accumulation_steps == -1:
            accumulation_divisor = int(data_parallel_world_size) * micro_batch_size
            if accumulation_divisor <= 0 or target_global_batch_size % accumulation_divisor != 0:
                raise ValueError(
                    "`optim.gradient_accumulation_steps` cannot be inferred exactly: "
                    "`optim.global_batch_size` must be divisible by "
                    "`data_parallel_world_size * data.batch_size`."
                )
            self.optim.gradient_accumulation_steps = target_global_batch_size // accumulation_divisor
            accumulation_steps = int(self.optim.gradient_accumulation_steps)

        effective_global_batch_size = int(data_parallel_world_size) * micro_batch_size * accumulation_steps
        if effective_global_batch_size != target_global_batch_size:
            raise ValueError(
                "`optim.global_batch_size` does not match the configured batching: "
                f"expected {effective_global_batch_size} from "
                "`data_parallel_world_size * data.batch_size * optim.gradient_accumulation_steps`."
            )
