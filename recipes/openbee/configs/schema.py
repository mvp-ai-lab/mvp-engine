"""Pydantic schema for the OpenBee recipe."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mvp_engine.config.schema import BaseEngineConfig, BaseLoopConfig, BaseOptimConfig


class OpenBeeDataConfig(BaseModel):
    """Dataset and batching options for the OpenBee recipe."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    train_path: str = ""
    ref_columns: list[str] = Field(default_factory=lambda: ["images"])
    thinking_mode: bool | None | Literal["non-empty"] = "non-empty"
    packing_selection_strategy: Literal["random", "best_fit"] = "best_fit"
    packing_open_pack_limit: int = Field(8, ge=1)
    packing_buffer_size: int = Field(64, ge=0)
    max_seq_len: int = Field(2048, ge=1)
    batch_size: int = 1
    num_workers: int = Field(0, ge=0)

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


class OpenBeeGradientCheckpointingConfig(BaseModel):
    """Gradient checkpointing options passed to the Qwen3-VL model."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    enabled: bool = False
    use_reentrant: bool = False


class OpenBeeCompileConfig(BaseModel):
    """Model compile options for the OpenBee recipe."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    enabled: bool = True
    backend: str = "inductor"
    mode: str = "default"


class OpenBeeModelConfig(BaseModel):
    """Model loading, precision compatibility, and freeze-policy options."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    pretrained_model_name_or_path: str = "./recipes/openbee/pretrained/Qwen3-VL-8B-Instruct"
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "flash_attention_2"
    image_max_pixels: int | None = Field(None, ge=1)
    gradient_checkpointing: OpenBeeGradientCheckpointingConfig = Field(
        default_factory=OpenBeeGradientCheckpointingConfig
    )

    # Freeze flags for each sub-module.  Default follows the alignment-stage
    # convention: train only the merger while keeping ViT and LLM frozen.
    freeze_vit: bool = True
    freeze_projector: bool = False
    freeze_llm: bool = False

    compile: OpenBeeCompileConfig = Field(default_factory=OpenBeeCompileConfig)

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


class OpenBeeOptimConfig(BaseOptimConfig):
    """Optimizer options extended with OpenBee batch and loss-guard settings."""

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


class OpenBeeLoopConfig(BaseLoopConfig):
    """Iteration-loop options for OpenBee training."""

    model_config = ConfigDict(frozen=False)

    total_steps: int = 10000

    @field_validator("total_steps")
    @classmethod
    def validate_total_steps(cls, value: int) -> int:
        """Allow a positive step count or ``-1`` for dataset-based inference."""
        if value == 0 or value < -1:
            raise ValueError("`loop.total_steps` must be positive or exactly -1.")
        return value


class OpenBeeConfig(BaseEngineConfig):
    """Top-level OpenBee recipe config."""

    model_config = ConfigDict(frozen=False, extra="allow")

    data: OpenBeeDataConfig = Field(default_factory=OpenBeeDataConfig)
    model: OpenBeeModelConfig = Field(default_factory=OpenBeeModelConfig)
    optim: OpenBeeOptimConfig = Field(default_factory=OpenBeeOptimConfig)
    loop: OpenBeeLoopConfig = Field(default_factory=OpenBeeLoopConfig)

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
