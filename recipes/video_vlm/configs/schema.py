"""Pydantic schema for the Video VLM recipe."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mvp_engine.config.schema import BaseEngineConfig, BaseLoopConfig, BaseOptimConfig


class VideoVLMDataConfig(BaseModel):
    """Dataset and batching options for the Video VLM recipe."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    train_path: str = ""
    ref_columns: list[str] = Field(default_factory=lambda: ["images"])
    thinking_mode: bool | None | Literal["non-empty"] = "non-empty"
    video_placeholder: str = "<video>"
    codec_enabled: bool = True
    codec_num_frames: int = Field(64, ge=1)
    codec_packed_frames: int = Field(8, ge=1)
    codec_frame_size: int = Field(224, ge=1)
    codec_patch_size: int = Field(14, ge=1)
    codec_k_keep: int = Field(2048, ge=1)
    hevc_decoder_bin: str | None = None
    cv_reader_required: bool = True
    packing: bool = False
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

    @field_validator("video_placeholder")
    @classmethod
    def validate_video_placeholder(cls, value: str) -> str:
        """Normalize the text marker that is replaced by video tokens."""
        if not isinstance(value, str):
            raise TypeError("`data.video_placeholder` must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ValueError("`data.video_placeholder` must not be empty.")
        return normalized

    @field_validator("codec_k_keep")
    @classmethod
    def validate_codec_k_keep(cls, value: int, info) -> int:
        """Require packed codec patches to form an integral dense frame stack."""
        data = info.data
        frame_size = int(data.get("codec_frame_size", 224))
        patch_size = int(data.get("codec_patch_size", 14))
        packed_frames = int(data.get("codec_packed_frames", 8))
        if frame_size % patch_size != 0:
            raise ValueError("`data.codec_frame_size` must be divisible by `data.codec_patch_size`.")
        patches_per_frame = (frame_size // patch_size) ** 2
        expected = packed_frames * patches_per_frame
        if int(value) != expected:
            raise ValueError(
                "`data.codec_k_keep` must equal "
                "`data.codec_packed_frames * (data.codec_frame_size / data.codec_patch_size) ** 2`; "
                f"expected {expected}, got {value}."
            )
        return value


class VideoVLMGradientCheckpointingConfig(BaseModel):
    """Gradient checkpointing options passed to the Qwen3-VL model."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    enabled: bool = False
    use_reentrant: bool = False


class VideoVLMModelConfig(BaseModel):
    """Model loading, precision compatibility, and freeze-policy options."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    pretrained_model_name_or_path: str = "./recipes/video_vlm/pretrained/Qwen3-VL-8B-Instruct"
    vision_encoder_name_or_path: str = "lmms-lab-encoder/onevision-encoder-large"
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "flash_attention_2"
    image_max_pixels: int | None = Field(None, ge=1)
    gradient_checkpointing: VideoVLMGradientCheckpointingConfig = Field(
        default_factory=VideoVLMGradientCheckpointingConfig
    )

    # Freeze flags for each sub-module.  Default follows the alignment-stage
    # convention: train only the merger while keeping ViT and LLM frozen.
    freeze_vit: bool = True
    freeze_vision_encoder: bool = True
    freeze_merger: bool = False
    freeze_llm: bool = False

    compile: bool = True
    compile_backend: str = "inductor"
    compile_mode: str = "default"

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


class VideoVLMOptimConfig(BaseOptimConfig):
    """Optimizer options extended with Video VLM batch and loss-guard settings."""

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


class VideoVLMLoopConfig(BaseLoopConfig):
    """Iteration-loop options for Video VLM training."""

    model_config = ConfigDict(frozen=False)

    total_steps: int = 10000

    @field_validator("total_steps")
    @classmethod
    def validate_total_steps(cls, value: int) -> int:
        """Allow a positive step count or ``-1`` for dataset-based inference."""
        if value == 0 or value < -1:
            raise ValueError("`loop.total_steps` must be positive or exactly -1.")
        return value


class VideoVLMConfig(BaseEngineConfig):
    """Top-level Video VLM recipe config."""

    model_config = ConfigDict(frozen=False, extra="allow")

    data: VideoVLMDataConfig = Field(default_factory=VideoVLMDataConfig)
    model: VideoVLMModelConfig = Field(default_factory=VideoVLMModelConfig)
    optim: VideoVLMOptimConfig = Field(default_factory=VideoVLMOptimConfig)
    loop: VideoVLMLoopConfig = Field(default_factory=VideoVLMLoopConfig)
