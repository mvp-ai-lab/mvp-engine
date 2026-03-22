"""Pydantic v2 schema for the archived ViT model-compile reference."""

from pydantic import BaseModel, ConfigDict, Field

from mvp_engine.config.schema import BaseEngineConfig, BaseOptimConfig


class ViTDataConfig(BaseModel):
    model_config = ConfigDict(frozen=False)

    use_fake_data: bool = True
    train_path: str = "./data/imagenet/train"
    val_path: str = "./data/imagenet/val"
    fake_train_size: int = 1024
    fake_val_size: int = 256
    num_classes: int = Field(1000, ge=1)
    image_size: int = Field(224, ge=1)
    mean: list[float] = [0.485, 0.456, 0.406]
    std: list[float] = [0.229, 0.224, 0.225]
    batch_size: int = Field(64, ge=1)
    num_workers: int = Field(4, ge=0)


class ViTModelConfig(BaseModel):
    model_config = ConfigDict(frozen=False)

    pretrained_model_name_or_path: str = "google/vit-base-patch16-224"
    load_pretrained_weights: bool = False
    num_classes: int = Field(1000, ge=1)
    image_size: int = Field(224, ge=1)
    patch_size: int = Field(16, ge=1)
    num_channels: int = Field(3, ge=1)
    hidden_size: int = Field(768, ge=1)
    intermediate_size: int = Field(3072, ge=1)
    num_hidden_layers: int = Field(12, ge=1)
    num_attention_heads: int = Field(12, ge=1)
    hidden_dropout_prob: float = Field(0.0, ge=0.0, le=1.0)
    attention_dropout_prob: float = Field(0.0, ge=0.0, le=1.0)
    compile: bool = False
    compile_backend: str = "inductor"
    compile_mode: str = "default"


class ViTClassificationConfig(BaseEngineConfig):
    data: ViTDataConfig = Field(default_factory=ViTDataConfig)
    model: ViTModelConfig = Field(default_factory=ViTModelConfig)
    optim: BaseOptimConfig = Field(default_factory=BaseOptimConfig)
