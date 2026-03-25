"""MFU-only schema excerpt (non-MFU fields intentionally hidden)."""

from pydantic import BaseModel, Field


class ViTMFUConfig(BaseModel):
    device_name: str = "NVIDIA H200"
    peak_tflops: float = Field(989.0, gt=0.0)


class ViTModelConfig(BaseModel):
    image_size: int = Field(224, ge=1)
    patch_size: int = Field(16, ge=1)


class ViTLogConfig(BaseModel):
    mfu: ViTMFUConfig = Field(default_factory=ViTMFUConfig)


class ViTClassificationConfig(BaseModel):
    model: ViTModelConfig = Field(default_factory=ViTModelConfig)
    log: ViTLogConfig = Field(default_factory=ViTLogConfig)


# Other training/data/runtime schema fields are intentionally hidden.
