"""Minimal VLM engine scaffold."""

from typing import Any

import torch
from torch.utils.data import DataLoader

from mvp_engine.engine import ENGINE_REGISTRY, Engine


@ENGINE_REGISTRY.register()
class MinimalVlmEngine(Engine):
    """Minimal engine scaffold for a vision-language training recipe."""

    def prepare_dataloader(self, workflow: str = "train") -> DataLoader:
        """Build the dataloader that yields image-text batches."""
        raise NotImplementedError("Implement prepare_dataloader() in recipes/minimal_vlm/engine/minimal_vlm_engine.py.")

    def prepare_model(self) -> torch.nn.Module:
        """Build the VLM used by the recipe."""
        raise NotImplementedError("Implement prepare_model() in recipes/minimal_vlm/engine/minimal_vlm_engine.py.")

    def prepare_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer for the VLM parameters."""
        raise NotImplementedError("Implement prepare_optimizer() in recipes/minimal_vlm/engine/minimal_vlm_engine.py.")

    def prepare_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Build the learning-rate scheduler for training."""
        raise NotImplementedError("Implement prepare_scheduler() in recipes/minimal_vlm/engine/minimal_vlm_engine.py.")

    def train_pre_step(self, data: Any) -> Any:
        """Normalize a multimodal batch before the forward pass."""
        raise NotImplementedError("Implement train_pre_step() in recipes/minimal_vlm/engine/minimal_vlm_engine.py.")

    def train_one_step(self, data: Any) -> dict[str, Any]:
        """Run one VLM training step and return the loss plus scalar logs."""
        raise NotImplementedError("Implement train_one_step() in recipes/minimal_vlm/engine/minimal_vlm_engine.py.")
