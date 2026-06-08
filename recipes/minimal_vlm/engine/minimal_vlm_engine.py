"""Training engine for the minimal VLM recipe."""

from __future__ import annotations

import math
from typing import Any

import torch
from mvp_dataset import TorchLoader
from transformers import get_scheduler

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.engine import ENGINE_REGISTRY, Engine, TrainStepContext
from mvp_engine.kit import CPKit
from mvp_engine.utils.log import logger

from ..configs.schema import MinimalVLMConfig
from ..dataset.collator import MinimalVLMCollator
from ..dataset.dataset import build_dataset
from ..dataset.processor import build_qwen3_vl_processor
from ..dataset.types import ModelInputs
from ..model.qwen3_vl import (
    build_qwen3_vl_model,
    install_qwen3_vl_tensor_parallel_grad_sync,
    prepare_qwen3_vl_mrope_position_ids,
)


@ENGINE_REGISTRY.register()
class MinimalVLMEngine(Engine):
    """Recipe-local engine for supervised Qwen3-VL fine-tuning."""

    ConfigClass = MinimalVLMConfig
    config: MinimalVLMConfig

    processor: Any | None = None

    def __init__(self, config: MinimalVLMConfig):
        """Initialize recipe-local reusable kits."""
        super().__init__(config)
        self.cp_kit = CPKit()

    def prepare_dataloader(self, workflow: str = "train"):
        """Build the train-only dataloader over preprocessed multimodal samples.

        Args:
            workflow: Workflow name passed by the shared engine. Only ``train``
                is supported by this recipe.

        Returns:
            A ``TorchLoader`` pipeline that yields padded multimodal batches.
        """
        if workflow != "train":
            logger.warning(f"Minimal VLM engine does not support workflow '{workflow}'.")
            return

        # Step 1: build the processor
        if self.processor is None:
            self.processor = build_qwen3_vl_processor(self.config.model)

        # Step 2: build the real training dataset with the full preprocess.
        dataset = build_dataset(self.config, processor=self.processor, device_mesh=self.device_mesh)
        collate_fn = MinimalVLMCollator(
            pad_token_id=int(self.processor.tokenizer.pad_token_id),
        )

        # Step 3: wrap the dataset in the normal TorchLoader and return the
        # batched dataloader used by the engine.
        loader = TorchLoader(
            dataset,
            num_workers=int(self.config.data.num_workers),
            pin_memory=self.device.type in ["cuda", "npu"],
            persistent_workers=False,
        )
        return loader.batch(
            batch_size=int(self.config.data.batch_size),
            drop_last=True,
            collate_fn=collate_fn,
        )

    def prepare_model(self) -> torch.nn.Module:
        """Build, patch, and parallelize the recipe model.

        Args:
            None.

        Returns:
            The distributed-ready Qwen3-VL model.
        """
        model = build_qwen3_vl_model(self.config.model).to(self.device)
        logger.info(f"Model name: {model.__class__.__name__}")

        parallelized_model = parallelize_model(
            model,
            device_mesh=self.device_mesh,
            backend_kwargs=self.config.parallel.backend_kwargs.model_dump(),
        )
        install_qwen3_vl_tensor_parallel_grad_sync(parallelized_model, self.device_mesh)

        return parallelized_model

    def prepare_optimizer(self) -> torch.optim.Optimizer:
        """Construct AdamW over the subset of trainable model parameters.

        Args:
            None.

        Returns:
            The optimizer used by this recipe.
        """
        trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise ValueError("No trainable parameters found for the minimal VLM recipe.")

        return torch.optim.AdamW(
            trainable_parameters,
            lr=float(self.config.optim.lr),
            weight_decay=float(self.config.optim.weight_decay),
        )

    def prepare_scheduler(self):
        """Construct the learning-rate schedule used by this recipe.

        Args:
            None.

        Returns:
            The warmup-plus-cosine scheduler for the current optimizer.
        """
        warmup_steps = math.ceil(self.total_steps * float(self.config.optim.warmup_ratio))
        return get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

    def train_pre_step(self, ctx: TrainStepContext) -> TrainStepContext:
        """Prepare one micro-batch for forward."""
        data: ModelInputs = ctx.data
        batch: ModelInputs = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=True)
            else:
                batch[key] = value
        if self.config.parallel.backend_kwargs.long_context.enabled:
            batch = self._prepare_long_context_batch(batch)
        ctx.data = batch
        return ctx

    def _prepare_long_context_batch(self, batch: ModelInputs) -> ModelInputs:
        """Shard token-aligned tensors across the context mesh."""
        long_context_config = self.config.parallel.backend_kwargs.long_context.model_dump()
        prepare_qwen3_vl_mrope_position_ids(self.unwrapped_model, batch)
        prepared = self.cp_kit.prepare_causal_batch(
            batch,
            device_mesh=self.device_mesh,
            config=long_context_config,
            pad_token_id=int(self.processor.tokenizer.pad_token_id),
            ignore_index=-100,
        )
        local_batch = prepared.local_batch
        self._slice_long_context_vision_inputs(
            full_batch=prepared.global_batch,
            local_batch=local_batch,
            local_position_indices=prepared.layout.local_position_indices,
        )
        self._long_context_global_seq_len = prepared.layout.global_seq_len
        self._long_context_local_seq_len = prepared.layout.local_seq_len
        self._long_context_local_position_indices = prepared.layout.local_position_indices
        return local_batch

    def _slice_long_context_vision_inputs(
        self,
        *,
        full_batch: ModelInputs,
        local_batch: ModelInputs,
        local_position_indices: torch.Tensor,
    ) -> None:
        """Keep only vision tensors whose placeholder tokens are fully local."""
        if "pixel_values" not in full_batch or "image_grid_thw" not in full_batch:
            return

        model_config = self.unwrapped_model.config
        image_token_id = int(model_config.image_token_id)
        spatial_merge_size = int(model_config.vision_config.spatial_merge_size)

        image_grid_thw = full_batch["image_grid_thw"]
        language_token_counts = (image_grid_thw.prod(dim=-1) // (spatial_merge_size**2)).tolist()
        pixel_token_counts = image_grid_thw.prod(dim=-1).tolist()
        pixel_chunks = list(
            torch.split(full_batch["pixel_values"], [int(count) for count in pixel_token_counts], dim=0)
        )

        position_to_local_index = {
            int(position): local_index for local_index, position in enumerate(local_position_indices.tolist())
        }
        selected_images: list[tuple[int, int]] = []
        image_index = 0
        for row in full_batch["input_ids"]:
            image_positions = torch.nonzero(row == image_token_id, as_tuple=False).flatten().tolist()
            for image_start, image_end in _iter_contiguous_ranges(image_positions):
                if image_index >= len(language_token_counts):
                    raise ValueError("Qwen3-VL image placeholders exceed image_grid_thw entries.")
                expected_tokens = int(language_token_counts[image_index])
                if image_end - image_start != expected_tokens:
                    raise ValueError("Qwen3-VL image placeholder span does not match image_grid_thw token count.")

                local_indices = [position_to_local_index.get(position) for position in range(image_start, image_end)]
                local_token_count = sum(index is not None for index in local_indices)
                if 0 < local_token_count < expected_tokens:
                    raise ValueError(
                        "Minimal VLM long-context cannot split one image placeholder span across context ranks."
                    )
                if local_token_count == expected_tokens:
                    sorted_local_indices = sorted(int(index) for index in local_indices if index is not None)
                    expected_local_indices = list(
                        range(sorted_local_indices[0], sorted_local_indices[0] + expected_tokens)
                    )
                    if sorted_local_indices != expected_local_indices:
                        raise ValueError(
                            "Minimal VLM long-context requires local image placeholder spans to stay contiguous."
                        )
                    selected_images.append((sorted_local_indices[0], image_index))
                image_index += 1

        if image_index != len(language_token_counts):
            raise ValueError("Qwen3-VL image_grid_thw entries exceed image placeholders.")

        if not selected_images:
            local_batch.pop("pixel_values", None)
            local_batch.pop("image_grid_thw", None)
            return

        selected_image_indices = [image_index for _, image_index in sorted(selected_images)]
        local_batch["image_grid_thw"] = torch.stack([image_grid_thw[index] for index in selected_image_indices], dim=0)
        local_batch["pixel_values"] = torch.cat([pixel_chunks[index] for index in selected_image_indices], dim=0)

        local_image_tokens = int((local_batch["input_ids"] == image_token_id).sum().item())
        expected_local_tokens = sum(int(language_token_counts[index]) for index in selected_image_indices)
        if local_image_tokens != expected_local_tokens:
            raise ValueError("Local Qwen3-VL image tokens do not match selected image features.")

    def forward_step(self, ctx: TrainStepContext) -> None:
        """Run one forward pass and collect training metrics.

        Args:
            ctx: Current training step context.
        """
        data: ModelInputs = ctx.data
        long_context_enabled = self.config.parallel.backend_kwargs.long_context.enabled
        model_inputs = dict(data)
        labels = model_inputs.pop("labels") if long_context_enabled else None
        if long_context_enabled:
            model_inputs["use_cache"] = False
        with torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        ):
            outputs = self.model(**model_inputs) if long_context_enabled else self.model(**data)

        loss = self._compute_long_context_loss(outputs.logits, labels) if long_context_enabled else outputs.loss
        log_loss = float(loss.item())
        if long_context_enabled:
            loss_stats = self._last_long_context_loss_stats
            log_loss = float((loss_stats.global_loss_sum / loss_stats.global_valid_tokens).item())

        ctx.outputs = {
            "loss": loss,
            "logs": {
                "train/loss": log_loss,
            },
        }

    def _compute_long_context_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute global-token-normalized causal LM loss across context ranks."""
        loss_stats = self.cp_kit.compute_cross_entropy_loss(
            logits,
            labels,
            device_mesh=self.device_mesh,
            ignore_index=-100,
        )
        self._last_long_context_loss_stats = loss_stats
        return loss_stats.loss


def _iter_contiguous_ranges(values: list[int]):
    """Yield half-open contiguous ranges from sorted integer values."""
    if not values:
        return
    start = values[0]
    previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        yield start, previous + 1
        start = value
        previous = value
    yield start, previous + 1
