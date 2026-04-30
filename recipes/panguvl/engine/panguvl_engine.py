"""Training engine for the PanguVL recipe."""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mvp_dataset import TorchLoader
from transformers.optimization import get_scheduler
from transformers.utils.logging import disable_progress_bar

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import get_data_parallel_world_size, is_main_process
from mvp_engine.engine import ENGINE_REGISTRY, Engine
from mvp_engine.utils.log import logger
from mvp_engine.utils.misc import calculate_model_size
from mvp_engine.utils.training import accumulate_gradients, clip_grad_norm_

from ..configs.schema import PanguvlConfig
from ..dataset import (
    ModelInputs,
    PanguvlCollator,
    build_dataset,
    build_qwen3_vl_processor,
    lightweight_process_sample,
)
from ..model import build_qwen3_vl_model
from ..model.packing import apply_packed_fa2_patch, prepare_packed_model_inputs
from ..utils.log.mfu import build_mfu_log
from ..utils.metrics import MetricAccumulator


@ENGINE_REGISTRY.register()
class PanguvlEngine(Engine):
    """Recipe-local engine for the PanguVL alignment stage."""

    ConfigClass = PanguvlConfig
    config: PanguvlConfig

    processor: Any | None = None

    def __init__(self, config):
        super().__init__(config)
        self.dp_world_size = get_data_parallel_world_size(self.device_mesh)
        disable_progress_bar()
        self.metric_accumulator = MetricAccumulator()
        self.metric_accumulator.register("global_total_token_count", "last")
        self.metric_accumulator.register("global_token_count", "last")
        self.metric_accumulator.register("local_token_count", "last")
        self.metric_accumulator.register("local_loss_sum", "sum")
        self.metric_accumulator.register("local_model_flops", "sum")

    def _resolve_batching_config(self) -> None:
        """Resolve PanguVL global batch size into micro batch size or accumulation."""

        global_batch_size = self.config.optim.global_batch_size
        gradient_accumulation_steps = int(self.config.optim.gradient_accumulation_steps)
        micro_batch_size = int(self.config.data.batch_size)

        if global_batch_size is None:
            if gradient_accumulation_steps == -1:
                raise ValueError("`optim.gradient_accumulation_steps=-1` requires `optim.global_batch_size`.")
            if micro_batch_size == -1:
                raise ValueError("`data.batch_size=-1` requires `optim.global_batch_size`.")
            return

        if micro_batch_size == -1 and gradient_accumulation_steps == -1:
            raise ValueError(
                "`optim.global_batch_size` cannot infer both `data.batch_size` and "
                "`optim.gradient_accumulation_steps` at the same time."
            )

        dp_world_size = self.dp_world_size

        if micro_batch_size == -1:
            divisor = dp_world_size * gradient_accumulation_steps
            if divisor <= 0 or global_batch_size % divisor != 0:
                raise ValueError(
                    "`data.batch_size` cannot be inferred exactly: "
                    "`optim.global_batch_size` must be divisible by "
                    "`data_parallel_world_size * optim.gradient_accumulation_steps`."
                )
            self.config.data.batch_size = global_batch_size // divisor
            micro_batch_size = int(self.config.data.batch_size)
        elif gradient_accumulation_steps == -1:
            divisor = dp_world_size * micro_batch_size
            if divisor <= 0 or global_batch_size % divisor != 0:
                raise ValueError(
                    "`optim.gradient_accumulation_steps` cannot be inferred exactly: "
                    "`optim.global_batch_size` must be divisible by "
                    "`data_parallel_world_size * data.batch_size`."
                )
            self.config.optim.gradient_accumulation_steps = global_batch_size // divisor
            gradient_accumulation_steps = int(self.config.optim.gradient_accumulation_steps)

        effective_global_batch_size = dp_world_size * micro_batch_size * gradient_accumulation_steps
        if effective_global_batch_size != global_batch_size:
            raise ValueError(
                "`optim.global_batch_size` does not match the configured batching: "
                f"expected {effective_global_batch_size} from "
                "`data_parallel_world_size * data.batch_size * optim.gradient_accumulation_steps`."
            )

    def prepare_dataloader(self, workflow: str = "train"):
        """Build the train-only dataloader over preprocessed multimodal samples.

        Args:
            workflow: Workflow name passed by the shared engine. Only ``train``
                is supported by this recipe.

        Returns:
            A ``TorchLoader`` pipeline that yields padded multimodal batches.
        """
        self._resolve_batching_config()

        # Step 1: build the shared processor once so both the temporary counting
        # loader and the real training loader use the exact same tokenizer setup.
        if self.processor is None:
            self.processor = build_qwen3_vl_processor(self.config.model)

        # Step 2: when total_steps=-1, run one finite lightweight data pass to
        # count packed samples and infer one-epoch optimization steps.
        if workflow == "train" and int(self.config.loop.total_steps) == -1:
            temp_config = self.config.model_copy(deep=True)

            # Step 2.1: disable cache for the temporary counting pass so step
            # inference does not spend time or disk on cache construction.
            if temp_config.data.cache:
                logger.info("PanguVL step inference: disabling cache for the temporary lightweight dataloader.")
                temp_config.data.cache = False

            # Step 2.2: build a temporary dataset/loader that uses the
            # lightweight preprocess but keeps the same packing behaviour.
            temp_collate_fn = PanguvlCollator(
                pad_token_id=int(self.processor.tokenizer.pad_token_id),
                processor=self.processor,
            )
            temp_dataset = build_dataset(
                temp_config,
                processor=self.processor,
                process_fn=lightweight_process_sample,
                resample=False,
            )
            temp_loader = TorchLoader(
                temp_dataset,
                num_workers=int(temp_config.data.num_workers),
                pin_memory=self.device.type in ["cuda", "npu"],
                persistent_workers=False,
                drop_last=False,
            ).batch(
                batch_size=int(temp_config.data.batch_size),
                drop_last=True,
                collate_fn=temp_collate_fn,
            )

            # Step 2.3: count how many packed samples this rank receives from
            # the temporary loader.
            count_start_time = time.perf_counter()
            last_log_time = count_start_time
            last_log_sample_count = 0
            local_sample_count = 0
            for batch_index, batch in enumerate(temp_loader, start=1):
                local_sample_count += int(batch["input_ids"].shape[0])
                now = time.perf_counter()
                if now - last_log_time >= 10.0:
                    interval_elapsed = now - last_log_time
                    interval_sample_count = local_sample_count - last_log_sample_count
                    logger.info(
                        f"Inferring training steps: {interval_sample_count / max(interval_elapsed, 1e-6):.2f} samples/s"
                    )
                    last_log_time = now
                    last_log_sample_count = local_sample_count

            # Step 2.4: reduce all local counts to get the real global packed
            # sample count across all ranks.
            total_sample_count = torch.tensor(local_sample_count, device=self.device, dtype=torch.long)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(total_sample_count, op=dist.ReduceOp.SUM)

            total_sample_count_value = int(total_sample_count.item())
            if total_sample_count_value <= 0:
                raise RuntimeError("PanguVL step inference found no packed training samples.")

            # Step 2.5: compute the real DP/FSDP size from the device mesh.
            # Tensor-parallel ranks do not consume extra samples per optimizer
            # step, so only the non-tensor mesh dimensions belong here.
            dp_world_size = self.dp_world_size

            # Step 2.6: infer one-epoch optimizer steps from the packed sample
            # count, per-rank batch size, and gradient accumulation. Round up
            # so tail samples extend training by a few extra steps instead of
            # being dropped by the step-count math.
            samples_per_optimization_step = (
                dp_world_size * int(self.config.data.batch_size) * int(self.config.optim.gradient_accumulation_steps)
            )
            inferred_total_steps = (
                total_sample_count_value + samples_per_optimization_step - 1
            ) // samples_per_optimization_step
            if inferred_total_steps <= 0:
                raise RuntimeError("Step inference found fewer packed samples than one optimization step requires.")

            extra_capacity = inferred_total_steps * samples_per_optimization_step - total_sample_count_value
            self.config.loop.total_steps = inferred_total_steps

            if is_main_process():
                logger.info(
                    f"Step inference: packed_samples={total_sample_count_value}, "
                    f"dp_world_size={dp_world_size}, "
                    f"inferred_total_steps={inferred_total_steps}, "
                    f"extra_capacity={extra_capacity}"
                )

        # Step 3: build the real training/eval dataset with the full preprocess.
        dataset = build_dataset(self.config, processor=self.processor)
        collate_fn = PanguvlCollator(
            pad_token_id=int(self.processor.tokenizer.pad_token_id),
            processor=self.processor,
        )

        # Step 4: wrap the dataset in the normal TorchLoader and return the
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
        """Build the recipe model and wrap it for distributed training.

        Args:
            None.

        Returns:
            The distributed-ready Qwen3-VL model.
        """
        using_fsdp2 = self.device_mesh["shard"].size() * self.device_mesh["tensor"].size() > 1
        model_config = self.config.model.model_copy(deep=True)
        if using_fsdp2 and bool(model_config.gradient_checkpointing.enabled):
            model_config.gradient_checkpointing.use_reentrant = True
            logger.info("Using model-side gradient checkpointing with use_reentrant=True under FSDP2.")

        model = build_qwen3_vl_model(model_config).to(self.device)
        if self.config.data.packing and getattr(model.config, "_attn_implementation", None) == "flash_attention_2":
            apply_packed_fa2_patch()
        logger.info(f"Model name: {model.__class__.__name__}")

        if self.config.model.compile:
            # The ViT encoder computes data-dependent cu_seqlens/max_seqlen for
            # flash_attn_varlen_func directly (not via _get_unpad_data), so dynamo cannot
            # trace it — flash_attn C++ requires a Python int but receives a FakeTensor.
            # Wrapping its forward with compiler.disable causes a graph break there so it
            # runs eagerly.
            if hasattr(model, "model") and hasattr(model.model, "visual"):
                model.model.visual.forward = torch.compiler.disable(model.model.visual.forward)
            model.compile(
                backend=self.config.model.compile_backend,
                mode=self.config.model.compile_mode,
            )

        parallelized_model = parallelize_model(
            model,
            device_mesh=self.device_mesh,
            backend_kwargs=self.config.parallel.backend_kwargs.model_dump(),
        )

        if is_main_process():
            model_size, trainable_size = calculate_model_size(parallelized_model)
            logger.info(f" - Model size: {model_size / 1e9:.4f} B")
            logger.info(f" - Trainable model size: {trainable_size / 1e9:.4f} B")

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
            raise ValueError("No trainable parameters found for the PanguVL recipe.")

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
            The HF cosine scheduler aligned with LF-private.
        """
        warmup_steps = math.ceil(self.total_steps * float(self.config.optim.warmup_ratio))
        return get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

    def run_iter_train(self) -> None:
        """Run iteration-based training with window-prefetched token normalization."""

        def _count_total_tokens(attention_mask: torch.Tensor) -> int:
            """Count all non-padding tokens in one collated micro-batch."""
            return int(attention_mask.sum().item())

        def _count_valid_loss_tokens(labels: torch.Tensor, *, ignore_index: int = -100) -> int:
            """Count the valid shifted labels used by causal-LM cross entropy."""
            shifted_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:]
            return int((shifted_labels != ignore_index).sum().item())

        train_iterator = iter(self.train_loader)
        gradient_accumulation_steps = int(self.config.optim.gradient_accumulation_steps)

        while self.step < self.total_steps:
            remaining_micro_steps = gradient_accumulation_steps - int(self._accumulate_step)
            if remaining_micro_steps <= 0:
                remaining_micro_steps = gradient_accumulation_steps

            local_total_token_count = 0
            local_token_count = 0
            micro_batches: list[ModelInputs] = []
            for _ in range(remaining_micro_steps):
                try:
                    data = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(self.train_loader)
                    try:
                        data = next(train_iterator)
                    except StopIteration as exc:
                        raise RuntimeError("PanguVL train loader did not yield any batches.") from exc
                local_total_token_count += _count_total_tokens(data["attention_mask"])
                effective_token_num = _count_valid_loss_tokens(data["labels"])
                data["effective_token_num"] = effective_token_num
                local_token_count += effective_token_num

                micro_batches.append(data)

            token_counts = torch.tensor(
                [local_total_token_count, local_token_count],
                device=self.device,
                dtype=torch.long,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(token_counts, op=dist.ReduceOp.SUM)

            self.metric_accumulator.reset()
            self.metric_accumulator.update(
                global_total_token_count=int(token_counts[0].item()),
                global_token_count=int(token_counts[1].item()),
                local_token_count=local_token_count,
            )

            global_loss_token_count = self.metric_accumulator.get("global_token_count")
            if global_loss_token_count is None or int(global_loss_token_count) <= 0:
                raise ValueError("Accumulation window must contain at least one supervised token.")

            for data in micro_batches:
                self.train_after_step(self.train_one_step(self.train_pre_step(data)))

    def train_pre_step(self, data: ModelInputs) -> ModelInputs:
        """Move the collated batch to the local device and normalize keys.

        Args:
            data: Raw batch emitted by the dataloader.

        Returns:
            A normalized batch dictionary ready for the model forward pass.
        """
        batch: ModelInputs = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=True)
            else:
                batch[key] = value

        batch = prepare_packed_model_inputs(
            batch,
            model_config=self.unwrapped_model.config,
            attn_implementation=getattr(self.unwrapped_model.config, "_attn_implementation", None),
            mask_dtype=self.dtype if self.dtype.is_floating_point else torch.float32,
        )

        def _cast_visual_inputs(batch: dict[str, Any]) -> dict[str, Any]:
            """Match LF-private by casting visual tensors to the active mixed-precision dtype."""
            if self.dtype == torch.float32:
                return batch

            for key in ("pixel_values", "pixel_values_videos"):
                value = batch.get(key)
                if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                    batch[key] = value.to(dtype=self.dtype)
            return batch

        batch = _cast_visual_inputs(batch)
        return batch

    def train_one_step(self, data: ModelInputs) -> dict[str, Any]:
        """Run one forward pass and collect training metrics.

        Args:
            data: Normalized multimodal batch on the local device.

        Returns:
            A dictionary containing the loss tensor and logging scalars.
        """
        with torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        ):
            outputs = self.model(**data)

        return {
            "loss": outputs.loss,
            "logs": {
                "train/loss": 0.0,
            },
            "effective_tokens": data["effective_token_num"],
            "model_flops": float(
                self.unwrapped_model.calculate_model_flops(
                    batch_size=int(data["input_ids"].shape[0]),
                    seq_len=int(data["input_ids"].shape[1]),
                    image_grid_thw=data.get("image_grid_thw"),
                    is_training=True,
                    freeze_vit=bool(self.config.model.freeze_vit),
                    freeze_merger=bool(self.config.model.freeze_merger),
                    freeze_llm=bool(self.config.model.freeze_llm),
                )
            ),
        }

    def train_after_step(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Run the optimizer step and include recipe-local MFU metrics in logs."""
        assert "loss" in outputs, "The model output must contain 'loss' key."
        assert "logs" in outputs, "The model output must contain 'logs' key."

        is_sync = self.accumulate_step()
        global_total_token_count = self.metric_accumulator.get("global_total_token_count")
        if global_total_token_count is None or int(global_total_token_count) <= 0:
            raise ValueError("PanguVL accumulation window is missing a valid global total token count.")
        global_loss_token_count = self.metric_accumulator.get("global_token_count")
        if global_loss_token_count is None or int(global_loss_token_count) <= 0:
            raise ValueError("PanguVL accumulation window is missing a valid global token count.")
        local_token_count = self.metric_accumulator.get("local_token_count")
        if local_token_count is None or int(local_token_count) <= 0:
            raise ValueError("PanguVL accumulation window is missing a valid local token count.")

        self.metric_accumulator.update(
            local_loss_sum=outputs["loss"].detach().to(device=self.device, dtype=torch.float64).sum(),
            local_model_flops=float(outputs["model_flops"]),
        )

        # DDP/FSDP averages gradients across the DP/FSDP ranks, so each
        # micro-step uses the local summed loss with the globally reduced token
        # denominator and compensates by the real DP world size.
        loss = outputs["loss"].sum() / int(global_loss_token_count)
        if self.dp_world_size > 1:
            loss = loss * float(self.dp_world_size)

        with accumulate_gradients(self.model, sync=is_sync):
            self.scaler.scale(loss).backward()

        if not is_sync:
            return outputs

        self.scaler.unscale_(self.optimizer)

        max_grad_norm = self.config.optim.clip_grad_norm
        if max_grad_norm is not None:
            clip_grad_norm_(self.model, max_grad_norm)

        step_lrs = [float(param_group["lr"]) for param_group in self.optimizer.param_groups]

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.step += 1
        self.timer.tick()

        global_loss_sum = self.metric_accumulator.get("local_loss_sum")
        if global_loss_sum is None:
            raise ValueError("PanguVL accumulation window is missing a valid local loss sum.")
        if not isinstance(global_loss_sum, torch.Tensor):
            global_loss_sum = torch.tensor(global_loss_sum, device=self.device, dtype=torch.float64)
        else:
            global_loss_sum = global_loss_sum.detach().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)

        accumulated_metrics = self.metric_accumulator.finalize()
        other_logs = {
            "eta": self.timer.eta_string,
            "perf/batch_time": self.timer.batch_time,
            "perf/toks_per_sec": int(global_loss_token_count) / self.timer.batch_time,
            "tokens/global_total": int(global_total_token_count),
            "tokens/global_loss": int(global_loss_token_count),
        }
        other_logs.update(
            build_mfu_log(
                model_flops_per_step=float(accumulated_metrics["local_model_flops"]),
                device_type=self.device.type,
                precision=str(self.config.optim.mixed_precision),
                step_time_seconds=float(self.timer.batch_time_latest),
            )
        )

        for i, lr in enumerate(step_lrs):
            other_logs[f"lr/group_{i}"] = lr

        outputs["logs"]["train/loss"] = float(global_loss_sum / int(global_loss_token_count))
        logger.log_metrics(
            {**outputs["logs"], **other_logs},
            step=self.step,
        )

        self.metric_accumulator.reset()

        self.save()

        return outputs
