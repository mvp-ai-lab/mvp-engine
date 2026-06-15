"""Training engine for the PanguVL recipe."""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mvp_dataset import TorchLoader

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import (
    get_data_parallel_group,
    get_data_parallel_world_size,
    is_main_process,
)
from mvp_engine.engine import ENGINE_REGISTRY, Engine, TrainStepContext
from mvp_engine.kit import (
    MFUKit,
    MLLMDataKit,
    MLLMModelKit,
    ModelInputs,
    OptimKit,
    TokenNormedLossKit,
)
from mvp_engine.utils.log import logger
from mvp_engine.utils.training import accumulate_gradients, clip_grad_norm_

from ..configs.schema import PanguvlConfig
from ..dataset import (
    PanguvlCollator,
    build_dataset,
    build_qwen3_vl_processor,
    build_step_count_dataset,
)
from ..model import build_qwen3_vl_model
from ..model.packing import apply_packed_fa2_patch, prepare_packed_model_inputs

torch.set_float32_matmul_precision("high")


@ENGINE_REGISTRY.register()
class PanguvlEngine(Engine):
    """Recipe-local engine for the PanguVL alignment stage."""

    ConfigClass = PanguvlConfig
    config: PanguvlConfig

    processor: Any | None = None
    data_kit: MLLMDataKit
    model_kit: MLLMModelKit
    mfu_kit: MFUKit
    optim_kit: OptimKit
    token_loss_kit: TokenNormedLossKit

    def __init__(self, config):
        super().__init__(config)
        self.dp_world_size = get_data_parallel_world_size(self.device_mesh)
        self.dp_group = get_data_parallel_group(self.device_mesh)
        self.config.resolve_batching_config(data_parallel_world_size=self.dp_world_size)
        self.data_kit = MLLMDataKit()
        self.model_kit = MLLMModelKit()
        self.mfu_kit = MFUKit()
        self.optim_kit = OptimKit()
        self.token_loss_kit = TokenNormedLossKit(
            device=self.device,
            dp_world_size=self.dp_world_size,
            dp_group=self.dp_group,
        )
        self.token_loss_kit.build_loss_guard(
            spike_multiplier=self.config.optim.loss_spike_skip_multiplier,
            window_size=int(self.config.optim.loss_spike_skip_window_size),
            min_history=int(self.config.optim.loss_spike_skip_min_history),
        )

    def prepare_dataloader(self, workflow: str = "train"):
        """Build the train-only dataloader over preprocessed multimodal samples.

        Args:
            workflow: Workflow name passed by the shared engine. Only ``train``
                is supported by this recipe.

        Returns:
            A ``TorchLoader`` pipeline that yields padded multimodal batches.
        """
        if workflow != "train":
            logger.warning(f"PanguVL engine does not support workflow '{workflow}'.")
            return

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

            # Step 2.2: build a temporary dataset/loader that uses length-only
            # preprocessing and avoids reading heavy image bytes.
            temp_dataset = build_step_count_dataset(temp_config, processor=self.processor)
            temp_loader = TorchLoader(
                temp_dataset,
                num_workers=int(temp_config.data.num_workers),
                pin_memory=self.device.type in ["cuda", "npu"],
                persistent_workers=False,
                drop_last=False,
            )

            # Step 2.3: count how many packed samples this rank receives from
            # the temporary loader.
            count_start_time = time.perf_counter()
            last_log_time = count_start_time
            last_log_sample_count = 0
            local_sample_count = 0
            for sample in temp_loader:
                if "sample_count" in sample:
                    local_sample_count += 1
                elif "input_ids" in sample and int(sample["input_ids"].numel()) > 0:
                    local_sample_count += 1
                now = time.perf_counter()
                if now - last_log_time >= 10.0:
                    interval_elapsed = now - last_log_time
                    interval_sample_count = local_sample_count - last_log_sample_count
                    logger.info(
                        f"Inferring training steps: {interval_sample_count / max(interval_elapsed, 1e-6):.2f} samples/s"
                    )
                    last_log_time = now
                    last_log_sample_count = local_sample_count

            batch_size = int(temp_config.data.batch_size)
            local_sample_count = (local_sample_count // batch_size) * batch_size

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
        if (
            self.config.data.packing
            and getattr(model.config, "_attn_implementation", None) == "flash_attention_2"
            and self.device.type != "npu"
        ):
            apply_packed_fa2_patch()
        logger.info(f"Model name: {model.__class__.__name__}")

        if self.config.model.compile.enabled:
            model = self.model_kit.apply_model_compile(
                model,
                backend=self.config.model.compile.backend,
                mode=self.config.model.compile.mode,
            )

        parallelized_model = parallelize_model(
            model,
            device_mesh=self.device_mesh,
            backend_kwargs=self.config.parallel.backend_kwargs.model_dump(),
        )

        return parallelized_model

    def prepare_optimizer(self) -> torch.optim.Optimizer:
        """Construct AdamW over the subset of trainable model parameters.

        Args:
            None.

        Returns:
            The optimizer used by this recipe.
        """
        return self.optim_kit.build_optimizer(
            self.model,
            optimizer=self.config.optim.optimizer,
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
        return self.optim_kit.build_lr_scheduler(
            optimizer=self.optimizer,
            lr_scheduler="cosine",
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

    def train_pre_step(self, ctx: TrainStepContext) -> TrainStepContext:
        """Prepare one micro-batch for forward."""
        data: ModelInputs = ctx.data
        data["total_tokens"] = int(data["attention_mask"].sum().item())
        shifted_labels = F.pad(data["labels"], (0, 1), value=-100)[..., 1:]
        data["effective_tokens"] = int((shifted_labels != -100).sum().item())

        batch: ModelInputs = self.data_kit.to_device(data, self.device)

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
        ctx.data = batch
        return ctx

    def forward_step(self, ctx: TrainStepContext) -> None:
        """Run one forward pass and collect micro-batch training metrics."""
        data: ModelInputs = ctx.data
        with torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        ):
            outputs = self.model(**data)

        ctx.outputs = {
            "loss": outputs.loss,
            "logs": {
                "train/loss": 0.0,
            },
        }
        self.mfu_kit.accumulate_microbatch(
            model=self.unwrapped_model,
            batch_size=int(data["input_ids"].shape[0]),
            seq_len=int(data["input_ids"].shape[1]),
            image_grid_thw=data.get("image_grid_thw"),
            is_training=True,
            freeze_vit=bool(self.config.model.freeze_vit),
            freeze_merger=bool(self.config.model.freeze_merger),
            freeze_llm=bool(self.config.model.freeze_llm),
        )

    def backward_step(self, ctx: TrainStepContext) -> None:
        """Backward one micro-batch with delayed global token normalization."""
        outputs = ctx.outputs
        assert outputs is not None, "The forward step must populate ctx.outputs."
        assert "loss" in outputs, "The model output must contain 'loss' key."
        assert "logs" in outputs, "The model output must contain 'logs' key."

        ctx.should_sync = self.ga_state.advance()

        local_micro_loss_sum = outputs["loss"].sum()
        micro_effective_token_count = int(ctx.data["effective_tokens"])
        micro_total_token_count = int(ctx.data["total_tokens"])

        backward_loss_divisor = (
            int(self.config.data.batch_size)
            * int(self.config.data.max_seq_len)
            * int(self.config.optim.gradient_accumulation_steps)
        )

        if not self.token_loss_kit.guard_loss(
            local_micro_loss_sum,
            micro_effective_token_count,
            step=int(self.step),
        ):
            local_micro_loss_sum = local_micro_loss_sum * 0.0

        ctx.loss = self.token_loss_kit.accumulate_microbatch(
            loss_sum=local_micro_loss_sum,
            effective_tokens=micro_effective_token_count,
            total_tokens=micro_total_token_count,
            backward_divisor=backward_loss_divisor,
        )
        with accumulate_gradients(self.model, sync=ctx.should_sync):
            self.scaler.scale(ctx.loss).backward()

    def optimizer_step(self, ctx: TrainStepContext) -> None:
        """Scale accumulated gradients by global tokens and apply the optimizer step."""
        if not ctx.should_sync:
            return

        token_loss_stats = self.token_loss_kit.reduce_window()

        self.scaler.unscale_(self.optimizer)
        self.token_loss_kit.rescale_gradients(self.model.parameters(), token_loss_stats)

        max_grad_norm = self.config.optim.clip_grad_norm
        if max_grad_norm is not None:
            clip_grad_norm_(self.model, max_grad_norm)

        step_lrs = [float(param_group["lr"]) for param_group in self.optimizer.param_groups]

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        outputs = ctx.outputs
        assert outputs is not None, "The forward step must populate ctx.outputs."
        ctx.outputs["global_total_token_count"] = token_loss_stats.global_total_tokens
        ctx.outputs["global_effective_token_count"] = token_loss_stats.global_effective_tokens
        ctx.outputs["global_loss_sum"] = token_loss_stats.global_loss_sum
        ctx.outputs["step_lrs"] = step_lrs
        ctx.optimizer_step_completed = True
        self.token_loss_kit.reset()

    def train_post_step(self, ctx: TrainStepContext) -> None:
        """Log one synchronized optimizer step and save checkpoints."""
        outputs = ctx.outputs
        assert outputs is not None, "The forward step must populate ctx.outputs."

        step_time_seconds = float(self.timer.progress_time_latest)
        global_total_token_count = int(outputs["global_total_token_count"])
        global_effective_token_count = int(outputs["global_effective_token_count"])
        global_loss_sum = outputs["global_loss_sum"]

        outputs["logs"]["train/loss"] = float(global_loss_sum / int(global_effective_token_count))
        outputs["logs"].update(
            {
                "eta": self.timer.eta_string,
                "perf/batch_time": step_time_seconds,
                "perf/data_time": self.timer.get_scope_time("data_time"),
                "perf/exec_time": self.timer.get_scope_time("exec_time"),
                "perf/toks_per_sec": global_total_token_count / step_time_seconds if step_time_seconds > 0 else 1e-8,
                "tokens/total": global_total_token_count,
                "tokens/effective": global_effective_token_count,
            }
        )
        outputs["logs"].update(
            self.mfu_kit.build_log(
                device_type=self.device.type,
                precision=str(self.config.optim.mixed_precision),
                step_time_seconds=step_time_seconds,
            )
        )

        for i, lr in enumerate(outputs["step_lrs"]):
            outputs["logs"][f"lr/group_{i}"] = lr

        logger.log_metrics(
            outputs["logs"],
            step=self.step,
            total_steps=self.total_steps,
        )
