"""Training engine for the Basic VLM recipe."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from mvp_dataset import TorchLoader
from transformers import ProcessorMixin
from transformers.optimization import get_scheduler
from transformers.utils.logging import disable_progress_bar

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import get_data_parallel_world_size
from mvp_engine.engine import ENGINE_REGISTRY, Engine, TrainStepContext
from mvp_engine.utils.log import logger
from mvp_engine.utils.metrics import DistributedMetricAccumulator
from mvp_engine.utils.training import accumulate_gradients, clip_grad_norm_

from ..configs.schema import BasicVLMConfig
from ..dataset.collator import BasicVLMCollator
from ..dataset.dataset import build_dataset
from ..dataset.processor import build_qwen3_vl_processor
from ..dataset.types import ModelInputs
from ..guards.loss import PerTokenLossGuard
from ..model import build_qwen3_vl_model
from ..model.packing import apply_packed_fa2_patch, prepare_packed_model_inputs
from ..utils.log.mfu import build_mfu_log
from ..utils.misc import infer_total_steps, resolve_batching_config


@ENGINE_REGISTRY.register()
class BasicVLMEngine(Engine):
    """Recipe-local engine for the Basic VLM."""

    ConfigClass = BasicVLMConfig
    config: BasicVLMConfig

    processor: ProcessorMixin | None = None
    loss_guard: PerTokenLossGuard
    metric_accumulator: DistributedMetricAccumulator

    def __init__(self, config):
        """Initialize Basic VLM-local distributed state and metric reducers."""
        super().__init__(config)
        self.dp_world_size = get_data_parallel_world_size(self.device_mesh)
        disable_progress_bar()
        self.metric_accumulator = DistributedMetricAccumulator(device=self.device)
        self.metric_accumulator.register("total_token_count", accumulate="sum", reduce="sum")
        self.metric_accumulator.register("effective_token_count", accumulate="sum", reduce="sum")
        self.metric_accumulator.register("loss_sum", accumulate="sum", reduce="sum")
        self.metric_accumulator.register("model_flops", accumulate="sum", reduce="none")

    def prepare_dataloader(self, workflow: str = "train"):
        """Build the train dataloader over preprocessed multimodal samples.

        Args:
            workflow: Workflow name passed by the shared engine. Only ``train``
                is supported by this recipe.

        Returns:
            A ``TorchLoader`` pipeline that yields padded multimodal batches,
            or ``None`` for unsupported non-training workflows.
        """
        if workflow != "train":
            logger.warning(f"Basic VLM engine does not support workflow '{workflow}'.")
            return

        resolve_batching_config(self.config, data_parallel_world_size=self.dp_world_size)

        # Step 1: build the shared processor once so both the temporary counting
        # loader and the real training loader use the exact same tokenizer setup.
        if self.processor is None:
            self.processor = build_qwen3_vl_processor(self.config.model)

        # Step 2: when total_steps=-1, run one finite lightweight data pass to
        # count packed samples and infer one-epoch optimization steps.
        if int(self.config.loop.total_steps) == -1:
            self.config.loop.total_steps = infer_total_steps(
                self.config,
                processor=self.processor,
                device=self.device,
                data_parallel_world_size=self.dp_world_size,
            )

        # Step 3: build the real training/eval dataset with the full preprocess.
        dataset = build_dataset(self.config, processor=self.processor)
        collate_fn = BasicVLMCollator(
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
            multiprocessing_context="spawn",
        )
        return loader.batch(
            batch_size=int(self.config.data.batch_size),
            drop_last=True,
            collate_fn=collate_fn,
        )

    def prepare_model(self) -> torch.nn.Module:
        """Build, patch, compile, and parallelize the recipe model.

        Args:
            None.

        Returns:
            The distributed-ready Qwen3-VL model.
        """
        model = build_qwen3_vl_model(self.config.model).to(self.device)
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

        return parallelized_model

    def prepare_optimizer(self) -> torch.optim.Optimizer:
        """Construct AdamW and initialize the per-token loss guard.

        Args:
            None.

        Returns:
            The optimizer used by this recipe.
        """
        self.loss_guard = PerTokenLossGuard(
            spike_multiplier=self.config.optim.loss_spike_skip_multiplier,
            window_size=int(self.config.optim.loss_spike_skip_window_size),
            min_history=int(self.config.optim.loss_spike_skip_min_history),
        )

        trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise ValueError("No trainable parameters found for the Basic VLM recipe.")

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
            The Hugging Face cosine scheduler used by this recipe.
        """
        warmup_steps = math.ceil(self.total_steps * float(self.config.optim.warmup_ratio))
        return get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

    def train_pre_step(self, ctx: TrainStepContext) -> TrainStepContext:
        """Prepare one micro-batch for forward.

        This counts total and shifted-label supervised tokens before moving
        tensors to the local device, prepares packed attention inputs, and casts
        visual tensors to the active mixed-precision dtype.
        """
        data: ModelInputs = ctx.data
        data["total_token_num"] = int(data["attention_mask"].sum().item())
        shifted_labels = F.pad(data["labels"], (0, 1), value=-100)[..., 1:]
        data["effective_token_num"] = int((shifted_labels != -100).sum().item())

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

        for key in ("pixel_values", "pixel_values_videos"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                batch[key] = value.to(dtype=self.dtype)
        ctx.data = batch
        return ctx

    def forward_step(self, ctx: TrainStepContext) -> None:
        """Run one forward pass and collect micro-batch training metrics.

        The patched model returns unreduced per-token CE loss. Token counts and
        local FLOPs are carried into later hooks for accumulation-window
        reduction and logging.
        """
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
            "effective_tokens": data["effective_token_num"],
            "total_tokens": data["total_token_num"],
            "model_flops": float(
                self.unwrapped_model.calculate_model_flops(
                    batch_size=int(data["input_ids"].shape[0]),
                    seq_len=int(data["input_ids"].shape[1]),
                    attention_mask=data.get("attention_mask"),
                    image_grid_thw=data.get("image_grid_thw"),
                    is_training=True,
                    freeze_vit=bool(self.config.model.freeze_vit),
                    freeze_merger=bool(self.config.model.freeze_merger),
                    freeze_llm=bool(self.config.model.freeze_llm),
                )
            ),
        }

    def backward_step(self, ctx: TrainStepContext) -> None:
        """Backward one micro-batch with delayed global token normalization.

        Per-token loss is backpropagated immediately with a fixed provisional
        divisor so data loading can overlap with GPU work. At the sync micro
        step, accumulated gradients are rescaled by the reduced global supervised token count.
        """
        outputs = ctx.outputs
        assert outputs is not None, "The forward step must populate ctx.outputs."
        assert "loss" in outputs, "The model output must contain 'loss' key."
        assert "logs" in outputs, "The model output must contain 'logs' key."

        ctx.should_sync = self.ga_state.advance()

        # 1. Read this micro-step's local token/loss/flops.
        local_micro_loss_sum = outputs["loss"].sum()
        micro_effective_token_count = int(outputs["effective_tokens"])
        micro_total_token_count = int(outputs["total_tokens"])

        # 2. Backward immediately so the dataloader can prepare later micro-batches
        # while GPU work is active. The exact per-token denominator is applied to
        # accumulated gradients once the full accumulation window token count is known.
        backward_loss_divisor = (
            int(self.config.data.batch_size)
            * int(self.config.data.max_seq_len)
            * int(self.config.optim.gradient_accumulation_steps)
        )
        loss = local_micro_loss_sum / float(backward_loss_divisor)

        if self.loss_guard.check(
            local_micro_loss_sum,
            micro_effective_token_count,
            step=int(self.step),
            device=self.device,
        ):
            loss = loss * 0.0
            local_micro_loss_sum = local_micro_loss_sum * 0.0

        self.metric_accumulator.update(
            total_token_count=micro_total_token_count,
            effective_token_count=micro_effective_token_count,
            loss_sum=local_micro_loss_sum.detach(),
            model_flops=float(outputs["model_flops"]),
        )

        outputs["backward_loss_divisor"] = backward_loss_divisor
        ctx.loss = loss
        with accumulate_gradients(self.model, sync=ctx.should_sync):
            self.scaler.scale(ctx.loss).backward()

    def optimizer_step(self, ctx: TrainStepContext) -> None:
        """Scale accumulated gradients by global tokens and apply the optimizer step."""
        if not ctx.should_sync:
            return

        self.metric_accumulator.reduce_all()
        global_total_token_count = int(self.metric_accumulator.total_token_count.global_value)
        global_effective_token_count = int(self.metric_accumulator.effective_token_count.global_value)
        if global_effective_token_count <= 0:
            raise ValueError("Accumulation window must contain at least one supervised token.")

        self.scaler.unscale_(self.optimizer)
        outputs = ctx.outputs
        assert outputs is not None, "The forward step must populate ctx.outputs."
        backward_loss_divisor = int(outputs["backward_loss_divisor"])
        gradient_scale = float(backward_loss_divisor) * float(self.dp_world_size) / float(global_effective_token_count)
        with torch.no_grad():
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(gradient_scale)

        # 4. Apply the optimizer update at the end of the accumulation window.
        max_grad_norm = self.config.optim.clip_grad_norm
        if max_grad_norm is not None:
            clip_grad_norm_(self.model, max_grad_norm)

        step_lrs = [float(param_group["lr"]) for param_group in self.optimizer.param_groups]

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        ctx.outputs["global_total_token_count"] = global_total_token_count
        ctx.outputs["global_effective_token_count"] = global_effective_token_count
        ctx.outputs["global_loss_sum"] = self.metric_accumulator.loss_sum.global_value
        ctx.outputs["model_flops_per_step"] = self.metric_accumulator.model_flops.local_value
        ctx.outputs["step_lrs"] = step_lrs
        ctx.optimizer_step_completed = True

    def train_post_step(self, ctx: TrainStepContext) -> None:
        """Log one synchronized optimizer step and save checkpoints."""
        if not ctx.optimizer_step_completed:
            return

        outputs = ctx.outputs
        assert outputs is not None, "The forward step must populate ctx.outputs."

        self.step += 1
        self.timer.tick()
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
            build_mfu_log(
                model_flops_per_step=outputs["model_flops_per_step"],
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

        self.metric_accumulator.reset()

        self.save()
