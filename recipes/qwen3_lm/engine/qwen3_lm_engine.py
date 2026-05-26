"""Training engine for the Qwen3 LM recipe."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from transformers.optimization import get_scheduler
from transformers.utils.logging import disable_progress_bar

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import get_data_parallel_world_size
from mvp_engine.engine import ENGINE_REGISTRY, Engine, TrainStepContext
from mvp_engine.utils.log import logger
from mvp_engine.utils.metrics import DistributedMetricAccumulator
from mvp_engine.utils.training import accumulate_gradients, clip_grad_norm_

from ..configs.schema import Qwen3LMConfig
from ..dataset.collator import Qwen3LMCollator
from ..dataset.dataset import build_dataset
from ..dataset.processor import build_qwen3_tokenizer
from ..dataset.types import ModelInputs
from ..guards.loss import PerTokenLossGuard
from ..model import build_qwen3_model
from ..model.packing import apply_packed_fa2_patch, prepare_packed_model_inputs
from ..utils.log.mfu import build_mfu_log
from ..utils.misc import build_torch_loader, infer_total_steps, resolve_batching_config


@ENGINE_REGISTRY.register()
class Qwen3LMEngine(Engine):
    """Recipe-local engine for Qwen3 text-only supervised fine-tuning."""

    ConfigClass = Qwen3LMConfig
    config: Qwen3LMConfig

    tokenizer = None
    loss_guard: PerTokenLossGuard
    metric_accumulator: DistributedMetricAccumulator

    def __init__(self, config):
        """Initialize Qwen3 LM-local distributed state and metric reducers."""
        super().__init__(config)
        self.dp_world_size = get_data_parallel_world_size(self.device_mesh)
        disable_progress_bar()
        self.metric_accumulator = DistributedMetricAccumulator(device=self.device)
        self.metric_accumulator.register("total_token_count", accumulate="sum", reduce="sum")
        self.metric_accumulator.register("effective_token_count", accumulate="sum", reduce="sum")
        self.metric_accumulator.register("loss_sum", accumulate="sum", reduce="sum")
        self.metric_accumulator.register("model_flops", accumulate="sum", reduce="none")

    def prepare_dataloader(self, workflow: str = "train"):
        """Build the train dataloader over preprocessed text samples."""
        if workflow != "train":
            logger.warning(f"Qwen3 LM engine does not support workflow '{workflow}'.")
            return

        resolve_batching_config(self.config, data_parallel_world_size=self.dp_world_size)
        if self.tokenizer is None:
            self.tokenizer = build_qwen3_tokenizer(self.config.model)

        if int(self.config.loop.total_steps) == -1:
            self.config.loop.total_steps = infer_total_steps(
                self.config,
                tokenizer=self.tokenizer,
                device=self.device,
                data_parallel_world_size=self.dp_world_size,
            )

        dataset = build_dataset(self.config, tokenizer=self.tokenizer)
        collate_fn = Qwen3LMCollator(pad_token_id=int(self.tokenizer.pad_token_id))
        return build_torch_loader(
            self.config,
            dataset,
            collate_fn=collate_fn,
            device=self.device,
        )

    def prepare_model(self) -> torch.nn.Module:
        """Build, patch, compile, and parallelize the recipe model."""
        model = build_qwen3_model(self.config.model).to(self.device)
        if self.config.data.packing and getattr(model.config, "_attn_implementation", None) == "flash_attention_2":
            apply_packed_fa2_patch()
        logger.info(f"Model name: {model.__class__.__name__}")

        if self.config.model.compile:
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
        """Construct AdamW and initialize the per-token loss guard."""
        self.loss_guard = PerTokenLossGuard(
            spike_multiplier=self.config.optim.loss_spike_skip_multiplier,
            window_size=int(self.config.optim.loss_spike_skip_window_size),
            min_history=int(self.config.optim.loss_spike_skip_min_history),
        )

        trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise ValueError("No trainable parameters found for the Qwen3 LM recipe.")

        return torch.optim.AdamW(
            trainable_parameters,
            lr=float(self.config.optim.lr),
            weight_decay=float(self.config.optim.weight_decay),
        )

    def prepare_scheduler(self):
        """Construct the learning-rate schedule used by this recipe."""
        warmup_steps = math.ceil(self.total_steps * float(self.config.optim.warmup_ratio))
        return get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

    def train_pre_step(self, ctx: TrainStepContext) -> TrainStepContext:
        """Prepare one micro-batch for forward and count token metrics."""
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
            attn_implementation=getattr(self.unwrapped_model.config, "_attn_implementation", None),
            mask_dtype=self.dtype if self.dtype.is_floating_point else torch.float32,
        )
        ctx.data = batch
        return ctx

    def forward_step(self, ctx: TrainStepContext) -> None:
        """Run one forward pass and collect micro-batch training metrics."""
        data: ModelInputs = ctx.data
        model_inputs = {
            key: value
            for key, value in data.items()
            if key not in {"total_token_num", "effective_token_num", "source_sample_num"}
        }
        with torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        ):
            outputs = self.model(**model_inputs)

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
                    is_training=True,
                )
            ),
        }

    def backward_step(self, ctx: TrainStepContext) -> None:
        """Backward one micro-batch with delayed global token normalization."""
        outputs = ctx.outputs
        assert outputs is not None, "The forward step must populate ctx.outputs."
        assert "loss" in outputs, "The model output must contain 'loss' key."
        assert "logs" in outputs, "The model output must contain 'logs' key."

        ctx.should_sync = self.ga_state.advance()
        local_micro_loss_sum = outputs["loss"].sum()
        micro_effective_token_count = int(outputs["effective_tokens"])
        micro_total_token_count = int(outputs["total_tokens"])

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
