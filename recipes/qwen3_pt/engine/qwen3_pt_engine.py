"""Training engine for the Qwen3 text-only pretraining recipe."""

from __future__ import annotations

from typing import Any

import torch

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import (
    get_data_parallel_group,
    get_data_parallel_world_size,
)
from mvp_engine.engine import ENGINE_REGISTRY, Engine, TrainStepContext
from mvp_engine.kit import MFUKit, OptimKit, TokenNormedLossKit
from mvp_engine.kit.llm import LLMDataKit, LLMModelKit, ModelInputs, PackingOptions
from mvp_engine.utils.log import logger
from mvp_engine.utils.training import accumulate_gradients, clip_grad_norm_

from ..configs.schema import Qwen3PTConfig
from ..model import patch_qwen3_model_flops
from ..model.packing import prepare_packed_model_inputs
from ..utils.misc import infer_total_steps


@ENGINE_REGISTRY.register()
class Qwen3PTEngine(Engine):
    """Recipe-local engine for Qwen3 text-only pretraining."""

    ConfigClass = Qwen3PTConfig
    config: Qwen3PTConfig

    tokenizer: Any

    data_kit: LLMDataKit
    model_kit: LLMModelKit
    mfu_kit: MFUKit
    optim_kit: OptimKit
    token_loss_kit: TokenNormedLossKit

    def __init__(self, config):
        """Initialize recipe-local distributed state and reusable kits."""
        super().__init__(config)
        self.dp_world_size = get_data_parallel_world_size(self.device_mesh)
        self.dp_group = get_data_parallel_group(self.device_mesh)
        self.config.resolve_batching_config(data_parallel_world_size=self.dp_world_size)
        self.data_kit = LLMDataKit()
        self.model_kit = LLMModelKit()
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
        """Build the train dataloader over packed text pretraining samples."""
        if workflow != "train":
            logger.warning(f"Qwen3 pretraining engine does not support workflow '{workflow}'.")
            return None

        self.tokenizer = self.data_kit.build_tokenizer(self.config.model.pretrained_model_name_or_path)

        # When total_steps=-1, do one finite data pass to count packed samples.
        if int(self.config.loop.total_steps) == -1:
            self.config.loop.total_steps = infer_total_steps(
                self.config,
                tokenizer=self.tokenizer,
                device=self.device,
                data_parallel_world_size=self.dp_world_size,
                data_parallel_group=self.dp_group,
            )

        packing = PackingOptions(
            selection_strategy=self.config.data.packing_selection_strategy,
            open_pack_limit=int(self.config.data.packing_open_pack_limit),
            buffer_size=int(self.config.data.packing_buffer_size),
        )
        dataset = self.data_kit.build_dataset(
            dataset_path=self.config.data.train_path,
            tokenizer=self.tokenizer,
            max_seq_len=int(self.config.data.max_seq_len),
            text_field=self.config.data.text_field,
            seed=self.config.seed,
            packing=packing,
        )
        collate_fn = self.data_kit.build_collator(pad_token_id=int(self.tokenizer.pad_token_id))
        return self.data_kit.build_dataloader(
            dataset,
            batch_size=int(self.config.data.batch_size),
            num_workers=int(self.config.data.num_workers),
            pin_memory=self.device.type in ["cuda", "npu"],
            collate_fn=collate_fn,
        )

    def prepare_model(self) -> torch.nn.Module:
        """Build, patch, and parallelize the Qwen3 text model."""
        model = self.model_kit.build_model(
            self.config.model.pretrained_model_name_or_path,
            train_from_scratch=self.config.model.train_from_scratch,
            init_seed=int(self.config.model.init_seed),
            torch_dtype=getattr(self.config.model, "torch_dtype", "auto"),
            attn_implementation=self.config.model.attn_implementation,
        ).to(self.device)
        logger.info(f"Model name: {model.__class__.__name__}")

        model = self.model_kit.apply_model_patches(model, [patch_qwen3_model_flops])
        model = self.token_loss_kit.apply_chunked_token_loss_patch(model)
        model = self.model_kit.apply_freeze_policy(model, freeze_llm=self.config.model.freeze_llm)

        for parameter in model.parameters():
            if parameter.requires_grad and parameter.dtype != torch.float32:
                parameter.data = parameter.data.to(dtype=torch.float32)

        if self.config.model.gradient_checkpointing.enabled:
            model = self.model_kit.apply_gradient_checkpointing(
                model,
                use_reentrant=self.config.model.gradient_checkpointing.use_reentrant,
            )

        if self.config.model.compile.enabled:
            model = self.model_kit.apply_model_compile(
                model,
                backend=self.config.model.compile.backend,
                mode=self.config.model.compile.mode,
            )

        return parallelize_model(
            model,
            device_mesh=self.device_mesh,
            backend_kwargs=self.config.parallel.backend_kwargs.model_dump(),
        )

    def prepare_optimizer(self) -> torch.optim.Optimizer:
        """Construct the AdamW optimizer used by this recipe."""
        return self.optim_kit.build_optimizer(
            self.model,
            optimizer=self.config.optim.optimizer,
            lr=float(self.config.optim.lr),
            weight_decay=float(self.config.optim.weight_decay),
            betas=(float(self.config.optim.beta1), float(self.config.optim.beta2)),
        )

    def prepare_scheduler(self):
        """Construct the cosine-with-min-lr schedule used by this recipe."""
        return self.optim_kit.build_lr_scheduler(
            optimizer=self.optimizer,
            lr_scheduler="cosine_with_min_lr",
            num_warmup_steps=int(self.config.optim.warmup_steps),
            num_training_steps=self.total_steps,
            scheduler_specific_kwargs={"min_lr_rate": float(self.config.optim.min_lr_rate)},
        )

    def train_pre_step(self, ctx: TrainStepContext) -> TrainStepContext:
        """Move one micro-batch to device and build packed text model inputs."""
        batch: ModelInputs = self.data_kit.to_device(ctx.data, self.device)
        ctx.data = prepare_packed_model_inputs(
            batch,
            mask_dtype=self.dtype if self.dtype.is_floating_point else torch.float32,
        )
        return ctx

    def forward_step(self, ctx: TrainStepContext) -> None:
        """Run one forward pass and accumulate micro-batch FLOPs."""
        data: ModelInputs = ctx.data
        with torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        ):
            model_inputs = {
                key: value
                for key, value in data.items()
                if key not in {"pack_segment_ids", "total_tokens", "effective_tokens"}
            }
            outputs = self.model(**model_inputs)

        flops_attention_mask = data.get("pack_segment_ids")
        if flops_attention_mask is None:
            flops_attention_mask = data.get("attention_mask")
        self.mfu_kit.accumulate_microbatch(
            model=self.unwrapped_model,
            batch_size=int(data["input_ids"].shape[0]),
            seq_len=int(data["input_ids"].shape[1]),
            attention_mask=flops_attention_mask,
            is_training=True,
            freeze_llm=bool(self.config.model.freeze_llm),
        )

        ctx.outputs = {
            "loss": outputs.loss,
            "logs": {
                "train/loss": 0,
            },
        }

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

        loss = self.token_loss_kit.accumulate_microbatch(
            loss_sum=local_micro_loss_sum,
            effective_tokens=micro_effective_token_count,
            total_tokens=micro_total_token_count,
            backward_divisor=backward_loss_divisor,
        )

        ctx.loss = loss
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
        outputs["global_total_token_count"] = token_loss_stats.global_total_tokens
        outputs["global_effective_token_count"] = token_loss_stats.global_effective_tokens
        outputs["global_loss_sum"] = token_loss_stats.global_loss_sum
        ctx.outputs["step_lrs"] = step_lrs
        ctx.optimizer_step_completed = True
        self.token_loss_kit.reset()

    def train_post_step(self, ctx: TrainStepContext) -> None:
        """Log one synchronized optimizer step and save checkpoints."""
        outputs = ctx.outputs

        step_time_seconds = float(self.timer.progress_time_latest)
        global_total_token_count = int(outputs["global_total_token_count"])
        global_effective_token_count = int(outputs["global_effective_token_count"])

        outputs["logs"].update(
            {
                "train/loss": float(outputs["global_loss_sum"] / int(global_effective_token_count)),
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
