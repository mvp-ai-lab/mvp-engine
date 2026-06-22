"""Training engine for the Qwen3 text-only recipe."""

from __future__ import annotations

import math
from typing import Any

import torch

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import (
    get_data_parallel_group,
    get_data_parallel_world_size,
)
from mvp_engine.engine import ENGINE_REGISTRY, Engine, TrainStepContext
from mvp_engine.kit import (
    LLMDataKit,
    LLMDataSpec,
    LLMLoaderSpec,
    LLMModelKit,
    LLMPackingSpec,
    LLMPretrainTextSchemaHandler,
    LLMPretrainTextTokenizationHandler,
    LLMSampleSpec,
    LLMSourceSpec,
    LLMStepEstimationKit,
    MFUKit,
    OptimKit,
    TokenNormedLossKit,
)
from mvp_engine.kit.llm import ModelInputs
from mvp_engine.utils.log import logger
from mvp_engine.utils.training import accumulate_gradients, clip_grad_norm_

from ..configs.schema import Qwen3Config
from ..model import patch_qwen3_model_flops
from ..model.packing import prepare_packed_model_inputs


@ENGINE_REGISTRY.register()
class Qwen3Engine(Engine):
    """Recipe-local engine for Qwen3 text-only stages."""

    ConfigClass = Qwen3Config
    config: Qwen3Config

    tokenizer: Any

    data_kit: LLMDataKit
    model_kit: LLMModelKit
    step_estimation_kit: LLMStepEstimationKit
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
        self.step_estimation_kit = LLMStepEstimationKit()
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
        """Build the train dataloader over packed text pretrain samples.

        Args:
            workflow: Workflow name passed by the shared engine. Only ``train``
                is supported by this recipe.

        Returns:
            A ``TorchLoader`` pipeline that yields padded text-LM batches, or
            ``None`` for unsupported non-training workflows.
        """
        if workflow != "train":
            logger.warning(f"Qwen3 pretrain engine does not support workflow '{workflow}'.")
            return

        # Step 1: build the shared tokenizer.
        self.tokenizer = self.data_kit.build_tokenizer(self.config.model.pretrained_model_name_or_path)

        # Step 2: declare the Qwen pretrain sample handlers.
        sample_spec = LLMSampleSpec(
            schema_handler=LLMPretrainTextSchemaHandler(text_field=self.config.data.text_field),
            tokenization_handler=LLMPretrainTextTokenizationHandler(
                tokenizer=self.tokenizer,
                max_seq_len=int(self.config.data.max_seq_len),
            ),
        )

        # Step 3: declare distributed placement and the training data spec.
        distribution = self.data_kit.build_distribution_spec()
        packing_spec = LLMPackingSpec(
            max_seq_len=int(self.config.data.max_seq_len),
            tail_policy=self.config.data.packing_tail_policy,
            isolate_attention=self.config.data.packing_isolate_attention,
            isolate_position_ids=self.config.data.packing_isolate_position_ids,
        )
        loader_spec = LLMLoaderSpec(
            batch_size=int(self.config.data.batch_size),
            num_workers=int(self.config.data.num_workers),
        )
        data_spec = LLMDataSpec(
            source=LLMSourceSpec(
                dataset_path=self.config.data.train_path,
                seed=int(self.config.seed),
                resample=True,
                shuffle_mode="chunk",
            ),
            sample=sample_spec,
            packing=packing_spec,
            loader=loader_spec,
            distribution=distribution,
        )

        # Step 4: when total_steps=-1, consume one finite packed data pass to estimate one-epoch steps.
        if int(self.config.loop.total_steps) == -1:
            estimation_spec = LLMDataSpec(
                source=LLMSourceSpec(
                    dataset_path=self.config.data.train_path,
                    seed=int(self.config.seed),
                    resample=False,
                    shuffle_mode="none",
                ),
                sample=sample_spec,
                packing=packing_spec,
                loader=LLMLoaderSpec(
                    batch_size=int(self.config.data.batch_size),
                    num_workers=int(self.config.data.num_workers),
                    drop_last=False,
                ),
                distribution=distribution,
            )
            estimation_dataset = self.data_kit.build_dataset(estimation_spec)
            estimate = self.step_estimation_kit.estimate_total_steps(
                estimation_dataset,
                batch_size=int(self.config.data.batch_size),
                gradient_accumulation_steps=int(self.config.optim.gradient_accumulation_steps),
                data_parallel_world_size=self.dp_world_size,
                data_parallel_group=self.dp_group,
                device=self.device,
            )
            self.config.loop.total_steps = estimate.total_steps

        # Step 5: build the real training dataset with resampling enabled.
        dataset = self.data_kit.build_dataset(data_spec)

        # Step 6: wrap the dataset in the normal TorchLoader.
        return self.data_kit.build_dataloader(dataset, data_spec, device=self.device)

    def prepare_model(self) -> torch.nn.Module:
        """Build, patch, compile, and parallelize the recipe model.

        Returns:
            The distributed-ready Qwen3 text model.
        """
        model = self.model_kit.build_model(
            self.config.model.pretrained_model_name_or_path,
            load_pretrained_model=self.config.model.load_pretrained_model,
            random_init_seed=int(self.config.seed),
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
                mode="hf",
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
        )

    def prepare_scheduler(self):
        """Construct the learning-rate schedule used by this recipe."""
        warmup_steps = math.ceil(self.total_steps * float(self.config.optim.warmup_ratio))
        return self.optim_kit.build_lr_scheduler(
            optimizer=self.optimizer,
            lr_scheduler="cosine",
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

    def train_pre_step(self, ctx: TrainStepContext) -> TrainStepContext:
        """Move one micro-batch to device and build packed text model inputs."""
        batch: ModelInputs = self.data_kit.to_device(ctx.data, self.device)
        ctx.data = prepare_packed_model_inputs(
            batch,
            attn_implementation=getattr(self.unwrapped_model.config, "_attn_implementation", None),
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
