"""Training engine for the Qwen3-VL recipe."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import (
    get_data_parallel_group,
    get_data_parallel_world_size,
)
from mvp_engine.engine import ENGINE_REGISTRY, Engine, TrainStepContext
from mvp_engine.kit import (
    MFUKit,
    MLLMDataKit,
    MLLMModelKit,
    ModelInputs,
    OptimKit,
    PackingOptions,
    PerTokenLossGuard,
    TokenNormedLossKit,
)
from mvp_engine.utils.log import logger
from mvp_engine.utils.training import accumulate_gradients, clip_grad_norm_

from ..configs.schema import Qwen3VLConfig
from ..model import patch_qwen3vl_conv3d, patch_qwen3vl_model_flops
from ..model.packing import prepare_packed_model_inputs
from ..utils.misc import infer_total_steps

if TYPE_CHECKING:
    from transformers import ProcessorMixin


@ENGINE_REGISTRY.register()
class Qwen3VLEngine(Engine):
    """Recipe-local engine for the Qwen3-VL recipe."""

    ConfigClass = Qwen3VLConfig
    config: Qwen3VLConfig

    processor: ProcessorMixin
    loss_guard: PerTokenLossGuard

    data_kit: MLLMDataKit
    model_kit: MLLMModelKit
    mfu_kit: MFUKit
    optim_kit: OptimKit
    token_loss_kit: TokenNormedLossKit

    def __init__(self, config):
        """Initialize Qwen3-VL-local distributed state and metric reducers."""
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
            logger.warning(f"Qwen3-VL engine does not support workflow '{workflow}'.")
            return

        # Step 1: build the shared processor.
        self.processor = self.data_kit.build_processor(
            self.config.model.pretrained_model_name_or_path,
            image_max_pixels=self.config.model.image_max_pixels,
        )

        # Step 2: when total_steps=-1, run one finite lightweight data pass to
        # count packed samples and infer one-epoch optimization steps.
        if int(self.config.loop.total_steps) == -1:
            self.config.loop.total_steps = infer_total_steps(
                self.config,
                processor=self.processor,
                device=self.device,
                data_parallel_world_size=self.dp_world_size,
                data_parallel_group=self.dp_group,
            )

        # Step 3: build the real training/eval dataset with the full preprocess.
        packing = PackingOptions(
            selection_strategy=self.config.data.packing_selection_strategy,
            open_pack_limit=int(self.config.data.packing_open_pack_limit),
            buffer_size=int(self.config.data.packing_buffer_size),
        )
        dataset = self.data_kit.build_dataset(
            dataset_path=self.config.data.train_path,
            processor=self.processor,
            max_seq_len=int(self.config.data.max_seq_len),
            resample=True,
            resolve_refs=True,
            ref_columns=self.config.data.ref_columns,
            seed=self.config.seed,
            packing=packing,
            thinking_mode=self.config.data.thinking_mode,
        )

        # Step 4: wrap the dataset in the normal TorchLoader and return the
        # batched dataloader used by the engine.
        collate_fn = self.data_kit.build_collator(
            pad_token_id=int(self.processor.tokenizer.pad_token_id),
            processor=self.processor,
        )
        return self.data_kit.build_dataloader(
            dataset,
            batch_size=int(self.config.data.batch_size),
            num_workers=int(self.config.data.num_workers),
            pin_memory=self.device.type in ["cuda", "npu"],
            collate_fn=collate_fn,
        )

    def prepare_model(self) -> torch.nn.Module:
        """Build, patch, compile, and parallelize the recipe model.

        Returns:
            The distributed-ready Qwen3-VL model.
        """
        model = self.model_kit.build_model(
            self.config.model.pretrained_model_name_or_path,
            trust_remote_code=getattr(self.config.model, "trust_remote_code", True),
            torch_dtype=getattr(self.config.model, "torch_dtype", "auto"),
            attn_implementation=self.config.model.attn_implementation,
        ).to(self.device)
        logger.info(f"Model name: {model.__class__.__name__}")

        model = self.model_kit.apply_model_patches(model, [patch_qwen3vl_conv3d, patch_qwen3vl_model_flops])
        model = self.token_loss_kit.apply_chunked_token_loss_patch(model)

        model = self.model_kit.apply_freeze_policy(
            model,
            freeze_vit=self.config.model.freeze_vit,
            freeze_projector=self.config.model.freeze_projector,
            freeze_llm=self.config.model.freeze_llm,
        )

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

        parallelized_model = parallelize_model(
            model,
            device_mesh=self.device_mesh,
            backend_kwargs=self.config.parallel.backend_kwargs.model_dump(),
        )

        return parallelized_model

    def prepare_optimizer(self) -> torch.optim.Optimizer:
        """Construct AdamW and initialize the per-token loss guard.

        Returns:
            The optimizer used by this recipe.
        """
        self.loss_guard = PerTokenLossGuard(
            spike_multiplier=self.config.optim.loss_spike_skip_multiplier,
            window_size=int(self.config.optim.loss_spike_skip_window_size),
            min_history=int(self.config.optim.loss_spike_skip_min_history),
            group=self.dp_group,
            group_world_size=self.dp_world_size,
        )
        return self.optim_kit.build_optimizer(
            self.model,
            optimizer=self.config.optim.optimizer,
            lr=float(self.config.optim.lr),
            weight_decay=float(self.config.optim.weight_decay),
        )

    def prepare_scheduler(self):
        """Construct the learning-rate schedule used by this recipe.

        Returns:
            The Hugging Face cosine scheduler used by this recipe.
        """
        warmup_steps = math.ceil(self.total_steps * float(self.config.optim.warmup_ratio))
        return self.optim_kit.build_lr_scheduler(
            optimizer=self.optimizer,
            lr_scheduler="cosine",
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

    def train_pre_step(self, ctx: TrainStepContext) -> TrainStepContext:
        """Prepare one micro-batch for forward.

        This reads DataKit token counts, moves tensors to the local device,
        prepares packed Qwen3-VL inputs, and casts visual tensors to the active
        mixed-precision dtype.
        """
        batch: ModelInputs = self.data_kit.to_device(ctx.data, self.device)

        batch = prepare_packed_model_inputs(
            batch,
            model_config=self.unwrapped_model.config,
            attn_implementation=getattr(self.unwrapped_model.config, "_attn_implementation", None),
            mask_dtype=self.dtype if self.dtype.is_floating_point else torch.float32,
        )

        ctx.data = self.data_kit.to_device(batch, self.device)
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
            image_grid_thw=data.get("image_grid_thw"),
            is_training=True,
            freeze_vit=bool(self.config.model.freeze_vit),
            freeze_projector=bool(self.config.model.freeze_projector),
            freeze_llm=bool(self.config.model.freeze_llm),
        )

        ctx.outputs = {
            "loss": outputs.loss,
            "logs": {
                "train/loss": 0,
            },
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
        micro_effective_token_count = int(ctx.data["effective_tokens"])
        micro_total_token_count = int(ctx.data["total_tokens"])

        # 2. Backward immediately so the dataloader can prepare later micro-batches
        # while GPU work is active. The exact per-token denominator is applied to
        # accumulated gradients once the full accumulation window token count is known.
        backward_loss_divisor = (
            int(self.config.data.batch_size)
            * int(self.config.data.max_seq_len)
            * int(self.config.optim.gradient_accumulation_steps)
        )

        if self.loss_guard.check(
            local_micro_loss_sum,
            micro_effective_token_count,
            step=int(self.step),
            device=self.device,
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
