"""Training engine for the video MLLM recipe."""

from __future__ import annotations

import math
from functools import partial

import torch
from mvp_dataset import TorchLoader

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import (
    get_data_parallel_group,
    get_data_parallel_world_size,
)
from mvp_engine.engine import ENGINE_REGISTRY, Engine, TrainStepContext
from mvp_engine.kit import MFUKit, MLLMModelKit, OptimKit, TokenNormedLossKit
from mvp_engine.utils.log import logger
from mvp_engine.utils.training import accumulate_gradients, clip_grad_norm_

from ..configs.schema import VideoMLLMConfig
from ..dataset.collator import VideoMLLMCollator
from ..dataset.dataset import build_dataset
from ..dataset.processor import build_qwen3_vl_processor
from ..guards.loss import PerTokenLossGuard
from ..model import patch_qwen3vl_conv3d, patch_qwen3vl_model_flops
from ..model.onevision import apply_onevision_swap


@ENGINE_REGISTRY.register()
class VideoMLLMEngine(Engine):
    """Recipe-local engine for supervised Qwen3-VL video fine-tuning.

    Mirrors the Qwen3-VL recipe engine but keeps video samples unpacked: uniform
    frame sampling and decode-then-expand preprocessing live in the recipe-local
    dataset. Both the uniform and codec paths deliberately omit
    ``mm_token_type_ids``, so ``compute_3d_position_ids`` returns None and the LLM
    falls back to default **1-D** positions (M-RoPE is not active). Proper M-RoPE
    (expanding ``video_grid_thw`` into ``grid_t`` rows of ``(1, h, w)``) is a
    tracked follow-up; see the NOTE in ``dataset/preprocess.py``.
    """

    ConfigClass = VideoMLLMConfig
    config: VideoMLLMConfig

    loss_guard: PerTokenLossGuard

    model_kit: MLLMModelKit
    mfu_kit: MFUKit
    optim_kit: OptimKit
    token_loss_kit: TokenNormedLossKit

    def __init__(self, config):
        """Initialize video-MLLM-local distributed state and metric reducers."""
        super().__init__(config)
        self.dp_world_size = get_data_parallel_world_size(self.device_mesh)
        self.dp_group = get_data_parallel_group(self.device_mesh)
        self.config.resolve_batching_config(data_parallel_world_size=self.dp_world_size)
        self.model_kit = MLLMModelKit()
        self.mfu_kit = MFUKit()
        self.optim_kit = OptimKit()
        self.token_loss_kit = TokenNormedLossKit(
            device=self.device,
            dp_world_size=self.dp_world_size,
            dp_group=self.dp_group,
        )

    def prepare_dataloader(self, workflow: str = "train"):
        """Build the train dataloader over preprocessed video samples.

        Args:
            workflow: Workflow name passed by the shared engine. Only ``train``
                is supported by this recipe.

        Returns:
            A ``TorchLoader`` pipeline that yields padded multimodal video
            batches, or ``None`` for unsupported non-training workflows.
        """
        if workflow != "train":
            logger.warning(f"Video MLLM engine does not support workflow '{workflow}'.")
            return

        self.processor = build_qwen3_vl_processor(
            self.config.model,
            video_encoding_strategy=self.config.data.video_encoding_strategy,
            vision_encoder_backend=self.config.model.vision_encoder_backend,
        )

        dataset = build_dataset(self.config, processor=self.processor)
        collate_fn = VideoMLLMCollator(pad_token_id=int(self.processor.tokenizer.pad_token_id))

        loader = TorchLoader(
            dataset,
            num_workers=int(self.config.data.num_workers),
            pin_memory=self.device.type in ["cuda", "npu"],
        )
        return loader.batch(
            batch_size=int(self.config.data.batch_size),
            drop_last=True,
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

        # Data strategy and visual backend are separate axes. The native Qwen3-VL
        # backend needs the conv3d compatibility patch; OneVision replaces the
        # visual tower and routes codec patch positions.
        model_patches = [patch_qwen3vl_model_flops]
        if self.config.model.uses_onevision_encoder:
            model_patches.append(
                partial(
                    apply_onevision_swap,
                    vision_encoder_name_or_path=self.config.model.vision_encoder_name_or_path,
                    attn_implementation="eager",
                    freeze_vision_encoder=bool(self.config.model.freeze_vision_encoder),
                )
            )
        else:
            model_patches.insert(0, patch_qwen3vl_conv3d)
        model = self.model_kit.apply_model_patches(model, model_patches)
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
        """Move one micro-batch to device and cast visual tensors to the active dtype.

        Video samples are unpacked, so this only relocates tensors and casts
        ``pixel_values_videos`` to the mixed-precision dtype. ``mm_token_type_ids``
        is not provided, so the LLM uses default **1-D** positions (M-RoPE is not
        active); see the class docstring and ``dataset/preprocess.py``.
        """
        device_batch = {}
        for key, value in ctx.data.items():
            device_batch[key] = value.to(self.device) if isinstance(value, torch.Tensor) else value

        pixel_values_videos = device_batch.get("pixel_values_videos")
        if pixel_values_videos is not None and pixel_values_videos.is_floating_point():
            device_batch["pixel_values_videos"] = pixel_values_videos.to(self.dtype)

        # Codec-patch path: hand the OneVision tower this micro-batch's patch positions via the
        # hidden attribute its routing patch reads. patch_positions is not a model kwarg, so it
        # must be popped from the batch (it would otherwise be forwarded into model(**inputs)).
        if self.config.data.uses_codec_patches:
            self.unwrapped_model.model._video_vlm_patch_positions = device_batch.pop("patch_positions", None)

        ctx.data = device_batch
        return ctx

    def forward_step(self, ctx: TrainStepContext) -> None:
        """Run one forward pass and collect micro-batch training metrics.

        The patched model returns unreduced per-token CE loss. Token counts and
        local FLOPs are carried into later hooks for accumulation-window
        reduction and logging.
        """
        data = ctx.data
        with torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.dtype != torch.float32,
        ):
            model_inputs = {
                key: value for key, value in data.items() if key not in {"total_tokens", "effective_tokens"}
            }
            outputs = self.model(**model_inputs)

        self.mfu_kit.accumulate_microbatch(
            model=self.unwrapped_model,
            batch_size=int(data["input_ids"].shape[0]),
            seq_len=int(data["input_ids"].shape[1]),
            attention_mask=data.get("attention_mask"),
            image_grid_thw=data.get("video_grid_thw"),
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
