"""Training engine for the video MLLM recipe."""

from __future__ import annotations

import math
from functools import partial

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
    MLLMDataSpec,
    MLLMLoaderSpec,
    MLLMModelKit,
    MLLMPackingSpec,
    MLLMSampleSpec,
    MLLMSourceSpec,
    MLLMTokenizationHandler,
    ModelInputs,
    OptimKit,
    TokenNormedLossKit,
)
from mvp_engine.utils.log import logger
from mvp_engine.utils.training import accumulate_gradients, clip_grad_norm_

from ..configs.schema import VideoMLLMConfig
from ..dataset.media import (
    build_image_media_handler,
    build_mixed_media_handler,
    build_video_media_handler,
)
from ..dataset.processor import attach_onevision_processor
from ..dataset.schema import VideoChatSchemaHandler
from ..model import patch_qwen3vl_model_flops
from ..model.onevision import apply_onevision_swap, bind_video_layout
from ..model.packing import prepare_packed_video_model_inputs


@ENGINE_REGISTRY.register()
class VideoMLLMEngine(Engine):
    """Recipe-local engine for supervised Qwen3-VL video fine-tuning.

    DataKit owns source loading, chat tokenization, packing, media loading, and
    collation. Recipe-local schema/media handlers keep video strategy dispatch
    and OneVision tensor layout outside the generic kit.
    """

    ConfigClass = VideoMLLMConfig
    config: VideoMLLMConfig

    data_kit: MLLMDataKit
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

        self.processor = self.data_kit.build_processor(
            self.config.model.pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        attach_onevision_processor(self.processor, self.config.model)
        if self.config.data.modality == "image":
            media_handler = build_image_media_handler(self.config, processor=self.processor)
        elif self.config.data.modality == "mixed":
            media_handler = build_mixed_media_handler(self.config, processor=self.processor)
        else:
            media_handler = build_video_media_handler(self.config, processor=self.processor)
        sample_spec = MLLMSampleSpec(
            schema_handler=VideoChatSchemaHandler(processor=self.processor),
            media_handler=media_handler,
            tokenization_handler=MLLMTokenizationHandler(
                processor=self.processor,
                max_seq_len=int(self.config.data.max_seq_len),
            ),
        )
        distribution = self.data_kit.build_distribution_spec(device_mesh=self.device_mesh)
        packing_spec = MLLMPackingSpec(
            max_seq_len=int(self.config.data.max_seq_len),
            algorithm="multi_pack",
            selection_strategy=self.config.data.packing_selection_strategy,
            open_pack_limit=int(self.config.data.packing_open_pack_limit),
            buffer_size=int(self.config.data.packing_buffer_size),
            block_causal=True,
        )
        loader_spec = MLLMLoaderSpec(
            batch_size=int(self.config.data.batch_size),
            num_workers=int(self.config.data.num_workers),
        )
        data_spec = MLLMDataSpec(
            source=MLLMSourceSpec(
                dataset_path=self.config.data.train_path,
                dataset_source=self.config.data.source,
                ref_columns=tuple(self.config.data.ref_columns),
                seed=int(self.config.seed),
                resample=True,
                resolve_refs=True,
            ),
            sample=sample_spec,
            packing=packing_spec,
            loader=loader_spec,
            distribution=distribution,
        )
        dataset = self.data_kit.build_dataset(data_spec)
        return self.data_kit.build_dataloader(dataset, data_spec, device=self.device)

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

        # All video encoding strategies feed the OneVision visual tower. Strategy-specific
        # preprocessing changes the pixels/grid/patch positions, not the model backend.
        model_patches = [
            patch_qwen3vl_model_flops,
            partial(
                apply_onevision_swap,
                vision_encoder_name_or_path=self.config.model.vision_encoder_name_or_path,
                attn_implementation="eager",
                freeze_vision_encoder=bool(self.config.model.freeze_vision_encoder),
            ),
        ]
        model = self.model_kit.apply_model_patches(model, model_patches)
        model = self.token_loss_kit.apply_chunked_token_loss_patch(model)

        # Resume post-swap weights (e.g. Stage-1 aligned projector) AFTER the OneVision swap,
        # so the OneVision tower/merger keys exist. strict=False: LLM/encoder match the base,
        # only the trained projector differs.
        if self.config.model.init_weights_from:
            from safetensors.torch import load_file

            state = load_file(self.config.model.init_weights_from)
            result = model.load_state_dict(state, strict=False)
            # Fail loudly if the trained projector/merger is absent from the checkpoint: strict=False
            # would otherwise leave a randomly-initialized merger and silently train Stage-2 from scratch.
            merger_keys = {k for k in model.state_dict() if "visual.merger." in k}
            missing_merger = sorted(merger_keys - set(state))
            if not merger_keys:
                raise RuntimeError(
                    f"init_weights_from {self.config.model.init_weights_from}: no visual.merger.* params found "
                    "in the model; cannot verify the Stage-1 projector loaded."
                )
            if missing_merger:
                raise RuntimeError(
                    f"init_weights_from {self.config.model.init_weights_from}: {len(missing_merger)} visual.merger "
                    f"weights absent from the checkpoint (e.g. {missing_merger[:3]}); Stage-2 would train from a "
                    "random projector."
                )
            logger.info(
                f"init_weights_from {self.config.model.init_weights_from}: loaded {len(state)} tensors "
                f"(merger keys={len(merger_keys)}), missing={len(result.missing_keys)} "
                f"unexpected={len(result.unexpected_keys)}"
            )

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
                mode=self.config.model.gradient_checkpointing.mode,
                target_modules=self.config.model.gradient_checkpointing.target_modules,
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
        """Construct AdamW.

        When ``optim.vision_lr`` is set and the OneVision encoder is unfrozen, its
        parameters (``model.visual.encoder.*``) get a separate, lower learning rate
        (mirrors LLaVA-OneVision-2.0's ``--vision_lr``); the merger/projector and the
        LLM keep the main ``optim.lr``.

        Returns:
            The optimizer used by this recipe.
        """
        vision_lr = getattr(self.config.optim, "vision_lr", None)
        lr_groups = None
        if vision_lr is not None:
            lr_groups = [(("model.visual.encoder.",), float(vision_lr))]
        optimizer = self.optim_kit.build_optimizer(
            self.model,
            optimizer=self.config.optim.optimizer,
            lr=float(self.config.optim.lr),
            weight_decay=float(self.config.optim.weight_decay),
            lr_groups=lr_groups,
        )
        group_summary = [
            (f"{group['lr']:.2e}", sum(p.numel() for p in group["params"])) for group in optimizer.param_groups
        ]
        logger.info(f"optimizer LR groups (lr, n_params): {group_summary}")
        return optimizer

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
        """Move one packed DataKit batch to device and prepare model inputs."""
        device_batch: ModelInputs = self.data_kit.to_device(ctx.data, self.device)

        pixel_values_videos = device_batch.get("pixel_values_videos")
        if pixel_values_videos is not None and pixel_values_videos.is_floating_point():
            device_batch["pixel_values_videos"] = pixel_values_videos.to(self.dtype)
            # NaN probe: flag non-finite decoded video tensors (data-side bug) with the offending rank.
            if not torch.isfinite(device_batch["pixel_values_videos"]).all():
                _r = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                logger.error(
                    f"[NAN-PROBE] non-finite pixel_values_videos rank={_r} node={_r // 8} "
                    f"shape={tuple(pixel_values_videos.shape)}"
                )

        device_batch = prepare_packed_video_model_inputs(
            device_batch,
            attn_implementation=getattr(self.unwrapped_model.config, "_attn_implementation", None),
            mask_dtype=self.dtype if self.dtype.is_floating_point else torch.float32,
        )

        # Bind visual-token layout onto the OneVision adapter and keep the remaining model kwargs.
        # Shared with the eval path (one feeding implementation) so train and eval never drift.
        ctx.data = bind_video_layout(self.unwrapped_model.model, device_batch)
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
                key: value
                for key, value in data.items()
                if key not in {"pack_segment_ids", "total_tokens", "effective_tokens"}
            }
            outputs = self.model(**model_inputs)

        # NaN probe: when local loss is non-finite, log the rank/node and whether the input video
        # tensors vs the logits are the source (data-side vs model/hardware). Cheap: only on nan.
        if not torch.isfinite(outputs.loss).all():
            _r = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            _pv = data.get("pixel_values_videos")
            _logits = getattr(outputs, "logits", None)
            logger.error(
                f"[NAN-PROBE] non-finite loss rank={_r} node={_r // 8} "
                f"pixel_values_finite={bool(torch.isfinite(_pv).all()) if _pv is not None else None} "
                f"logits_finite={bool(torch.isfinite(_logits).all()) if _logits is not None else None} "
                f"input_ids_shape={tuple(data['input_ids'].shape)} effective_tokens={data.get('effective_tokens')}"
            )

        # `video_grid_thw` here is a synthetic [1, visual_token_count, 1] placeholder row, not a
        # real spatial grid, and the visual tower is the (frozen) OneVision encoder rather than the
        # native Qwen3-VL ViT that the FLOPs estimator models. Skip the vision-FLOPs term instead of
        # feeding a placeholder grid into the native-ViT formula; the dominant trained-LLM FLOPs are unaffected.
        flops_attention_mask = data.get("pack_segment_ids")
        if flops_attention_mask is None:
            flops_attention_mask = data.get("attention_mask")
        self.mfu_kit.accumulate_microbatch(
            model=self.unwrapped_model,
            batch_size=int(data["input_ids"].shape[0]),
            seq_len=int(data["input_ids"].shape[1]),
            attention_mask=flops_attention_mask,
            image_grid_thw=None,
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
