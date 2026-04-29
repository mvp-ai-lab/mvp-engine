"""Training engine for the OpenBee recipe."""

from __future__ import annotations

import math
import random
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mvp_dataset import TorchLoader
from transformers import ProcessorMixin
from transformers.optimization import get_scheduler
from transformers.utils.logging import disable_progress_bar

from mvp_engine.distributed.parallelize import parallelize_model
from mvp_engine.distributed.utils import get_data_parallel_world_size, is_main_process
from mvp_engine.engine import ENGINE_REGISTRY, Engine
from mvp_engine.utils.log import logger
from mvp_engine.utils.training import accumulate_gradients, clip_grad_norm_

from ..configs.schema import OpenbeeConfig
from ..dataset.collator import OpenbeeCollator
from ..dataset.dataset import build_dataset
from ..dataset.processor import build_qwen3_vl_processor
from ..dataset.types import ModelInputs
from ..guards.loss import PerTokenLossGuard
from ..model import build_qwen3_vl_model
from ..model.packing import apply_packed_fa2_patch, prepare_packed_model_inputs
from ..utils.log.mfu import build_mfu_log
from ..utils.metrics import DistributedMetricAccumulator
from ..utils.misc import infer_total_steps, resolve_batching_config


@ENGINE_REGISTRY.register()
class OpenbeeEngine(Engine):
    """Recipe-local engine for the OpenBee alignment stage."""

    ConfigClass = OpenbeeConfig
    config: OpenbeeConfig

    processor: ProcessorMixin | None = None
    loss_guard: PerTokenLossGuard
    metric_accumulator: DistributedMetricAccumulator

    def __init__(self, config):
        """Initialize OpenBee-local distributed state and metric reducers."""
        super().__init__(config)
        self.dp_world_size = get_data_parallel_world_size(self.device_mesh)
        disable_progress_bar()
        self.metric_accumulator = DistributedMetricAccumulator(device=self.device)
        self.metric_accumulator.register("total_token_count", accumulate="sum", reduce="sum")
        self.metric_accumulator.register("loss_token_count", accumulate="sum", reduce="sum")
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
            logger.warning(f"OpenBee engine does not support workflow '{workflow}'.")
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
        collate_fn = OpenbeeCollator(
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
        # FIXME: canbe removed once the patch system PR is merged
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
            raise ValueError("No trainable parameters found for the OpenBee recipe.")

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

    def before_train(self) -> None:
        """Initialize training, then apply post-pack fast resume when needed."""
        super().before_train()
        self.timer.set_progress(self.step, self.total_steps)

        if self.config.resume is None or self.step >= self.total_steps:
            return

        gradient_accumulation_steps = int(self.config.optim.gradient_accumulation_steps)
        resume_skip_micro_batches = self.step * gradient_accumulation_steps + int(self._accumulate_step)
        if resume_skip_micro_batches > 0:
            self.train_loader = self._build_fast_resume_train_loader(resume_skip_micro_batches)

    def _build_fast_resume_train_loader(self, resume_skip_micro_batches: int) -> Any:
        """Build a train loader advanced to the saved post-pack micro-batch boundary."""
        if self.processor is None:
            self.processor = build_qwen3_vl_processor(self.config.model)

        torch_rng_state = torch.get_rng_state()
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        skip_start_time = time.perf_counter()
        skip_counts: dict[int, int] = {}
        total_skip_markers = 0
        fast_resume_error: Exception | None = None

        try:
            if is_main_process():
                logger.info(
                    "OpenBee fast resume post-pack skip: "
                    f"step={self.step}, "
                    f"accumulate_step={self._accumulate_step}, "
                    f"micro_batches={resume_skip_micro_batches}"
                )

            pre_skip_dataset = build_dataset(
                self.config,
                processor=self.processor,
                resolve_refs=False,
                skip_mode="pre_calculate",
            )
            pre_skip_loader = TorchLoader(
                pre_skip_dataset,
                num_workers=int(self.config.data.num_workers),
                pin_memory=self.device.type in ["cuda", "npu"],
                persistent_workers=False,
                multiprocessing_context="spawn",
            ).batch(
                batch_size=int(self.config.data.batch_size),
                drop_last=True,
            )
            pre_skip_iterator = iter(pre_skip_loader)
            log_interval = 50

            for skipped_micro_batches in range(resume_skip_micro_batches):
                try:
                    marker_batch = next(pre_skip_iterator)
                except StopIteration:
                    pre_skip_iterator = iter(pre_skip_loader)
                    try:
                        marker_batch = next(pre_skip_iterator)
                    except StopIteration as exc:
                        raise RuntimeError("OpenBee fast resume marker loader did not yield any batches.") from exc

                if not isinstance(marker_batch, list) or not marker_batch:
                    raise RuntimeError("OpenBee fast resume marker loader yielded an invalid marker batch.")

                for marker in marker_batch:
                    worker_slot = int(marker["worker_slot"])
                    skip_counts[worker_slot] = skip_counts.get(worker_slot, 0) + 1
                    total_skip_markers += 1

                current_skip_count = skipped_micro_batches + 1
                if is_main_process() and current_skip_count % log_interval == 0:
                    logger.info(
                        "OpenBee fast resume skip progress: "
                        f"{current_skip_count}/{resume_skip_micro_batches} micro-batches"
                    )

            expected_skip_markers = resume_skip_micro_batches * int(self.config.data.batch_size)
            if total_skip_markers != expected_skip_markers:
                raise RuntimeError(
                    "OpenBee fast resume marker count mismatch: "
                    f"expected={expected_skip_markers}, actual={total_skip_markers}."
                )
            if not skip_counts:
                raise RuntimeError("OpenBee fast resume produced no post-pack skip markers.")
        except Exception as exc:
            fast_resume_error = exc
        finally:
            torch.set_rng_state(torch_rng_state)
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)

        fast_resume_ok = torch.tensor(
            0 if fast_resume_error is not None else 1,
            device=self.device,
            dtype=torch.long,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(fast_resume_ok, op=dist.ReduceOp.MIN)
        if int(fast_resume_ok.item()) == 0:
            if fast_resume_error is not None:
                raise RuntimeError("OpenBee fast resume failed on this rank.") from fast_resume_error
            raise RuntimeError("OpenBee fast resume failed on another rank.")

        if is_main_process():
            elapsed = time.perf_counter() - skip_start_time
            logger.info(
                "OpenBee fast resume skip counts ready: "
                f"workers={len(skip_counts)}, "
                f"post_pack_outputs={total_skip_markers}, "
                f"elapsed={elapsed:.1f}s"
            )

        dataset = build_dataset(
            self.config,
            processor=self.processor,
            skip_mode="perform",
            skip_counts=skip_counts,
        )
        collate_fn = OpenbeeCollator(
            pad_token_id=int(self.processor.tokenizer.pad_token_id),
            processor=self.processor,
        )
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

    def train_pre_step(self, data: ModelInputs) -> ModelInputs:
        """Prepare one micro-batch for forward.

        This counts total and shifted-label supervised tokens before moving
        tensors to the local device, prepares packed attention inputs, and casts
        visual tensors to the active mixed-precision dtype.
        """
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
        return batch

    def train_one_step(self, data: ModelInputs) -> dict[str, Any]:
        """Run one forward pass and collect micro-batch training metrics.

        The patched model returns unreduced per-token CE loss. Token counts and
        local FLOPs are carried into ``train_after_step`` for accumulation-window
        reduction and logging.
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

    def train_after_step(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Backward one micro-batch and step/log at accumulation boundaries.

        Per-token loss is backpropagated immediately with a fixed provisional
        divisor so data loading can overlap with GPU work. At the sync micro
        step, accumulated gradients are rescaled by the reduced global supervised
        token count before clipping and optimizer stepping. Loss-guarded spike
        micro-batches contribute zero loss to both gradients and logs.
        """
        assert "loss" in outputs, "The model output must contain 'loss' key."
        assert "logs" in outputs, "The model output must contain 'logs' key."

        is_sync = self.accumulate_step()

        # 1. Read this micro-step's local token/loss/flops.
        local_micro_loss_sum = outputs["loss"].sum()
        micro_loss_token_count = int(outputs["effective_tokens"])
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
            micro_loss_token_count,
            step=int(self.step),
            device=self.device,
        ):
            loss = loss * 0.0
            local_micro_loss_sum = local_micro_loss_sum * 0.0

        self.metric_accumulator.update(
            total_token_count=micro_total_token_count,
            loss_token_count=micro_loss_token_count,
            loss_sum=local_micro_loss_sum.detach(),
            model_flops=float(outputs["model_flops"]),
        )

        with accumulate_gradients(self.model, sync=is_sync):
            self.scaler.scale(loss).backward()

        if not is_sync:
            return outputs

        # 3. Resolve the final global token denominator, then normalize accumulated gradients.
        self.metric_accumulator.reduce_all()
        global_total_token_count = int(self.metric_accumulator.total_token_count.global_value)
        global_loss_token_count = int(self.metric_accumulator.loss_token_count.global_value)
        if global_loss_token_count <= 0:
            raise ValueError("Accumulation window must contain at least one supervised token.")

        self.scaler.unscale_(self.optimizer)
        gradient_scale = float(backward_loss_divisor) * float(self.dp_world_size) / float(global_loss_token_count)
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

        self.step += 1
        self.timer.tick()

        # 5. Write one optimizer-step log record.
        global_loss_sum = self.metric_accumulator.loss_sum.global_value
        outputs["logs"]["train/loss"] = float(global_loss_sum / int(global_loss_token_count))
        outputs["logs"].update(
            {
                "eta": self.timer.eta_string,
                "perf/batch_time": self.timer.batch_time,
                "perf/toks_per_sec": global_loss_token_count / self.timer.batch_time,
                "tokens/global_total": global_total_token_count,
                "tokens/global_loss": global_loss_token_count,
            }
        )
        outputs["logs"].update(
            build_mfu_log(
                model_flops_per_step=self.metric_accumulator.model_flops.local_value,
                device_type=self.device.type,
                precision=str(self.config.optim.mixed_precision),
                step_time_seconds=float(self.timer.batch_time_latest),
            )
        )

        for i, lr in enumerate(step_lrs):
            outputs["logs"][f"lr/group_{i}"] = lr

        logger.log_metrics(
            outputs["logs"],
            step=self.step,
            total_steps=self.total_steps,
        )

        self.metric_accumulator.reset()

        self.save()

        return outputs
