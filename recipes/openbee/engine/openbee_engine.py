"""Training engine for the OpenBee recipe."""

from __future__ import annotations

import math
import random
import time
from collections import deque
from typing import Any

import numpy as np
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

from ..configs.schema import OpenbeeConfig
from ..dataset import (
    SOURCE_SAMPLE_COUNT_KEY,
    ModelInputs,
    OpenbeeCollator,
    build_dataset,
    build_qwen3_vl_processor,
    lightweight_process_sample,
    resolve_step_inference_dataset_source,
)
from ..model import build_qwen3_vl_model
from ..model.packing import apply_packed_fa2_patch, prepare_packed_model_inputs
from ..utils.log.mfu import build_mfu_log
from ..utils.metrics import MetricAccumulator


@ENGINE_REGISTRY.register()
class OpenbeeEngine(Engine):
    """Recipe-local engine for the OpenBee alignment stage."""

    ConfigClass = OpenbeeConfig
    config: OpenbeeConfig

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
        self._loss_spike_history: deque[float] = deque(maxlen=int(self.config.optim.loss_spike_skip_window_size))

    def _resolve_batching_config(self) -> None:
        """Resolve OpenBee global batch size into micro batch size or accumulation."""

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

    @staticmethod
    def _format_step_inference_duration(seconds: float | None) -> str:
        """Format a compact duration for step-inference progress logs."""
        if seconds is None or not math.isfinite(seconds) or seconds < 0:
            return "unknown"

        remaining_seconds = int(seconds)
        hours, remainder = divmod(remaining_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h{minutes:02d}m{secs:02d}s"
        if minutes > 0:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"

    @staticmethod
    def _format_step_inference_finish_time(seconds_from_now: float | None) -> str:
        """Format the estimated wall-clock finish time for progress logs."""
        if seconds_from_now is None or not math.isfinite(seconds_from_now) or seconds_from_now < 0:
            return "unknown"
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + seconds_from_now))

    @staticmethod
    def _count_step_inference_source_samples(batch: dict[str, Any]) -> int:
        """Count raw source samples represented by one lightweight batch."""
        source_sample_count = batch.get(SOURCE_SAMPLE_COUNT_KEY)
        if isinstance(source_sample_count, torch.Tensor):
            return int(source_sample_count.sum().item())
        if isinstance(source_sample_count, (list, tuple)):
            return sum(int(value) for value in source_sample_count)
        if source_sample_count is not None:
            return int(source_sample_count)
        return int(batch["input_ids"].shape[0])

    @staticmethod
    def _infer_dataset_source_total_rows(dataset: Any) -> int | None:
        """Best-effort row-count fallback from the underlying mvp_dataset source."""
        sources = getattr(dataset, "_source", None)
        if not isinstance(sources, (list, tuple)) or not sources:
            return None

        total_rows = 0
        for source in sources:
            source_total = getattr(source, "total_rows", None)
            if isinstance(source_total, int):
                total_rows += source_total
                continue

            num_rows = getattr(source, "num_rows", None)
            if isinstance(num_rows, int):
                total_rows += num_rows
                continue

            datasets = getattr(source, "datasets", None)
            if isinstance(datasets, (list, tuple)):
                for dataset_spec in datasets:
                    dataset_rows = getattr(dataset_spec, "num_rows", None)
                    if isinstance(dataset_rows, int):
                        total_rows += dataset_rows

        return total_rows if total_rows > 0 else None

    def _log_step_inference_progress(
        self,
        *,
        source_sample_count: int,
        total_source_sample_count: int | None,
        packed_sample_count: int,
        interval_source_sample_count: int,
        interval_seconds: float,
    ) -> None:
        """Print one rank-0 step-inference progress line."""
        if not is_main_process():
            return

        throughput = interval_source_sample_count / max(interval_seconds, 1e-6)
        if total_source_sample_count is not None and total_source_sample_count > 0:
            percent = min(source_sample_count / total_source_sample_count * 100.0, 100.0)
            total_text = str(total_source_sample_count)
            percent_text = f"{percent:.2f}%"
            remaining_samples = max(total_source_sample_count - source_sample_count, 0)
            eta_seconds = remaining_samples / throughput if throughput > 0 else None
        else:
            total_text = "unknown"
            percent_text = "unknown"
            eta_seconds = None

        logger.info(
            "Step inference progress: "
            f"samples={source_sample_count}/{total_text}, "
            f"percent={percent_text}, "
            f"throughput={throughput:.2f} samples/s, "
            f"eta={self._format_step_inference_duration(eta_seconds)}, "
            f"estimated_finish={self._format_step_inference_finish_time(eta_seconds)}, "
            f"packed_samples={packed_sample_count}"
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
                logger.info("OpenBee step inference: disabling cache for the temporary lightweight dataloader.")
                temp_config.data.cache = False

            step_dataset_paths, total_source_sample_count, used_main_table_only = resolve_step_inference_dataset_source(
                str(temp_config.data.train_path)
            )
            if used_main_table_only:
                logger.info(
                    "OpenBee step inference: using only the Lance main table from meta.json; "
                    "reference tables will not be opened."
                )

            # Step 2.2: build a temporary dataset/loader that uses the
            # lightweight preprocess but keeps the same packing behaviour.
            temp_collate_fn = OpenbeeCollator(
                pad_token_id=int(self.processor.tokenizer.pad_token_id),
                processor=self.processor,
            )
            temp_dataset = build_dataset(
                temp_config,
                processor=self.processor,
                process_fn=lightweight_process_sample,
                resample=False,
                dataset_paths=step_dataset_paths,
            )
            if total_source_sample_count is None:
                total_source_sample_count = self._infer_dataset_source_total_rows(temp_dataset)
            temp_loader = TorchLoader(
                temp_dataset,
                num_workers=int(temp_config.data.num_workers),
                pin_memory=self.device.type in ["cuda", "npu"],
                persistent_workers=False,
                drop_last=False,
                multiprocessing_context="spawn",
            ).batch(
                batch_size=int(temp_config.data.batch_size),
                drop_last=True,
                collate_fn=temp_collate_fn,
            )

            # Step 2.3: count how many packed samples this rank receives from
            # the temporary loader. Every N loader rounds, all ranks synchronize
            # for progress logging; ranks that finish early keep participating
            # in these interval reductions until every rank is done.
            log_interval = int(temp_config.data.step_inference_log_interval)
            count_start_time = time.perf_counter()
            last_log_time = count_start_time
            local_packed_sample_count = 0
            local_source_sample_count = 0
            interval_packed_sample_count = 0
            interval_source_sample_count = 0
            temp_iterator = iter(temp_loader)
            local_done = False
            loader_round = 0
            distributed = dist.is_available() and dist.is_initialized()

            while True:
                had_batch = 0
                if not local_done:
                    try:
                        batch = next(temp_iterator)
                    except StopIteration:
                        local_done = True
                    else:
                        packed_batch_count = int(batch["input_ids"].shape[0])
                        source_batch_count = self._count_step_inference_source_samples(batch)
                        local_packed_sample_count += packed_batch_count
                        local_source_sample_count += source_batch_count
                        interval_packed_sample_count += packed_batch_count
                        interval_source_sample_count += source_batch_count
                        had_batch = 1

                loader_round += 1
                should_sync = loader_round % log_interval == 0

                if not distributed:
                    if should_sync and interval_source_sample_count > 0:
                        now = time.perf_counter()
                        self._log_step_inference_progress(
                            source_sample_count=local_source_sample_count,
                            total_source_sample_count=total_source_sample_count,
                            packed_sample_count=local_packed_sample_count,
                            interval_source_sample_count=interval_source_sample_count,
                            interval_seconds=now - last_log_time,
                        )
                        interval_source_sample_count = 0
                        interval_packed_sample_count = 0
                        last_log_time = now

                    if local_done:
                        if interval_source_sample_count > 0:
                            now = time.perf_counter()
                            self._log_step_inference_progress(
                                source_sample_count=local_source_sample_count,
                                total_source_sample_count=total_source_sample_count,
                                packed_sample_count=local_packed_sample_count,
                                interval_source_sample_count=interval_source_sample_count,
                                interval_seconds=now - last_log_time,
                            )
                        break
                    continue

                if not should_sync:
                    continue

                now = time.perf_counter()
                interval_seconds = now - last_log_time
                count_stats = torch.tensor(
                    [
                        interval_source_sample_count,
                        local_source_sample_count,
                        interval_packed_sample_count,
                        local_packed_sample_count,
                        had_batch,
                    ],
                    device=self.device,
                    dtype=torch.long,
                )
                elapsed_stats = torch.tensor(interval_seconds, device=self.device, dtype=torch.float64)
                dist.all_reduce(count_stats, op=dist.ReduceOp.SUM)
                dist.all_reduce(elapsed_stats, op=dist.ReduceOp.MAX)

                global_interval_source_sample_count = int(count_stats[0].item())
                global_source_sample_count = int(count_stats[1].item())
                global_packed_sample_count = int(count_stats[3].item())
                active_rank_count = int(count_stats[4].item())

                if global_interval_source_sample_count > 0:
                    self._log_step_inference_progress(
                        source_sample_count=global_source_sample_count,
                        total_source_sample_count=total_source_sample_count,
                        packed_sample_count=global_packed_sample_count,
                        interval_source_sample_count=global_interval_source_sample_count,
                        interval_seconds=float(elapsed_stats.item()),
                    )

                interval_source_sample_count = 0
                interval_packed_sample_count = 0
                last_log_time = now

                if active_rank_count <= 0:
                    break

            # Step 2.4: reduce all local counts to get the real global packed
            # sample count across all ranks.
            total_sample_count = torch.tensor(local_packed_sample_count, device=self.device, dtype=torch.long)
            if distributed:
                dist.all_reduce(total_sample_count, op=dist.ReduceOp.SUM)

            total_sample_count_value = int(total_sample_count.item())
            if total_sample_count_value <= 0:
                raise RuntimeError("OpenBee step inference found no packed training samples.")

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

    def run_iter_train(self) -> None:
        """Run iteration-based training with window-prefetched token normalization."""

        def _count_total_tokens(attention_mask: torch.Tensor) -> int:
            """Count all non-padding tokens in one collated micro-batch."""
            return int(attention_mask.sum().item())

        def _count_valid_loss_tokens(labels: torch.Tensor, *, ignore_index: int = -100) -> int:
            """Count the valid shifted labels used by causal-LM cross entropy."""
            shifted_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:]
            return int((shifted_labels != ignore_index).sum().item())

        gradient_accumulation_steps = int(self.config.optim.gradient_accumulation_steps)
        train_iterator = iter(self.train_loader)

        if hasattr(self, "timer"):
            self.timer.set_progress(self.step, self.total_steps)

        resume_skip_micro_batches = 0
        if self.config.resume is not None and self.step < self.total_steps:
            resume_skip_micro_batches = self.step * gradient_accumulation_steps + int(self._accumulate_step)

        if resume_skip_micro_batches > 0:
            torch_rng_state = torch.get_rng_state()
            python_rng_state = random.getstate()
            numpy_rng_state = np.random.get_state()
            skip_start_time = time.perf_counter()
            log_interval = 50
            if is_main_process():
                logger.info(
                    "Resume data skip: "
                    f"step={self.step}, "
                    f"accumulate_step={self._accumulate_step}, "
                    f"micro_batches={resume_skip_micro_batches}"
                )

            try:
                for skipped_micro_batches in range(resume_skip_micro_batches):
                    try:
                        data = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(self.train_loader)
                        try:
                            data = next(train_iterator)
                        except StopIteration as exc:
                            raise RuntimeError(
                                "OpenBee train loader did not yield any batches during resume skip."
                            ) from exc
                    del data
                    torch.distributed.barrier()
                    current_skip_count = skipped_micro_batches + 1
                    if is_main_process() and current_skip_count % log_interval == 0:
                        logger.info(
                            f"Resume data skip progress: {current_skip_count}/{resume_skip_micro_batches} micro-batches"
                        )
            finally:
                torch.set_rng_state(torch_rng_state)
                random.setstate(python_rng_state)
                np.random.set_state(numpy_rng_state)

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            if is_main_process():
                elapsed = time.perf_counter() - skip_start_time
                logger.info(
                    "Resume data skip finished: "
                    f"micro_batches={resume_skip_micro_batches}, "
                    f"elapsed={self._format_step_inference_duration(elapsed)}"
                )

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
                        raise RuntimeError("OpenBee train loader did not yield any batches.") from exc
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
        """Run the optimizer step and include recipe-local MFU metrics in logs."""
        assert "loss" in outputs, "The model output must contain 'loss' key."
        assert "logs" in outputs, "The model output must contain 'logs' key."

        is_sync = self.accumulate_step()
        global_total_token_count = self.metric_accumulator.get("global_total_token_count")
        if global_total_token_count is None or int(global_total_token_count) <= 0:
            raise ValueError("OpenBee accumulation window is missing a valid global total token count.")
        global_loss_token_count = self.metric_accumulator.get("global_token_count")
        if global_loss_token_count is None or int(global_loss_token_count) <= 0:
            raise ValueError("OpenBee accumulation window is missing a valid global token count.")
        local_token_count = self.metric_accumulator.get("local_token_count")
        if local_token_count is None or int(local_token_count) <= 0:
            raise ValueError("OpenBee accumulation window is missing a valid local token count.")

        local_micro_loss_sum = outputs["loss"].sum()
        micro_loss_token_count = int(outputs["effective_tokens"])
        if micro_loss_token_count < 0:
            raise ValueError("OpenBee micro-batch has an invalid effective token count.")

        self.metric_accumulator.update(
            local_loss_sum=local_micro_loss_sum.detach().to(device=self.device, dtype=torch.float64),
            local_model_flops=float(outputs["model_flops"]),
        )

        # DDP/FSDP averages gradients across the DP/FSDP ranks, so each
        # micro-step uses the local summed loss with the globally reduced token
        # denominator and compensates by the real DP world size.
        loss = local_micro_loss_sum / int(global_loss_token_count)
        if self.dp_world_size > 1:
            loss = loss * float(self.dp_world_size)

        loss_spike_skip_multiplier = self.config.optim.loss_spike_skip_multiplier
        if loss_spike_skip_multiplier is not None:
            micro_loss_stats = torch.stack(
                (
                    local_micro_loss_sum.detach().to(device=self.device, dtype=torch.float64),
                    torch.tensor(float(micro_loss_token_count), device=self.device, dtype=torch.float64),
                )
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(micro_loss_stats, op=dist.ReduceOp.SUM)

            global_micro_token_count = int(micro_loss_stats[1].item())
            if global_micro_token_count > 0:
                current_loss = float((micro_loss_stats[0] / micro_loss_stats[1]).item())
                min_history = int(self.config.optim.loss_spike_skip_min_history)
                has_enough_history = len(self._loss_spike_history) >= min_history
                loss_spike_baseline = (
                    sum(self._loss_spike_history) / len(self._loss_spike_history) if has_enough_history else None
                )
                is_loss_spike = loss_spike_baseline is not None and current_loss > loss_spike_baseline * float(
                    loss_spike_skip_multiplier
                )
                if is_loss_spike:
                    baseline_text = "n/a" if loss_spike_baseline is None else f"{loss_spike_baseline:.4f}"
                    logger.warning(
                        f"Loss spike skip at step {self.step}: "
                        f"micro_loss={current_loss:.4f}, "
                        f"baseline_loss={baseline_text}, "
                        f"history_size={len(self._loss_spike_history)}, "
                        f"micro_tokens={global_micro_token_count}"
                    )
                    loss = loss * 0.0
                else:
                    self._loss_spike_history.append(current_loss)

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
            raise ValueError("OpenBee accumulation window is missing a valid local loss sum.")
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
            total_steps=self.total_steps,
        )

        self.metric_accumulator.reset()

        self.save()

        return outputs
