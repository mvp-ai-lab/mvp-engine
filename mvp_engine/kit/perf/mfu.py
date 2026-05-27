"""Reusable MFU utilities."""

from __future__ import annotations

import subprocess
from typing import Any

import torch
import torch.distributed as dist

from mvp_engine.distributed.utils import get_world_size


class MFUKit:
    """Build model FLOPs utilization metrics from recipe-provided FLOPs."""

    PEAK_TFLOPS_BY_DEVICE_AND_PRECISION = {
        ("NVIDIA H100 SXM", "bf16"): 989.0,
        ("NVIDIA H100 SXM", "fp16"): 989.0,
        ("NVIDIA H100 PCIe", "bf16"): 756.0,
        ("NVIDIA H100 PCIe", "fp16"): 756.0,
        ("NVIDIA H200", "bf16"): 989.0,
        ("NVIDIA H200", "fp16"): 989.0,
        ("NVIDIA A100 80GB", "bf16"): 312.0,
        ("NVIDIA A100 80GB", "fp16"): 312.0,
        ("NVIDIA A100 40GB", "bf16"): 312.0,
        ("NVIDIA A100 40GB", "fp16"): 312.0,
        ("NVIDIA L40S", "bf16"): 733.0,
        ("NVIDIA L40S", "fp16"): 733.0,
        ("NVIDIA RTX 4090", "bf16"): 330.0,
        ("NVIDIA RTX 4090", "fp16"): 330.0,
    }

    def __init__(self) -> None:
        """Initialize empty per-step FLOPs accounting state."""
        self.reset()

    def reset(self) -> None:
        """Clear accumulated FLOPs state."""
        self._accumulated_model_flops = 0.0

    def accumulate_microbatch(
        self,
        *,
        model_flops: float | None = None,
        model: Any | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        **model_flops_kwargs: Any,
    ) -> float:
        """Accumulate model FLOPs for one micro-batch."""
        microbatch_flops = model_flops
        if microbatch_flops is None:
            if model is None or batch_size is None or seq_len is None:
                raise ValueError("Either model_flops or model/batch_size/seq_len must be provided.")
            microbatch_flops = model.calculate_model_flops(
                batch_size=int(batch_size),
                seq_len=int(seq_len),
                **model_flops_kwargs,
            )
        if microbatch_flops < 0:
            raise ValueError("model_flops must be non-negative.")

        self._accumulated_model_flops += float(microbatch_flops)
        return float(microbatch_flops)

    def calculate_mfu(
        self,
        *,
        model_flops_per_step: float,
        step_time_seconds: float,
        device_peak_tflops: float,
        world_size: int,
    ) -> float:
        """Compute model FLOPs utilization for one synchronized optimizer step."""
        if step_time_seconds <= 0:
            raise ValueError("step_time_seconds must be > 0")
        if device_peak_tflops <= 0:
            raise ValueError("device_peak_tflops must be > 0")
        if world_size <= 0:
            raise ValueError("world_size must be > 0")

        total_peak_flops = device_peak_tflops * 1e12 * world_size
        achieved_flops_per_second = model_flops_per_step / step_time_seconds
        return float(achieved_flops_per_second / total_peak_flops)

    def reduce_to_global_flops_and_step_time(
        self,
        *,
        model_flops_per_step: float,
        step_time_seconds: float,
    ) -> tuple[float, float]:
        """Reduce local rank FLOPs by sum and synchronized step time by max."""
        if not dist.is_available() or not dist.is_initialized():
            return float(model_flops_per_step), float(step_time_seconds)

        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        values = torch.tensor(
            [float(model_flops_per_step), float(step_time_seconds)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(values[0], op=dist.ReduceOp.SUM)
        dist.all_reduce(values[1], op=dist.ReduceOp.MAX)
        return float(values[0].item()), float(values[1].item())

    def normalize_device_name(self, device_name: str) -> str | None:
        """Map a raw GPU name to a canonical peak-table key."""
        normalized = device_name.strip()
        if not normalized:
            return None

        candidates = {
            "H200": "NVIDIA H200",
            "H100 SXM": "NVIDIA H100 SXM",
            "H100 PCIE": "NVIDIA H100 PCIe",
            "A100 80GB": "NVIDIA A100 80GB",
            "A100 40GB": "NVIDIA A100 40GB",
            "L40S": "NVIDIA L40S",
            "RTX 4090": "NVIDIA RTX 4090",
        }
        upper_name = normalized.upper()
        for marker, canonical in candidates.items():
            if marker in upper_name:
                return canonical
        return normalized

    def detect_cuda_device_name(self) -> str | None:
        """Return the first CUDA device name reported by ``nvidia-smi``."""
        if torch.cuda.is_available():
            return self.normalize_device_name(torch.cuda.get_device_name(torch.cuda.current_device()))

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

        first_device = next((line.strip() for line in result.stdout.splitlines() if line.strip()), None)
        return self.normalize_device_name(first_device) if first_device is not None else None

    def resolve_peak_tflops(
        self,
        *,
        device_type: str,
        precision: str,
        configured_device_name: str | None = None,
        configured_peak_tflops: float | None = None,
    ) -> tuple[str | None, float | None]:
        """Resolve the device name and peak TFLOPs used for MFU logging."""
        if configured_peak_tflops is not None:
            resolved_device_name = (
                self.normalize_device_name(configured_device_name) if configured_device_name else None
            )
            return resolved_device_name, float(configured_peak_tflops)

        resolved_device_name = self.normalize_device_name(configured_device_name) if configured_device_name else None
        if resolved_device_name is None and device_type == "cuda":
            resolved_device_name = self.detect_cuda_device_name()
        if resolved_device_name is None:
            return None, None

        peak_tflops = self.PEAK_TFLOPS_BY_DEVICE_AND_PRECISION.get((resolved_device_name, precision))
        if peak_tflops is None:
            return resolved_device_name, None
        return resolved_device_name, float(peak_tflops)

    def build_log(
        self,
        *,
        model_flops_per_step: float | None = None,
        model: Any | None = None,
        device_type: str,
        precision: str,
        batch_size: int | None = None,
        seq_len: int | None = None,
        step_time_seconds: float,
        gradient_accumulation_steps: int = 1,
        configured_device_name: str | None = None,
        configured_peak_tflops: float | None = None,
        **model_flops_kwargs: Any,
    ) -> dict[str, float]:
        """Build a standard ``perf/mfu`` log payload."""
        if step_time_seconds <= 0:
            if model_flops_per_step is None and model is None:
                self.reset()
            return {}

        local_model_flops_per_step = model_flops_per_step
        if local_model_flops_per_step is None:
            if model is not None and batch_size is not None and seq_len is not None:
                local_model_flops_per_step = model.calculate_model_flops(
                    batch_size=int(batch_size),
                    seq_len=int(seq_len),
                    **model_flops_kwargs,
                ) * int(gradient_accumulation_steps)
            elif self._accumulated_model_flops > 0:
                local_model_flops_per_step = self._accumulated_model_flops
            else:
                raise ValueError("No model FLOPs are available for MFU logging.")

        if model_flops_per_step is None and model is None:
            self.reset()

        resolved_device_name, device_peak_tflops = self.resolve_peak_tflops(
            device_type=device_type,
            precision=precision,
            configured_device_name=configured_device_name,
            configured_peak_tflops=configured_peak_tflops,
        )
        if resolved_device_name is None or device_peak_tflops is None:
            return {}

        global_model_flops_per_step, global_step_time_seconds = self.reduce_to_global_flops_and_step_time(
            model_flops_per_step=float(local_model_flops_per_step),
            step_time_seconds=float(step_time_seconds),
        )

        return {
            "perf/mfu": self.calculate_mfu(
                model_flops_per_step=float(global_model_flops_per_step),
                step_time_seconds=float(global_step_time_seconds),
                device_peak_tflops=float(device_peak_tflops),
                world_size=int(get_world_size()),
            )
        }
