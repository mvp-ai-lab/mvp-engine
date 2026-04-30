"""Recipe-local MFU utilities for PanguVL."""

from __future__ import annotations

import subprocess
from typing import Any

import torch
import torch.distributed as dist

from mvp_engine.distributed.utils import get_world_size

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


def calculate_mfu(
    *,
    model_flops_per_step: float,
    step_time_seconds: float,
    device_peak_tflops: float,
    world_size: int,
) -> float:
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
    *,
    model_flops_per_step: float,
    step_time_seconds: float,
) -> tuple[float, float]:
    """Convert local-rank measurements into global-step quantities.

    Global MFU should compare:
    - global achieved FLOPs/s: sum of per-rank step FLOPs divided by step time
    - global peak FLOPs/s: per-device peak times world size

    For synchronous data parallel training, the effective step time is the slowest
    rank's step duration, so use a max reduction across ranks.
    """
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


def normalize_device_name(device_name: str) -> str | None:
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


def detect_cuda_device_name() -> str | None:
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
    if first_device is None:
        return None
    return normalize_device_name(first_device)


def resolve_peak_tflops(
    *,
    device_type: str,
    precision: str,
    configured_device_name: str | None,
    configured_peak_tflops: float | None,
) -> tuple[str | None, float | None]:
    if configured_peak_tflops is not None:
        return normalize_device_name(configured_device_name) if configured_device_name else None, float(
            configured_peak_tflops
        )

    resolved_device_name = normalize_device_name(configured_device_name) if configured_device_name else None
    if resolved_device_name is None and device_type == "cuda":
        resolved_device_name = detect_cuda_device_name()
    if resolved_device_name is None:
        return None, None

    peak_tflops = PEAK_TFLOPS_BY_DEVICE_AND_PRECISION.get((resolved_device_name, precision))
    if peak_tflops is None:
        return resolved_device_name, None
    return resolved_device_name, float(peak_tflops)


def build_mfu_log(
    *,
    model_flops_per_step: float | None = None,
    model: Any | None = None,
    device_type: str,
    precision: str,
    batch_size: int | None = None,
    seq_len: int | None = None,
    image_grid_thw: Any = None,
    step_time_seconds: float,
    gradient_accumulation_steps: int = 1,
    freeze_vit: bool = False,
    freeze_merger: bool = False,
    freeze_llm: bool = False,
) -> dict[str, float]:
    if step_time_seconds <= 0:
        return {}

    resolved_device_name, device_peak_tflops = resolve_peak_tflops(
        device_type=device_type,
        precision=precision,
        configured_device_name=None,
        configured_peak_tflops=None,
    )
    if resolved_device_name is None or device_peak_tflops is None:
        return {}

    local_model_flops_per_step = model_flops_per_step
    if local_model_flops_per_step is None:
        if model is None or batch_size is None or seq_len is None:
            raise ValueError("Either model_flops_per_step or model/batch_size/seq_len must be provided.")
        local_model_flops_per_step = model.calculate_model_flops(
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            image_grid_thw=image_grid_thw,
            is_training=True,
            freeze_vit=freeze_vit,
            freeze_merger=freeze_merger,
            freeze_llm=freeze_llm,
        ) * int(gradient_accumulation_steps)

    global_model_flops_per_step, global_step_time_seconds = reduce_to_global_flops_and_step_time(
        model_flops_per_step=float(local_model_flops_per_step),
        step_time_seconds=float(step_time_seconds),
    )

    return {
        "perf/mfu": float(
            calculate_mfu(
                model_flops_per_step=float(global_model_flops_per_step),
                step_time_seconds=float(global_step_time_seconds),
                device_peak_tflops=float(device_peak_tflops),
                world_size=int(get_world_size()),
            )
        )
    }
