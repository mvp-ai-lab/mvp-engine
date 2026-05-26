"""MFU logging helpers for the Qwen3 LM recipe."""

from __future__ import annotations

PEAK_FLOPS_BY_DEVICE = {
    "NVIDIA H200": {
        "bf16": 989e12,
        "fp16": 989e12,
        "fp32": 67e12,
    },
    "NVIDIA H100": {
        "bf16": 989e12,
        "fp16": 989e12,
        "fp32": 67e12,
    },
    "NVIDIA A100": {
        "bf16": 312e12,
        "fp16": 312e12,
        "fp32": 19.5e12,
    },
}


def build_mfu_log(
    *,
    model_flops_per_step: float,
    device_type: str,
    precision: str,
    step_time_seconds: float,
) -> dict[str, float]:
    """Build MFU logs from local FLOPs and measured step time."""
    logs = {"perf/model_flops": float(model_flops_per_step)}
    if step_time_seconds <= 0 or device_type != "cuda":
        return logs

    try:
        import torch

        device_name = torch.cuda.get_device_name()
    except Exception:
        return logs

    peak_flops = None
    for name_prefix, precision_peaks in PEAK_FLOPS_BY_DEVICE.items():
        if device_name.startswith(name_prefix):
            peak_flops = precision_peaks.get(precision)
            break
    if peak_flops is None:
        return logs

    logs["perf/mfu"] = float(model_flops_per_step) / (float(peak_flops) * float(step_time_seconds))
    return logs
