"""MFU-focused engine snippet (non-MFU parts intentionally hidden)."""

from mvp_engine.distributed.utils import get_world_size

PEAK_TFLOPS_BY_DEVICE_AND_PRECISION = {"NVIDIA H200": {"bf16": 989.0, "fp16": 989.0, "fp32": 67.0}}


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


class ViTClassificationEngine:
    """Only MFU-relevant methods are shown."""

    def log_mfu_metrics(self, log_dict: dict[str, float], step: int) -> None:
        model_flops = self.unwrapped_model.calculate_model_flops(
            batch_size=int(self.config.data.batch_size),
            image_size=int(self.config.model.image_size),
            patch_size=int(self.unwrapped_model.config.patch_size),
            is_training=True,
        )
        step_time = float(self.timer.batch_time_latest)
        peak_tflops = float(self.config.log.mfu.peak_tflops)

        mfu = calculate_mfu(
            model_flops_per_step=model_flops,
            step_time_seconds=step_time,
            device_peak_tflops=peak_tflops,
            world_size=get_world_size(),
        )

        log_dict["perf/mfu"] = float(mfu)
        log_dict["perf/step_time"] = step_time
        self.logger.log(log_dict, step=step)


# Train loop / dataloader / optimizer / scheduler details are intentionally hidden.
