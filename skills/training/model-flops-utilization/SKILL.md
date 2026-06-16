---
name: model-flops-utilization
description: Add, review, update, and validate model FLOPs utilization using
  MFUKit, including recipe-provided calculate_model_flops, microbatch
  accumulation, device peak resolution, distributed reduction, and perf/mfu
  logging.
---

# Model FLOPs Utilization

## Goal

Use `MFUKit` for standard optimizer-step MFU logging:

```text
global_mfu =
  global_logical_model_flops_per_optimizer_step
  / synchronized_step_time_seconds
  / total_active_device_peak_flops_per_second
```

The kit owns accumulation, distributed reduction, peak resolution, and log
payload construction. The recipe/model still owns architecture-specific
`calculate_model_flops(...)`.

## Required Inputs

- model creation path and recipe model patches;
- actual batch metadata used for FLOPs: sequence length, packed segments,
  visual grids, routing counts, or freeze flags;
- engine forward step and post-step logging paths;
- device type, effective precision, and optional configured hardware peak;
- distributed topology and gradient accumulation boundary.

## Workflow

### 1. Initialize MFUKit

```python
from mvp_engine.kit import MFUKit

self.mfu_kit = MFUKit()
```

### 2. Provide Model FLOPs

Patch or implement `model.calculate_model_flops(...)` near the model code. Pass
actual metadata such as `attention_mask`, `pack_segment_ids`, `image_grid_thw`,
and freeze flags. Do not compute MFU itself in the model patch.

### 3. Accumulate Microbatch FLOPs

In `forward_step()` after real model inputs are known:

```python
flops_attention_mask = data.get("pack_segment_ids")
if flops_attention_mask is None:
    flops_attention_mask = data.get("attention_mask")

self.mfu_kit.accumulate_microbatch(
    model=self.unwrapped_model,
    batch_size=int(data["input_ids"].shape[0]),
    seq_len=int(data["input_ids"].shape[1]),
    attention_mask=flops_attention_mask,
    image_grid_thw=data.get("image_grid_thw"),
)
```

### 4. Log At Optimizer-Step Boundary

In post-step logging:

```python
logs.update(
    self.mfu_kit.build_log(
        device_type=self.device.type,
        precision=str(self.config.optim.mixed_precision),
        step_time_seconds=step_time_seconds,
    )
)
```

`build_log()` may return `{}` when the device peak cannot be resolved.

## Validation

### Soft Validation

- FLOPs use actual batch metadata rather than static config defaults;
- accumulated FLOPs cover exactly one optimizer step;
- step time is synchronized optimizer-step time;
- effective precision and peak TFLOPs are explicit or resolved by `MFUKit`;
- missing peak data does not create a guessed MFU value.

### Hard Validation

Runtime MFU validation requires real accelerator logs. CPU-only tests can check
wiring but not reported metric accuracy.

## Output

- State where `calculate_model_flops(...)` is injected.
- State where `MFUKit.accumulate_microbatch()` and `build_log()` are called.
- State device, precision, peak source, topology, validation, and remaining
  runtime gaps.

## Read On Demand

- `skills/kit/mfu-kit/SKILL.md`: kit API guide.
- `references/flops_formulas.md`: model FLOPs formulas.
- `references/distributed_mfu.md`: distributed scope and effective precision.
- `references/hardware_peak_flops.csv`: known hardware peak table.
