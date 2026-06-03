---
name: mfu-kit
description: Use MFUKit for global model FLOPs utilization logging, including
  microbatch FLOPs accumulation, device peak resolution, distributed reduction,
  and perf/mfu log payload construction.
---

# MFU Kit

## Goal

Use `MFUKit` to log optimizer-step model FLOPs utilization:

- `accumulate_microbatch(...)` records local logical model FLOPs;
- `build_log(...)` resolves peak TFLOPs, reduces global FLOPs and step time, and
  returns `{"perf/mfu": ...}`;
- recipe/model code still owns architecture-specific `calculate_model_flops`.

## Required Inputs

- where the recipe computes real batch metadata;
- a model method or explicit value for local microbatch FLOPs;
- step timing at optimizer-step boundary;
- device type, effective precision, and optional configured device peak.

## Workflow

1. Initialize `self.mfu_kit = MFUKit()`.
2. Add a recipe model patch that attaches `calculate_model_flops(...)` when the
   model does not provide it.
3. In `forward_step()`, call `accumulate_microbatch(model=..., batch_size=...,
   seq_len=..., **metadata)`.
4. In post-step logging, call `build_log(device_type=..., precision=...,
   step_time_seconds=...)` and merge the returned payload into logs.

## Validation

### Soft Validation

- FLOPs use actual batch metadata, not only config defaults;
- gradient accumulation microbatches are included exactly once;
- step time is the synchronized optimizer-step time;
- effective precision and device peak are explicit or resolved by the kit;
- missing peak data results in no MFU log rather than a guessed value.

## Output

- State where FLOPs are computed, accumulated, and logged.
- State device/precision/peak source and validation status.

## Read On Demand

- `skills/training/model-flops-utilization/SKILL.md`: feature-oriented workflow
  and formula details.
