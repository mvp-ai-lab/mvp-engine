---
name: model-flops-utilization
description: Add, review, and validate model FLOPs utilization MFU support for
  mvp-engine recipes.
---

# Model FLOPs Utilization

## Goal

Add reliable global optimizer-step MFU for the current recipe:

- Inject `calculate_model_flops(...)` into the runtime model instance.
- Resolve hardware peak from real GPU name and effective precision.
- Log MFU under `perf/mfu`.

## Required Inputs

Identify these before editing:

- model creation entrypoint
- engine/training loop with step timing and logger access
- model architecture and executed components
- batch size, sequence/token counts, gradient accumulation
- distributed topology: DDP, FSDP, TP, PP, EP, or hybrid
- world size and data-parallel size
- declared precision and effective compute precision
- wrapper precision policy, such as FSDP2 `mp_policy`
- TF32 state: `torch.backends.cuda.matmul.allow_tf32` and `torch.get_float32_matmul_precision()`
- active GPU name and matching `references/hardware_peak_flops.csv` row. If no
  match, search the internet for the GPU's peak FLOPs at the effective precision
  or ask the user to provide it.

Ask the user only when hardware, precision, launch method, or topology cannot be derived.

## Workflow

Default metric:

```text
global_mfu =
  global_logical_model_flops_per_optimizer_step
  / synchronized_step_time_seconds
  / total_active_device_peak_flops_per_second
```

Do not mix local-rank FLOPs with global hardware peak. Treat communication,
data loading, optimizer math, and activation-checkpoint recompute as overhead
that lowers standard MFU through step time.

### 1. Locate Runtime Integration Points

Find the recipe-local places that correspond to these roles:

- model builder, before or after parallel wrapping;
- batch preparation, where actual sequence lengths, masks, image grids, packed
  segment ids, or sparse-routing metadata are available;
- forward step, where one micro-batch is executed;
- accumulation or backward step, where micro-batch metrics are accumulated;
- optimizer-step boundary, after all gradient accumulation micro-steps;
- post-step logging, where elapsed step time and logger payloads are assembled.

For example, Basic VLM injects FLOPs in `model/qwen3_vl.py`, computes per
micro-batch FLOPs in `engine/basic_vlm_engine.py::forward_step`, accumulates
them with a metric accumulator, and logs MFU through `utils/log/mfu.py`.

### 2. Add Model FLOPs Injection

Inject `calculate_model_flops(...)` into the runtime model instance. Do not
replace the model class unless the recipe already uses that pattern.

```python
from types import MethodType


def inject_model_flops_calculation(model):
    def calculate_model_flops(self, *, batch_size: int, seq_len: int, is_training: bool = True) -> float:
        ...

    model.calculate_model_flops = MethodType(calculate_model_flops, model)
    return model
```

Keep the signature architecture-specific and pass the metadata needed for an
accurate count, such as:

- `attention_mask`, packed segment ids, or sequence lengths for attention pair
  counts;
- `image_grid_thw`, frame counts, or visual token counts for VLMs;
- freeze flags or trainability metadata for forward-only, input-gradient-only,
  and fully trainable components;
- router or per-expert token counts for MoE and sparse routing.

Return logical model FLOPs for the local micro-batch by default. If the helper
returns any other scope, document it at the call site and convert it before MFU
logging.

Read `references/flops_formulas.md` for dense, VLM, MoE, sparse routing, packed
attention, freeze-policy, and KV reuse rules.

### 3. Wire Per-Micro-Batch FLOPs

Call `calculate_model_flops(...)` in the same step that owns the real model
inputs, usually immediately after or around the forward pass. Use the actual
prepared batch, not static config defaults, so variable-length, packed,
multimodal, and sparse batches are counted correctly.

Store the result with the forward outputs and accumulate it across gradient
accumulation micro-steps. For example, the `basic_vlm` recipe uses a `model_flops` metric accumulator with
`accumulate="sum"` and carries `model_flops_per_step` to post-step logging.

### 4. Resolve Distributed Scope

Default to global optimizer-step MFU. Use the same boundary for FLOPs, tokens,
and timing:

```text
one optimizer step after all gradient accumulation micro-steps
```

Rules:

- include all accumulated micro-batches in FLOPs and token counters;
- sum logical model FLOPs across ranks that process different samples;
- use data-parallel size for global token counts;
- include all active training devices in the hardware denominator;
- reduce step time with `max` across ranks for synchronous training;
- treat communication as runtime overhead, not model FLOPs.

Read `references/distributed_mfu.md` for DDP, FSDP, TP, PP, EP, hybrid, FSDP2
`replicate * shard`, and effective precision rules.

### 5. Resolve Hardware Peak

Match `references/hardware_peak_flops.csv` using:

- normalized active GPU name;
- effective compute precision.

Effective precision must come from the actual compute path, not only declared
config precision:

- use bf16/fp16 peak when the main matmul operands are actually bf16/fp16
  through FSDP param dtype, AMP/autocast, model weights, or activation casts;
- use TF32 peak for fp32 matmul when TF32 is active;
- use pure fp32 peak only when TF32 is disabled;
- do not infer low-precision compute from output dtype alone unless the wrapper
  guarantees output dtype also controls matmul input or parameter dtype.

If multiple hardware rows match, search the internet, ask the user, or require
explicit config. Do not silently choose the higher peak.

Expected CSV fields:

```text
device_name_pattern, precision, peak_tflops
```

### 6. Compute And Log MFU

Reuse existing optimizer-step timing. If measuring raw CUDA work directly,
synchronize before and after the measured region; otherwise avoid adding extra
synchronization to the training loop.

Build a small recipe-local logging helper when possible:

- validate positive `model_flops_per_step`, `step_time_seconds`, peak FLOPs, and
  device count;
- reduce local FLOPs and step time to global scope;
- compute `mfu = model_flops_per_step / step_time / total_peak_flops`;
- return a log payload instead of writing directly when the engine already owns
  logging.

Required metric:

```python
logs["perf/mfu"] = float(mfu)
```

## Validation

### Soft Validation

- model FLOPs are injected on the real runtime model path;
- per-micro-batch FLOPs use actual batch metadata such as masks, sequence
  lengths, visual grids, freeze state, or routing counts;
- distributed scope matches `references/distributed_mfu.md`;
- hardware peak uses real GPU name and effective precision;
- standard MFU excludes optimizer/data/IO/communication and activation
  checkpoint recompute;
- CPU-only, formula-only, or skipped checks are not reported as completed MFU
  validation.

### Hard Validation

Copy and adapt `references/asserts.py` into:

```text
recipes/<recipe>/tests/skills/model-flops-utilization/asserts.py
```

The hard validation requires real GPU training logs. CPU-only tests may catch
wiring mistakes, but they do not validate the reported MFU.

## Output

- State where `calculate_model_flops(...)` is injected.
- State where per-step FLOPs are accumulated and where `perf/mfu` is logged.
- State GPU, effective precision, peak TFLOPs source, and distributed topology.
- Report soft validation and hard validation status.

## Read On Demand

- `references/asserts.py`: recipe-local hard-validation assertion template.
- `references/flops_formulas.md`: dense, VLM, MoE, sparse, packed, and freeze FLOPs rules.
- `references/distributed_mfu.md`: distributed scope and effective precision rules.
- `references/hardware_peak_flops.csv`: known hardware peak FLOPs table.
