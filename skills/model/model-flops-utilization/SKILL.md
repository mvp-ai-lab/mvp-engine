---
name: model-flops-utilization
description: add, review, and validate model flops utilization mfu support for mvp-engine recipes. use for flops estimation, runtime step timing, hardware peak lookup, effective precision, tokens/sec, distributed MFU, or perf/mfu logging.
---

# Model FLOPs Utilization

## Goal

Add reliable global optimizer-step MFU for the current recipe.

Preserve these requirements:

1. Inject `calculate_model_flops(...)` into the runtime model instance.
2. Compute MFU from synchronized runtime step time.
3. Resolve hardware peak from real GPU name and effective precision.
4. Log MFU under `perf/mfu`.
5. Add recipe-local skill tests under `recipes/<recipe>/skill_tests/model-flops-utilization/`.

Default metric:

```text
global_mfu =
  global_logical_train_flops_per_optimizer_step
  / synchronized_step_time_seconds
  / total_participating_gpu_peak_flops_per_second
```

Do not mix rank-local FLOPs with global hardware peak.

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
- active GPU name and matching `references/hardware_peak_flops.csv` row

Ask the user only when hardware, precision, launch method, or topology cannot be derived.

## Workflow

### 1. Locate Existing Hooks

Find:

- where the model is built
- where wrappers such as DDP/FSDP are applied
- where optimizer-step timing exists
- where metrics are assembled and logged
- how the recipe tests skill wiring

### 2. Add Model FLOPs Injection

Inject into the runtime model instance; do not replace the model class by default.

```python
import types


def inject_model_flops_calculation(model):
    def calculate_model_flops(self, *, batch_size: int, is_training: bool = True) -> float:
        ...

    model.calculate_model_flops = types.MethodType(calculate_model_flops, model)
    return model
```

Keep the signature architecture-specific. Return global logical train FLOPs per optimizer step unless clearly documented otherwise.

When the model is wrapped, inject into the logical underlying model when possible and expose a forwarding method if the engine calls the wrapper.

Read `references/flops_formulas.md` for dense formulas.
Read `references/vlm_mfu.md` for VLM, vision, projector, frozen, and dual-stream rules.
Read `references/moe_mfu.md` for MoE, MoD, sparse routing, expert parallel, and KV reuse rules.

### 3. Resolve Distributed Scope

Default to global optimizer-step MFU.

Use one boundary for FLOPs, tokens, and timing:

```text
one optimizer step after all gradient accumulation micro-steps
```

Rules:

- include all accumulated micro-batches in FLOPs and tokens
- use data-parallel size for global tokens, not raw world size unless all ranks are data parallel
- include all participating GPUs in hardware peak
- treat communication as runtime overhead, not model FLOPs

Read `references/distributed_mfu.md` for DDP, FSDP, TP, PP, EP, hybrid, FSDP2 `replicate * shard`, and effective precision rules.

### 4. Resolve Hardware Peak

Match `references/hardware_peak_flops.csv` using:

- normalized active GPU name
- effective precision

Effective precision must come from the actual compute path, not only `optim.mixed_precision`.

For FSDP2:

- inspect `parallel.backend_kwargs.fsdp2.mp_policy`
- if `param_dtype` or `output_dtype` is `bfloat16`, use bf16 peak
- if `param_dtype` or `output_dtype` is `float16`, use fp16 peak
- treat as pure fp32 only when wrapper policy is float32 and TF32 is disabled

For fp32:

- use TF32 peak when TF32 matmul is active
- use pure FP32 peak only when TF32 is disabled

If multiple hardware rows match, ask the user or require explicit config. Do not silently choose the higher peak.

Expected CSV fields:

```text
device_name_pattern, precision, peak_tflops
```

### 5. Compute And Log MFU

Reuse existing engine timing. If timing CUDA work directly, synchronize before and after the measured region.

Required calculation checks:

- `model_flops_per_step >= 0`
- `step_time_seconds > 0`
- `device_peak_tflops > 0`
- `num_training_gpus > 0`

Required metric:

```python
logs["perf/mfu"] = float(mfu)
```

Recommended debug metrics:

```text
perf/model_flops_per_step
perf/achieved_tflops
perf/peak_tflops
perf/step_time_seconds
perf/tokens_per_step
perf/tokens_per_second
perf/num_training_gpus
```

Use warmup steps before trusting stability. Suspicious MFU values, especially near 0 or above 1, must be traced through FLOPs, timing, precision, hardware peak, and topology. If formulas are correct but MFU is very low for a tiny model, explain the overhead-bound workload and suggest validating with larger batch, sequence length, hidden size, or layer count before changing MFU code.

## Validation

Required checks:

- injected `calculate_model_flops(...)` exists on the runtime path
- FLOPs count uses executed architecture components
- runtime step time is synchronized and uses the optimizer-step boundary
- hardware peak uses real GPU name and effective precision
- FSDP2 `mp_policy` is checked before selecting fp32, tf32, bf16, or fp16 peak
- global MFU does not combine local-rank FLOPs with global hardware peak
- tokens use data-parallel size, including FSDP2 shard ranks
- `perf/mfu` is logged as a float
- MFU validation must use real CUDA GPU training logs. CPU-only runs, formula-only tests, and skipped tests do not count as MFU validation.
- Before GPU validation, search the repository for GPU resource commands such as `srun`, `sbatch`, `gres=gpu`, `torchrun`, `nvidia-smi`, `CUDA`, or `GPU`. Use the repo command if found; otherwise ask the user for the GPU request command.
- If GPU access is unavailable, stop validation and state that CPU MFU validation is not supported.
- MFU sanity is explained from logs

## Recipe-Local Tests

Add recipe-local tests under `recipes/<recipe>/skill_tests/model-flops-utilization/`:

- `test_structure.py`: verify recipe structure and core wiring.
- `test_runtime.py`: build recipe runtime objects through recipe entrypoints.
- `test_smoke.py`: run one real recipe-owned training step and checkpoint/log path.
- `test_effectiveness.py`: create a recipe-local test that uses
  `mvp_engine.test.recipe_probe` helpers, then add a method such as
  `assert_mfu_metrics_match_formulas(log_text)`. 
  Parse real CUDA GPU training logs, require finite numeric values for `perf/mfu` and required `perf/*` metrics, enforce `perf/mfu in [0, 1]`, and verify the TFLOPs, MFU, and tokens/sec formulas. When real GPU logs contain valid metrics with matching formulas, the effectiveness test can be treated as passing.

Run skill validation through `python -m tests.test_skills --recipe <recipe> --skill model-flops-utilization`, following the repository fresh-subagent layer workflow. Run GPU validation with `loop.total_steps=1000`, `log.interval=10`, and `checkpoint.interval=10000`.

## Output

Summarize:

- injection location
- MFU calculation and logging location
- GPU, effective precision, peak TFLOPs, and topology
- validation commands and status

## Read On Demand

- `references/hardware_peak_flops.csv`: hardware peak table
- `references/flops_formulas.md`: dense model FLOPs formulas
- `references/distributed_mfu.md`: distributed and effective precision rules
- `references/vlm_mfu.md`: VLM and multimodal rules
- `references/moe_mfu.md`: MoE, MoD, sparse routing, and KV reuse rules
