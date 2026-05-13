# Distributed MFU

## Use When

Use this reference when MFU is measured under DDP, FSDP, tensor parallel,
pipeline parallel, expert parallel, hybrid parallelism, or gradient
accumulation.

## Global MFU Definition

Default to global optimizer-step MFU:

```text
global_mfu =
  global_logical_train_flops_per_optimizer_step
  / synchronized_optimizer_step_time_seconds
  / total_participating_gpu_peak_flops_per_second
```

Use one boundary for FLOPs, tokens, and step time. The default boundary is one
optimizer step after all gradient accumulation micro-steps.

## Token and Step Boundary Rules

- Tokens per optimizer step:
  `micro_batch_size * effective_sequence_length * grad_accum_steps * data_parallel_size`.
- Use actual token counts when batches are packed, variable-length, or masked.
- Do not use raw `world_size` as the data multiplier unless all ranks are data
  parallel ranks.
- For asynchronous CUDA work, synchronize before and after the measured region.
- Pipeline bubbles, collectives, and communication reduce MFU through wall-clock
  step time, not by increasing model FLOPs.

## Effective Precision Rules

- Hardware peak must match effective compute precision, not only declared config precision.
- In `mvp_engine` FSDP2, check `parallel.backend_kwargs.fsdp2.mp_policy`.
- If FSDP2 `param_dtype` or `output_dtype` is `bfloat16` or `float16`, use bf16 or fp16 peak.
- Treat a run as pure fp32 only when wrapper policy is float32 and TF32 is disabled.

## Parallelism Rules

DDP:

- Count logical full-model FLOPs over global data-parallel tokens.
- Include all DDP ranks in total GPU peak FLOPs.
- Treat gradient all-reduce as runtime overhead.
- If DDP fails with unused parameters, set `ddp.find_unused_parameters=True` and rerun.

FSDP:

- Count logical full-model FLOPs, not shard-local parameter FLOPs.
- Do not divide model FLOPs by shard count.
- In `mvp_engine` mesh terms `(replicate, shard, tensor)`, use
  `data_parallel_size = replicate * shard` for FSDP global MFU. Do not use
  `replicate` alone.
- Include all FSDP ranks in total GPU peak FLOPs.
- Treat parameter all-gather and reduce-scatter as runtime overhead.

Tensor parallel:

- For global MFU, do not divide logical model FLOPs by `tensor_parallel_size`.
- Do not multiply tokens by `tensor_parallel_size`.
- Include tensor-parallel GPUs in total peak FLOPs.
- Treat tensor-parallel collectives as runtime overhead.

Pipeline parallel:

- Count the full logical model over global data-parallel tokens.
- Do not multiply tokens by `pipeline_parallel_size`.
- Include pipeline ranks in total peak FLOPs.
- Use steady-state optimizer-step timing when available.

Expert parallel:

- Count activated expert compute, not all owned experts.
- Do not multiply tokens by `expert_parallel_size` unless expert parallelism
  also changes the number of input samples.
- Include expert-parallel GPUs in total peak FLOPs.

## Common Mistakes

- Mixing local-rank FLOPs with global hardware peak.
- Multiplying tokens by `world_size` in TP/PP/EP setups.
- Estimating FSDP FLOPs from local shard parameters.
- Dropping the `shard` mesh dimension from FSDP `data_parallel_size`.
- Selecting fp32 peak while FSDP2 `mp_policy` still uses bf16.
- Reporting pure fp32 MFU without checking TF32 state.
- Dividing global logical FLOPs by TP size.
- Reporting micro-step MFU as optimizer-step MFU.
