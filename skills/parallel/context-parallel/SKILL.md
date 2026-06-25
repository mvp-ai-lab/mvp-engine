---
name: context-parallel
description: Add, review, update, and validate recipe-local context-parallel
  training using mvp-engine's context mesh, Ulysses attention, CPKit sequence
  specs, model-family CP helpers, and explicit CP gradient synchronization.
---

# Context Parallel

## Goal

Add context parallelism to long-sequence or multimodal training while preserving
the non-CP model semantics.

The skill should guide changes that:

- split sequence work across context ranks;
- keep each context group on the same logical samples;
- keep model-family data layout rules explicit in the recipe;
- preserve token-normalized loss and gradients;
- validate that CP-off and CP-on produce close loss and gradient results on the
  same deterministic input.

## Required Inputs

Before editing, identify:

- target recipe and model patch files;
- top-level model class used by training;
- attention module class names and Q/K/V layout, usually `BHSD` for HF modules;
- hidden-state sequence dimension;
- how `position_ids`, packed `cu_seq_lens_*`, masks, and labels are produced;
- which batch fields are dense sequence fields and which dimension each uses;
- optional model-family fields that need custom sequence rules, such as media
  tensors;
- intended `parallel.mesh.context`;
- active `shard` and `tensor` mesh settings;
- where optimizer unscale, token-loss grad rescale, clipping, and step happen.

Ask only if the target model, CP size, or validation hardware cannot be derived.

## Workflow

### 1. Locate The Existing Boundaries

Search first:

```bash
rg -n "CP_MODULE_CONFIG|parallelize_model|backend_kwargs.*cp|parallel.mesh" recipes/<recipe>
rg -n "q_proj|k_proj|v_proj|o_proj|flash_attention|attention" recipes/<recipe>
rg -n "train_pre_step|prepare_packed_model_inputs|position_ids|cu_seq_lens" recipes/<recipe>
rg -n "optimizer_step|rescale_grads|sync_cp_grads|clip_grad" recipes/<recipe>
```

Find the real top-level model class, attention modules, batch-preparation path,
and optimizer-step ordering. Do not add recipe behavior to shared engines.

### 2. Configure Mesh And Backend

Use a context mesh dimension plus FSDP2 shard dimension:

```yaml
parallel:
  mesh:
    shard: <fsdp2 shard size>
    context: <cp size>
    tensor: <tp size>
  backend_kwargs:
    cp:
      implementation: ulysses
      attn_implementation: flash_attention_2
      grad_sync: true
      grad_reduce_dtype: float32
```

Rules:

- `context > 1` activates CP.
- `shard > 1` is required because pure model parallelism is rejected.
- `tp.builtin_sequence_parallel=true` is not compatible with `context > 1`.
- Ranks that differ only by `context` must receive the same logical samples.
- Dataloader sharding and global batch accounting must exclude `context` and
  `tensor`.
- For BF16/FP16 training, prefer `cp.grad_reduce_dtype: float32`.

Runtime contract:

- `parallelize_model(...)` gets `parallel_mesh.cp.mesh` and calls
  `parallelize_model_with_context_parallel(model, cp_mesh, cp_config)` when CP
  is active.
- When `cp.grad_sync=true`, runtime attaches `_cp_grad_sync`; the recipe engine
  still calls `sync_cp_grads(model)` explicitly.

### 3. Bind Attention Metadata

Bind CP metadata on the model class that training actually instantiates:

```python
class <TopModelClass>(...):
    CP_MODULE_CONFIG = {
        "<AttentionRuntimeClass>": {"qkv_layout": "BHSD"},
    }
```

Supported layouts are `BSHD` and `BHSD`. The runtime registers `ulysses_sp` and
switches matched attention modules to it. Keep implementation names canonical;
do not add broad aliases.

### 4. Prepare Ready Local Batches

In recipe `train_pre_step`, build one explicit `CPSequenceSpec` list for all
dense sequence-like batch fields, pad the global dense fields, build model
metadata on the padded global batch, then slice dense fields to the local
context rank:

```python
cp_sequence_specs = [
    CPSequenceSpec("input_ids", dim=1, pad_value=pad_token_id),
    CPSequenceSpec("attention_mask", dim=1, pad_value=0),
    CPSequenceSpec("labels", dim=1, pad_value=-100),
    CPSequenceSpec("shift_labels", dim=1, pad_value=-100),
    CPSequenceSpec("pack_segment_ids", dim=1, pad_value=0),
    CPSequenceSpec("position_ids", dim=2, pad_value=0),
]

batch = self.cp_kit.pad_sequence_batch(batch, cp_sequence_specs)
batch = prepare_packed_model_inputs(...)
batch = self.cp_kit.slice_sequence_batch(batch, cp_sequence_specs)
```

Notes:

- DataKit should produce global, segment-safe `shift_labels`.
- Build global packed `position_ids` and `cu_seq_lens_*` before slicing.
- Keep `cu_seq_lens_*` global topology metadata; slice only dense token/media
  tensors.
- Do not pass prebuilt multi-dimensional attention masks into
  `CPKit.slice_sequence_batch(...)`.
- `position_ids` does not exist before `prepare_packed_model_inputs(...)`, so
  `pad_sequence_batch(...)` skips it naturally.
- For model-family fields whose smallest valid split unit is larger than one
  element, set `pad_scale` explicitly in that field's `CPSequenceSpec`.

Example, not a requirement: Qwen-VL raw visual patches should stay aligned to
merged visual token boundaries:

```python
spatial_merge_size = int(self.unwrapped_model.config.vision_config.spatial_merge_size)
visual_pad_scale = spatial_merge_size**2
cp_sequence_specs.extend(
    [
        CPSequenceSpec("pixel_values", dim=0, pad_value=0, pad_scale=visual_pad_scale),
        CPSequenceSpec("pixel_values_videos", dim=0, pad_value=0, pad_scale=visual_pad_scale),
    ]
)
```

### 5. Patch Model-Family Forward Logic

Use the base `CPKit` for generic sequence operations. Add a small model-family
extension only when the model needs reusable metadata helpers:

```python
from functools import partial

from mvp_engine.kit import CPKit
from ..model import patch_<model_family>_context_parallel

self.cp_kit = CPKit(self.parallel_mesh)
model_patches.append(partial(patch_<model_family>_context_parallel, cp_kit=self.cp_kit))
```

Patch around the model's real dataflow, not around an abstract CP layer. The
normal contract is:

- the recipe has already converted dense sequence tensors into context-local
  chunks;
- the patched VLM/LLM forward receives local token tensors such as
  `[batch, local_seq]` `input_ids`, local `position_ids`, local shifted labels
  outside the model, and any local model-family tensors;
- global topology metadata, such as packed `cu_seq_lens_*`, may remain global
  if the attention implementation expects global packed sequence boundaries;
- the language model should enter its normal transformer stack with local
  sequence / full hidden tensors, for example `[batch, local_seq, hidden]`.

For text-only LLMs, the model patch is usually small:

- ensure the model receives local `input_ids` or local `inputs_embeds`;
- ensure RoPE/position ids are global-position local slices, not restarted from
  zero on each context rank;
- pass packed `cu_seq_lens_*` through unchanged when the attention backend uses
  global packed metadata;
- let CP attention handle the per-layer Ulysses all-to-all.

For VLMs, split the patch by component:

- **VLM entry:** accept local text tokens and local media tensors from
  `train_pre_step`; do not rerun global slicing inside the model.
- **ViT/media encoder entry:** feed only the local media rows owned by this
  context rank. For models with merge groups or tubelets, local media rows must
  be cut on the model's valid media-token boundary.
- **Media metadata:** derive local media position or interpolation metadata from
  global metadata plus the local media indices. Keep model constants, such as
  merge size, in the recipe/model patch.
- **Feature merge point:** if text placeholders need a global sequence view,
  convert local sequence/full hidden to full sequence/hidden shard with
  `gather_seq_scatter_hidden(...)`, insert or scatter media features there, then
  convert back with `scatter_seq_gather_hidden(...)`.
- **LLM entry:** enter the language model with local sequence/full hidden,
  local position ids, and any local auxiliary tensors expected by the
  transformer stack.

Use the CPKit transforms deliberately:

- `gather_sequence(...)`: gather a token-aligned local sequence when matching
  global placeholder positions or masks.
- `gather_seq_scatter_hidden(...)`: convert local sequence/full hidden to full
  sequence/hidden shard when a merge operation needs full sequence positions
  without materializing full hidden on every rank.
- `scatter_seq_gather_hidden(...)`: return full sequence/hidden shard tensors to
  local sequence/full hidden before normal LLM execution.

This structure avoids sequence holes. Each rank computes only its owned sequence
or media slice, but temporary full-sequence views are available exactly where
global placeholder matching or media insertion needs them.

Example, not a requirement: Qwen-VL can use
`QwenVLCPKit.local_visual_patch_indices(...)` to select local raw visual patches
aligned by `pad_scale=spatial_merge_size**2`. The ViT receives local
`pixel_values`; visual features are gathered into a full-sequence/hidden-sharded
layout for placeholder insertion; the final LLM input is scattered back to
local sequence/full hidden. The base `CPKit` still must not learn Qwen/VL field
names.

### 6. Sync CP Gradients

At synchronized optimizer steps:

```python
stats = self.token_loss_kit.reduce_window()
self.scaler.unscale_(self.optimizer)
self.token_loss_kit.rescale_grads(self.model.parameters(), stats)
if self.parallel_mesh.cp.active:
    sync_cp_grads(self.model)
clip_grad_norm_(self.model, max_grad_norm)
```

The order matters: unscale, token/global gradient rescale, CP grad sync, clip,
optimizer step. Do not put `sync_cp_grads(...)` in shared base engines that do
not always run CP.

## Validation

### Soft Validation

Review the diff and confirm:

- `parallel.mesh.context` and `backend_kwargs.cp` are present only where CP is
  intended;
- `tp.builtin_sequence_parallel` is disabled when CP is active;
- context ranks read identical samples;
- DataKit provides `shift_labels` before CP slicing;
- `train_pre_step` pads, builds packed metadata, then slices with one
  `CPSequenceSpec` list;
- non-token sequence fields use the right `dim`, `pad_value`, and `pad_scale`;
- CP attention module class names and QKV layout match the actual model;
- optimizer step calls `rescale_grads(...)` before `sync_cp_grads(...)`, and
  `sync_cp_grads(...)` before clipping.

### Numerical Validation

Validate CP by comparing CP-off and CP-on on the same deterministic input.

Required comparison:

- build two models from identical weights;
- run the same batch with `parallel.mesh.context=1` and `parallel.mesh.context=N`;
- use the same dtype, attention backend, seeds, dropout setting, and packed
  metadata path;
- compare global loss after token normalization;
- compare gradients after the CP path has called `rescale_grads(...)` and
  `sync_cp_grads(...)`;
- report max/mean absolute gradient differences for shared trainable parameters.

Expected result:

- FP32 should be very close, usually near numerical noise;
- BF16/FlashAttention can have small kernel-order drift;
- differences should be small enough to be explained by dtype/kernel reduction
  order, not by missing tokens, wrong labels, wrong position ids, or missing CP
  gradient sync.

If loss or gradients differ materially:

- verify all context ranks saw the same logical batch;
- verify `input_ids`, `shift_labels`, `pack_segment_ids`, and `position_ids`
  reconstruct the CP-off global tensors when gathered;
- verify model-family sequence fields are padded/sliced by their correct
  `pad_scale`;
- verify global `cu_seq_lens_*` still describe the padded global packed layout;
- verify `sync_cp_grads(...)` ran after gradient rescale and before clipping.

### Runtime Validation

Run the smallest practical structure/smoke checks for the target recipe. Also
run a focused parity script or job that prints:

```text
loss_cp_off
loss_cp_on
loss_abs_diff
grad_max_abs_diff
grad_mean_abs_diff
```

Save exact commands, dtype, world size, and tolerance rationale in the final
report. Do not claim parity unless the CP-on/CP-off comparison actually ran.

## Output

Report:

- files changed;
- CP size and mesh config;
- attention module classes covered by `CP_MODULE_CONFIG`;
- batch fields covered by `CPSequenceSpec`;
- where model-family CP patching happens;
- where `sync_cp_grads(...)` runs;
- CP-off vs CP-on loss difference;
- gradient max/mean absolute differences;
- any missing validation, runtime dependency, or unexplained numerical drift.

## Read On Demand

- `references/cp_rules.md`: lower-level mesh, Ulysses, packing, and gradient
  rules.
- `references/asserts.py`: optional recipe-local assertion template.
