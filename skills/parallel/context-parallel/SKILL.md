---
name: context-parallel
description: Add, review, update, and validate recipe-local context-parallel
  training using mvp-engine's context mesh, Ulysses attention, CPKit sequence
  specs, model-family CP helpers, and explicit CP gradient synchronization.
---

# Context Parallel

## Goal

Add context parallelism to a recipe while preserving the non-CP training
semantics.

Use this skill as the route map:

- keep the step order and routing rules in this file;
- put stable runtime facts in `references/runtime_contract.md`;
- put optional failure-mechanism contracts in mechanism references;
- put test-layer policy in `references/validation_system.md`;
- put executable public-test helpers in `references/asserts.py`;
- keep model-family details recipe-local or mechanism-local.

Do not add one reference file per model. Add or use references by mechanism:
packed attention topology, VLM/media ownership, custom attention dispatch,
auxiliary hidden layout, gradient sync order, and runtime parity.

## File Roles

- `SKILL.md`: workflow, routing, and output expectations only.
- `references/runtime_contract.md`: stable CP runtime and kit contracts that
  apply before any model-family pattern.
- `references/*_*.md`: optional mechanism contracts; read only when that
  mechanism appears in the recipe.
- `references/validation_system.md`: test-layer boundaries and public
  validation workflow.
- `references/asserts.py`: canonical executable assertion template copied into
  recipe-local tests.

## Required Inputs

Identify these before editing:

- target recipe, config, engine, and model patch files;
- target `parallel.mesh.context`;
- active `shard` and `tensor` mesh settings;
- top-level model class instantiated by training;
- attention runtime class names and Q/K/V layout (`BHSD` or `BSHD`);
- dense sequence fields and their split dimension;
- model-family fields such as image/video/audio rows and their split unit;
- how labels, `shift_labels`, `position_ids`, `pack_segment_ids`, and
  `cu_seq_lens*` are built;
- optimizer-step order: unscale, token/global rescale, CP sync, clip, step;
- available runtime validation resources.

Ask only when the target model, CP size, or validation hardware cannot be
derived from the repository or user request.

## Workflow

### 1. Install Public Validation

Before large code edits, create the recipe-local validation surface:

- copy `references/asserts.py` to
  `recipes/<recipe>/tests/skills/context-parallel/asserts.py` if missing;
- keep that copied file as the canonical recipe-local CP assertion surface;
- add `tests/test_contract.py` from `tests/templates/test_contract.py.template`
  when semantic CP checks are needed;
- fill recipe-local knobs in assertions, such as attention class names or
  model-family field names;
- enable the mechanism-specific assertion hooks named by the references you
  read;
- run the public tests on the baseline and confirm they fail for missing CP
  behavior.

Public validation should check invariants, not expected diffs from a demo.

### 2. Route References

Read only the references that match the recipe:

- packed or varlen attention metadata:
  `references/packed_attention_topology.md`;
- VLM, audio, or other model-family media fields:
  `references/vlm_media_ownership.md`;
- custom attention wrappers or external attention dispatch:
  `references/custom_attention_dispatch.md`;
- auxiliary hidden tensors entering the LLM:
  `references/auxiliary_hidden_layout.md`;
- custom optimizer step, gradient scaling, or clipping:
  `references/gradient_sync_order.md`;
- real distributed correctness or impact claim:
  `references/runtime_parity_validation.md`;
- stable runtime and kit contracts:
  `references/runtime_contract.md`;
- validation layer mechanics:
  `references/validation_system.md`.

If a detail applies only to one recipe, keep it in that recipe. If it applies to
a model family, encode it as a conditional pattern or recipe-local assertion
knob. Promote only cross-recipe CP invariants to this file or
`references/asserts.py`.

### 3. Configure Mesh And Backend

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
- `tp.builtin_sequence_parallel=true` is not compatible with active CP.
- Context ranks must receive the same logical samples.
- Dataloader sharding and global batch accounting must exclude `context` and
  `tensor`.
- For BF16/FP16 training, prefer `cp.grad_reduce_dtype: float32`.

### 4. Bind Attention Metadata

Bind CP metadata on the top-level model class that training instantiates:

```python
class <TopModelClass>(...):
    CP_MODULE_CONFIG = {
        "<AttentionRuntimeClass>": {"qkv_layout": "BHSD"},
    }
```

`parallelize_model(...)` reads this metadata when CP is active and routes
matching attention modules through Ulysses. If the installed model wraps or
overrides attention dispatch, also validate the executable dispatch path with
`references/custom_attention_dispatch.md`.

### 5. Prepare Context-Local Batches

In recipe `train_pre_step`:

1. Build one explicit `CPSequenceSpec` list for dense token and model-family
   fields.
2. Pad global dense fields.
3. Build global packed metadata such as `position_ids` and `cu_seq_lens*`.
4. Slice dense fields to the local context rank.

Example:

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

Keep topology metadata global when the attention backend expects global packed
boundaries. Slice dense token/media tensors. For model-family fields whose valid
split unit is larger than one element, set `pad_scale` explicitly.

### 6. Patch Model-Family Forward Logic

Use `CPKit` for generic sequence operations. Add a model-family helper only when
the model needs reusable local-index or media metadata logic.

The model patch contract:

- it receives already-local token and model-family tensors from
  `train_pre_step`;
- it does not silently rerun global slicing inside model forward;
- local media fields are consumed by the media encoder or documented as
  native-local;
- temporary full-sequence views are used only where placeholder/media merge
  needs them;
- the LLM enters with local sequence/full hidden layout and matching auxiliary
  tensors.

Use CPKit transforms deliberately:

- `gather_sequence(...)` for token-aligned global masks or placeholder matching;
- `gather_seq_scatter_hidden(...)` for full sequence/hidden-sharded merge work;
- `scatter_seq_gather_hidden(...)` before returning to local sequence/full
  hidden LLM execution.

### 7. Sync CP Gradients

At synchronized optimizer steps:

```python
stats = self.token_loss_kit.reduce_window()
self.scaler.unscale_(self.optimizer)
self.token_loss_kit.rescale_grads(self.model.parameters(), stats)
if self.parallel_mesh.cp.active:
    sync_cp_grads(self.model)
clip_grad_norm_(self.model, max_grad_norm)
```

Order matters: unscale, token/global gradient rescale, CP gradient sync, clip,
optimizer step. Keep CP sync in recipe engines that actually run CP, not shared
base engines.

## Validation

Use the repository's layered test system:

- structure: files, imports, config, registry, and basic CP entrypoints;
- contract: cheap CP invariants such as specs, helper dataflow, bound patches,
  attention dispatch, and optimizer order;
- smoke: one small CP-on runtime path;
- parity/impact: real CP-off/CP-on loss, gradient, ownership, memory, or
  throughput artifacts.

If user or repository instructions provide accelerator/distributed resources,
attempt hard smoke/parity validation. A blocked or not-run hard validation is
unresolved, not a correctness pass.

Generate parity metrics non-invasively by default: use recipe-local runners,
smoke hooks, method wrappers, or generic observation surfaces before changing
production recipe engine/model code.

Use `references/validation_system.md` to map each mechanism reference to the
recipe knobs and assertion hooks that public tests should call.

## Output

Report:

- files changed;
- CP size and mesh config;
- attention classes covered by `CP_MODULE_CONFIG`;
- fields covered by `CPSequenceSpec`;
- model-family patch location and matched pattern references;
- where `sync_cp_grads(...)` runs;
- public validation commands and results;
- smoke/parity commands, artifacts, and results;
- unresolved blocked runtime validation, if any.

## Read On Demand

- `references/runtime_contract.md`: stable mesh, Ulysses, CPKit, and runtime
  contracts.
- `references/packed_attention_topology.md`: packed/varlen attention topology.
- `references/vlm_media_ownership.md`: dense model-family media ownership.
- `references/custom_attention_dispatch.md`: custom attention dispatch.
- `references/auxiliary_hidden_layout.md`: auxiliary hidden tensor layout.
- `references/gradient_sync_order.md`: optimizer and CP gradient sync order.
- `references/runtime_parity_validation.md`: hard runtime and parity validation.
- `references/validation_system.md`: validation layers, canonical assertions,
  and blocked hard-validation semantics.
- `references/asserts.py`: canonical recipe-local assertion template.
