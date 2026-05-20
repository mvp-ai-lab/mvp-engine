---
name: model-migration
description: Add, review, update, and validate migration of external model/configuration code into mvp-engine recipes with strict behavior parity, checkpoint compatibility, and optional NPU variant support.
---

# Model Migration

## Goal

Migrate an external model into `recipes/<recipe>/model/` without changing model
behavior:

- preserve architecture math, tensor shapes, parameter names, and checkpoint
  keys;
- keep source identity and any intentional recipe integration deltas auditable;
- validate strict checkpoint loading and behavior parity through recipe-local
  tests;
- add NPU-specific variants only when requested and keep them checkpoint
  compatible with the base implementation.

## Required Inputs

Identify these before editing:

- authoritative source `modeling_*.py` and `configuration_*.py` files;
- checkpoint directory or state dict whose keys must remain compatible;
- target recipe path and model builder/runtime entrypoint;
- expected model class names, config class names, and public exports;
- tokenizer/processor or input fixtures needed for parity checks;
- whether an NPU-specific implementation is required;
- recipe-local `tests/test_structure.py` and `tests/test_smoke.py`.

Ask the user only if the authoritative source, checkpoint, or required parity
inputs cannot be derived from local files.

## Workflow

### 1. Locate And Fingerprint Sources

Find the source assets before editing:

```bash
rg -n "class .*Config|class .*Model|from_pretrained|load_state_dict" SOURCE_DIR recipes/<recipe>
```

Record source identity with hashes and diffs when more than one copy exists:

```bash
sha256sum SOURCE_MODEL.py SOURCE_CONFIG.py
diff -u SOURCE_MODEL.py CHECKPOINT_DIR/modeling_*.py
```

If two copies differ, stop and decide which one is authoritative before porting.

### 2. Port The Base Implementation

Copy or adapt the CPU/GPU implementation first:

- keep class names, module structure, tensor operations, and parameter names
  unchanged unless a recipe boundary forces a documented adapter;
- keep configuration fields and defaults compatible with the source;
- add `__init__.py` exports and recipe builders only after imported classes work;
- avoid refactors, renames, formatting churn, and performance rewrites during
  migration.

Read `references/checkpoint_parity.md` before changing parameter names, config
fields, loading paths, or parity tests.

### 3. Wire The Recipe Entrypoint

Connect the migrated model to the recipe:

- add a recipe-local builder that instantiates the migrated config/model;
- keep checkpoint loading explicit and strict where possible;
- keep tokenizer/processor integration outside the model unless the source model
  owns it;
- preserve model outputs consumed by the engine, such as `loss`, `logits`, or
  named output dataclasses.

### 4. Add NPU Variant Only When Requested

If an NPU path is required, start from the migrated base implementation:

- create a near-copy such as `modeling_<name>_npu.py` only for NPU-specific ops;
- isolate substitutions to the smallest blocks, such as attention, rotary, norm,
  or fused kernels;
- keep state dict keys and public class behavior compatible with the base model;
- keep a non-NPU fallback path when importing or testing without `torch_npu`.

Do not add an NPU variant just because one might be useful later.

### 5. Add Recipe-Local Parity Tests

Add tests under:

```text
recipes/<recipe>/tests/skills/model-migration/
```

Use `asserts.py` for structure/smoke hooks. Add optional impact tests only when
structure and smoke cannot validate the migration's critical correctness
property. Name each file after the invariant it verifies, such as
`test_behavior_parity.py`, `test_checkpoint_compatibility.py`, or
`test_npu_parity.py`.

## Validation

### Soft Validation

Review the modified recipe without running tests:

- source modeling/config identity is recorded or intentional deltas are
  documented;
- migrated classes, config fields, parameter names, and output semantics match
  the source;
- recipe builders and exports instantiate the migrated implementation, not a toy
  replacement;
- checkpoint loading is strict or any unavoidable mismatch is documented with a
  concrete reason;
- parity tests use real source and migrated classes with deterministic inputs;
- NPU variants, if added, preserve state dict compatibility and have CPU/GPU
  fallbacks for non-NPU environments;
- no repo-wide model migration helper was added to `mvp_engine/`;
- import-only or CPU-only checks are not reported as completed checkpoint or NPU
  parity validation.

### Hard Validation

Copy and adapt `references/asserts.py` into:

```text
recipes/<recipe>/tests/skills/model-migration/asserts.py
```

Ensure the recipe has `tests/test_structure.py` and `tests/test_smoke.py`; use
`tests/templates/` if missing.

Run in fresh subagents, in order, stopping on first failure:

```bash
pytest recipes/<recipe>/tests/test_structure.py -q
pytest recipes/<recipe>/tests/test_smoke.py -q
```

If impact tests are added for parity or strict checkpoint loading, run each one
after smoke validation:

```bash
pytest recipes/<recipe>/tests/skills/model-migration/test_behavior_parity.py -q
pytest recipes/<recipe>/tests/skills/model-migration/test_checkpoint_compatibility.py -q
```

## Output

- State source files, target files, and any intentional migration deltas.
- State whether an NPU variant was added.
- State checkpoint compatibility and parity validation results.
- Report soft validation and hard validation status.
- Call out any remaining gap, such as missing checkpoint files or no NPU
  environment for NPU parity.

## Read On Demand

- `references/asserts.py`: recipe-local hard-validation assertion template.
- `references/checkpoint_parity.md`: strict load, key diffing, and parity-test
  patterns.
