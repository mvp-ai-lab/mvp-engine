---
name: model-migration
description: Migrate an external model into an mvp-engine recipe with exact math/output parity, optional NPU support, and strict checkpoint compatibility checks. Use when porting modeling/configuration code, validating state_dict key alignment, and adding recipe-local parity tests.
---

# Model Migration

## Goal

Port a source model into `recipes/<recipe>/model/` without changing behavior.

Enforce all of the following:
- Keep the same math and parameter naming.
- Produce identical outputs for identical inputs and weights.
- Load existing checkpoints with `strict=True` and zero key mismatches.
- Place migration tests under the recipe folder, not global `tests/`.

## Workflow

### 1. Locate and fingerprint source assets

- Find source `modeling_*.py`, `configuration_*.py`, and checkpoint files (`.safetensors`/`.bin`).
- Verify identity with hash and diff before porting.

```bash
sha256sum SOURCE_MODEL.py CHECKPOINT_DIR/modeling_*.py TARGET_MODEL.py
diff -u SOURCE_MODEL.py CHECKPOINT_DIR/modeling_*.py
```

If files are byte-identical, treat source model code as authoritative.

### 2. Port CPU/GPU model first

- Migrate source `configuration_*.py` and `modeling_*.py` into the recipe model folder according to user's requirement.
- Keep module/class names and parameter names unchanged unless there is a required recipe integration change.
- Update `__init__.py` and builder exports only after base model compiles.

Hard rule: avoid refactors during migration. Do minimal edits needed for integration.

### 3. Add NPU variant with minimal delta (Optional)

- You can ask user if they want NPU support during migration, but do not require it.
- Create `modeling_*_npu.py` as a near-copy of the CPU/GPU file.
- Apply NPU-only substitutions in small, isolated blocks (for example fused rotary/norm/attention ops).
- Keep parameter names and module structure aligned with CPU/GPU implementation so one checkpoint can load into both.
- The PyTorch for NPU's documents can be found here: https://www.hiascend.com/document/detail/zh/Pytorch/730/index/index.html

Recommended pattern:
- Import `torch_npu` with fallback.
- Use NPU fused op only when tensor device is NPU.
- Keep exact fallback math path for non-NPU devices.

### 4. Add recipe-local parity tests

Create tests in:
- `recipes/<recipe>/skill_tests/model-migration/`

Add:
- `test_spec.yaml`: declare the required test layers for this applied skill.
- `test_structure.py`: at least verify recipe import, registry wiring, config
  schema validation, required slots, and logger/checkpoint hooks; it must also
  verify the migrated recipe entrypoints and migrated model class wiring exist.
- `test_runtime.py`: at least build dataset, collator, model, optimizer,
  scheduler, and engine successfully without starting training; it must also
  verify parity-critical runtime paths are reachable through the migrated recipe
  entrypoints.
- `test_smoke.py`: cover one real recipe-owned single step: forward, loss,
  backward, optimizer step, logger write, and checkpoint noop or temporary
  save; it must also verify source-vs-migrated parity and strict checkpoint-load
  coverage through the migrated recipe entrypoints.
- `test_smoke.py` must use the full real capability path for this skill: real
  migrated recipe entrypoints, real parity checks, and real checkpoint-load /
  logger / checkpoint wiring. Do not short-circuit it with monkeypatch-based
  fake migrated models, fake load paths, or similar test-only stand-ins.
- If the recipe's full-capability single step only makes sense on GPU, NPU, or
  distributed hardware, write the smoke test as a real launcher-driven smoke
  test and set `gpu_preferred: true` in `test_spec.yaml`; do not degrade it
  into fake logic just to make it run in a weaker environment.

Include at least:
- Source vs migrated model parity on all supported inputs.
- CPU/GPU class vs NPU-class (fallback path) parity with shared weights.
- strict checkpoint-load coverage through the migrated recipe entrypoints.

When executing this skill for a user recipe, add these tests automatically. Do not
require the user to ask for the test layout separately. If execution is blocked by
device availability or permissions, return the exact `python -m tests.test_skills` command
and any required environment-specific launch command.

If the environment allows, run tests on both CPU/GPU and NPU devices to validate parity across implementations.

Parity assertion standard:
- Use `torch.equal` for strict identity when required.

### 5. Validate checkpoint compatibility

Run strict load tests for both classes.

```python
state = load_file(".../model.safetensors")
res = model.load_state_dict(state, strict=True)
assert len(res.missing_keys) == 0
assert len(res.unexpected_keys) == 0
```

Also run:
- `ModelClass.from_pretrained(<checkpoint_dir>)` smoke test.

If strict load fails:
- Diff `state_dict().keys()` and checkpoint keys.
- Fix naming/structure mismatch in migrated model (do not mutate checkpoint unless absolutely required).

### 6. Final acceptance checklist

Ship only when all pass:
- Source modeling/config identity confirmed (or justified deviations documented).
- Parity tests pass.
- Strict checkpoint load passes for migrated and NPU classes.
- No missing or unexpected keys.
- Lint/tests pass.

## Commands used often

```bash
# run recipe-local tests
python -m tests.test_skills --recipe <recipe> --skill model-migration

# lint migration files
uv run --with ruff ruff check recipes/<recipe>/model recipes/<recipe>/skill_tests/model-migration

# inspect changed files
git status --short --untracked-files=all
```
