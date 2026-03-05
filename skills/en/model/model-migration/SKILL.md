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
- `recipes/<recipe>/tests/`

Include at least:
- Source vs migrated model parity on all supported inputs.
- CPU/GPU class vs NPU-class (fallback path) parity with shared weights.

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
uv run --with pytest pytest -q recipes/<recipe>/tests/test_*migration*.py

# lint migration files
uv run --with ruff ruff check recipes/<recipe>/model recipes/<recipe>/tests

# inspect changed files
git status --short --untracked-files=all
```
