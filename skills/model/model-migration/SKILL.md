---
name: model-migration
description: Migrate an external model into an mvp-engine recipe with strict behavior parity, strict checkpoint compatibility, and optional NPU support. Use when porting modeling and configuration code into recipes/.
---

# Model Migration

## Goal

- Port a source model into `recipes/<recipe>/model/` without changing its math or parameter naming.
- Preserve checkpoint compatibility so existing weights load with `strict=True` and zero key mismatches.
- Add recipe-local parity tests instead of relying on global `tests/`.

## Required Inputs

- Source `modeling_*.py`, `configuration_*.py`, and the checkpoint files to preserve compatibility with.
- The target recipe path under `recipes/<recipe>/`.
- The runtime entrypoint or builder that will instantiate the migrated model.
- Whether an NPU-specific variant is required.
- An environment that can run parity tests and strict checkpoint-load checks.

## Workflow

### 1. Locate and fingerprint source assets

- Find the source modeling file, configuration file, and checkpoint directory before editing anything.
- Verify source identity with hashes and diffs when more than one candidate copy exists.

```bash
sha256sum SOURCE_MODEL.py CHECKPOINT_DIR/modeling_*.py TARGET_MODEL.py
diff -u SOURCE_MODEL.py CHECKPOINT_DIR/modeling_*.py
```

- If two copies are byte-identical, treat that source implementation as authoritative.

### 2. Port the CPU or GPU implementation first

- Copy the source `configuration_*.py` and `modeling_*.py` into the target recipe with the smallest possible integration delta.
- Keep module names, class names, and parameter names unchanged unless a recipe integration boundary forces a small adjustment.
- Update `__init__.py` exports and builders only after the base model compiles.
- Do not refactor while migrating; integrate first, then optimize later if the user asks.

### 3. Add an NPU variant only when needed

- If the user wants NPU support, start from the CPU or GPU implementation and create `modeling_*_npu.py` as a near-copy.
- Keep NPU-only substitutions isolated to the smallest possible blocks, such as fused rotary, norm, or attention ops.
- Preserve parameter names and module structure so one checkpoint can load into both implementations.
- Prefer a `torch_npu` import with fallback and use NPU fused ops only when tensors are actually on NPU.
- Keep an exact non-NPU fallback math path.

### 4. Add recipe-local parity tests

Create recipe-local assertions in:
- `recipes/<recipe>/skill_tests/model-migration/asserts.py`

Add:
- `skill_tests/test_structure.py`: verify recipe structure and model-migration wiring.
- `skill_tests/test_smoke.py`: run one real recipe-owned training step and checkpoint/log path.
- `asserts.py`: expose the standard `assert_structure(...)` and `assert_smoke(...)` hooks.

If the environment allows, run tests on both CPU/GPU and NPU devices to validate parity across implementations.

### 5. Validate checkpoint compatibility

- Run strict `load_state_dict(..., strict=True)` checks on every migrated class.
- Run a `from_pretrained(...)` smoke test against the checkpoint directory.
- If strict load fails, diff the model's `state_dict().keys()` against checkpoint keys and fix naming or structure mismatches in the migrated model before considering checkpoint rewrites.

### 6. Stop only after the acceptance bar is met

- Do not stop at a compiling port.
- Ship only after behavior parity, checkpoint compatibility, and recipe-local validation all pass or any remaining gap is clearly documented.

## Validation

- Source modeling and configuration identity was verified, or any intentional deviation is documented.
- Parity tests exist under the target recipe and pass in the available environment.
- Strict checkpoint load passes with zero missing or unexpected keys.
- `from_pretrained(...)` succeeds for the migrated class.
- Lint or targeted checks covering the migrated files were run.

## Output

- State which files were migrated or newly created.
- State whether an NPU variant was added.
- Summarize parity and strict-load validation results.
- Call out any unresolved environment gap, such as missing NPU hardware for full parity validation.

## Useful Commands

```bash
# run recipe-local tests in a fresh subagent
python -m tests.test_skills --recipe <recipe> --skill model-migration

# lint migration files
uv run --with ruff ruff check recipes/<recipe>/model recipes/<recipe>/skill_tests/model-migration

# inspect changed files
git status --short --untracked-files=all
```

## Read On Demand

- This skill has no bundled reference files. Read the source modeling and configuration files directly, and consult the Ascend PyTorch for NPU documentation only when you need NPU-specific operator substitutions.
