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
- Prefer copying `tests/test_structure_template.py`,
  `tests/test_runtime_template.py`, and `tests/test_smoke_template.py` into the
  recipe-local skill directory first, then only edit the import block and the
  migration-specific assertions you need.
- If this skill's smoke path needs distributed execution on the target recipe,
  the copied `test_smoke.py` should use `multi_rank_distributed_env(...)` from
  `tests/test_smoke_template.py` and configure the run as DDP, FSDP2 sharding,
  tensor parallel, or another required mode based on the skill requirement or
  user preference.
- `test_smoke.py` must use the full real capability path for this skill: real
  migrated recipe entrypoints, real parity checks, and real checkpoint-load /
  logger / checkpoint wiring. Do not short-circuit it with monkeypatch-based
  fake migrated models, fake load paths, or similar test-only stand-ins.
- If the recipe's full-capability single step only makes sense on GPU, NPU, or
  distributed hardware, write the smoke test as a real launcher-driven smoke
  test and set `gpu_preferred: true` in `test_spec.yaml`; do not degrade it
  into fake logic just to make it run in a weaker environment.

- Cover at least:
  - source model vs migrated model parity on supported inputs
  - CPU or GPU class vs NPU class parity on shared weights when an NPU variant exists
- Use strict comparisons such as `torch.equal` when the migration requires identity rather than loose closeness.

When executing this skill for a user recipe, add these tests automatically. Do not
require the user to ask for the test layout separately. Run validation only in
fresh subagents with `fork_context=false`. Do not run these
`python -m tests.test_skills` commands from the main agent's local terminal,
background terminal sessions, or any other non-subagent shell fallback. First run
`python -m tests.test_skills --recipe <recipe> --skill model-migration --layer structure`,
then a new subagent for `--layer runtime` only after structure passes, and then a
new subagent for `--layer smoke` only after runtime passes. The main agent should
summarize all three layer results. If `test_smoke.py` is blocked by device
availability, distributed-launch requirements, or permissions, the main agent
should return the exact `python -m tests.test_skills` command and any required
environment-specific launch command.

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
