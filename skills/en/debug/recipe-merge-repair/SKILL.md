---
name: recipe-merge-repair
description: Check whether recent merged shared-code changes broke the current recipe, then repair recipe-local code/configs and validate the result. Use when updates in mvp_engine/ or other shared modules may have invalidated a recipe under recipes/.
---

# recipe-merge-repair

> 中文版：`skills/zh-cn/debug/recipe-merge-repair/SKILL.md`
> English version: `skills/en/debug/recipe-merge-repair/SKILL.md`

## Goal

- Check whether newly merged shared-code changes affect the recipe currently being touched.
- Repair recipe-local code and configs when those shared changes broke the recipe.
- Keep the fix inside `recipes/<recipe>/` unless the problem is clearly a shared-engine bug.

## Required Inputs

- The target recipe path, config path, or engine/model files under `recipes/`.
- The incoming change scope:
  - base branch or commit range, or
  - the concrete merged commits/files to inspect.
- The validation path to run after the fix:
  - config validation only,
  - model/engine smoke test,
  - full training startup.
- Repo runtime requirements when relevant:
  - source `~/.bashrc`
  - activate `.venv` with `source .venv/bin/activate`
  - use the local cluster GPU command or alias before GPU validation

## Workflow

### 1. Build the merged-change map first

- Start from the incoming shared-code diff before touching the recipe.
- Prefer:
  - `git log --oneline <base>..HEAD`
  - `git diff --name-status <base>..HEAD`
  - `git diff <base>..HEAD -- mvp_engine/ recipes/<target_recipe>/`
- Group changes by contract type:
  - config schema/layout changes
  - engine lifecycle or method signature changes
  - distributed or checkpoint runtime changes
  - model wrapper or registry changes
  - dataset/dataloader interface changes

### 2. Map shared changes to the recipe surface

- Read the target recipe’s:
  - `engine/*.py`
  - `model/**/*.py`
  - `configs/*.yaml`
  - recipe-local `configs/schema.py` if it exists
- Identify direct dependencies on changed contracts.
- Do not stop at the first failure. Build a full breakage list before editing.

Common patterns in this repo:

- core config refactor landed, but the recipe still relies on raw `DictConfig` access and has no recipe-local `ConfigClass`
- shared parallel APIs changed:
  - mesh keys renamed
  - `backend_kwargs` nesting changed
  - `parallelize_model(...)` signature changed
  - checkpoint helpers now infer backend from `DeviceMesh`
- tensor-parallel runtime was introduced or changed, but the recipe model has no `TP_MODULE_CONFIG`
- fields moved from nested config blocks to top-level blocks and the recipe YAML still uses the old structure

### 3. Repair the recipe at the correct layer

- Prefer recipe-local fixes first:
  - add or update `recipes/<recipe>/configs/schema.py`
  - set `ConfigClass` on the recipe engine
  - migrate recipe YAML keys to the current shared schema
  - update recipe-local engine/model code to the current shared runtime contract
- Only edit `mvp_engine/` when the merged code is itself wrong for all recipes.
- Do not over-abstract. Keep repair code explicit and local to the affected recipe.

### 4. Handle config-schema fallout explicitly

- If the recipe engine subclasses `Engine` and accesses fields outside `BaseEngineConfig`, add a recipe-local schema.
- Include every recipe field that the engine/model actually reads.
- Check for silent drops caused by Pydantic validation:
  - `data.*`
  - `model.*`
  - recipe-specific `optim.*`
  - checkpoint settings that were moved out of old nested blocks
- For template recipes, also check whether schema-backed tuning fields are visible in the YAML.
  - If a field exists in the recipe schema but not in the checked-in config, plain Hydra overrides may fail under struct mode.
  - For common tuning knobs used in smoke tests, prefer exposing explicit defaults in the recipe YAML instead of requiring `+foo.bar=...`.
- If the recipe config used old mesh names such as `dp_size`, `fsdp2_size`, or `tp_size`, migrate them to:
  - `parallel.mesh.replicate`
  - `parallel.mesh.shard`
  - `parallel.mesh.tensor`
- If a smoke run uses `WORLD_SIZE=1`, confirm the fixed mesh still infers valid sizes.
  - A template config with `replicate: -1` and `shard: 8` will infer `replicate=0` on a 1-rank run and fail before the recipe logic is reached.
  - For single-rank smoke tests, prefer a temporary mesh like `replicate=1, shard=1, tensor=1` unless the recipe intentionally requires sharding.
- If `backend_kwargs` became backend-scoped, rewrite to:
  - `parallel.backend_kwargs.fsdp2.*`
  - `parallel.backend_kwargs.ddp.*`

### 5. Handle distributed-runtime fallout explicitly

- Update recipe calls to shared helpers to match the current signatures.
- In this repo today:
  - `parallelize_model(...)` takes `model`, `device_mesh`, and `backend_kwargs`
  - checkpoint helpers take `mesh` first and infer DDP vs FSDP2 internally
- If the recipe previously used `self.parallel_backend`, replace that logic with mesh-derived routing or the new helper behavior.
- If TP is enabled anywhere in config, confirm the model class defines a valid `TP_MODULE_CONFIG` and add `TP_MODULE_POSTPROCESSORS` only when reshape metadata still assumes global dimensions.

### 6. Validate the repaired recipe on the intended path

- Validate at the smallest level that proves the merged breakage is fixed:
  - schema/model import
  - engine initialization
  - model parallelization smoke test
  - training startup
- For GPU validation in this repo:
  - source `~/.bashrc`
  - enter a GPU shell with the local cluster command or alias
  - `source .venv/bin/activate`
- If real data or pretrained assets are unavailable, create the smallest temporary local fixture needed for a smoke test instead of skipping validation entirely.
- Prefer temporary overrides for smoke tests instead of permanently weakening recipe defaults.

## Validation

- Confirm the merged-code inspection happened before the fix.
- Confirm the recipe now matches current shared contracts in both code and YAML.
- Run targeted validation for every repaired breakage class.
- For this repo, prefer commands like:

```bash
python -m compileall recipes/<recipe>
```

```bash
python - <<'PY'
# recipe-local config/schema smoke test
PY
```

```bash
source ~/.bashrc
<gpu-shell-command>
source .venv/bin/activate
python - <<'PY'
# GPU smoke test for model/engine startup
PY
```

- If TP config was added, verify the plan keys match real module child names.
- If only a smoke test was run, state clearly what full training behavior remains unvalidated.

## Output

- Report:
  - merged files or commits inspected
  - breakages found in the target recipe
  - recipe-local fixes applied
  - validation commands run and results
  - residual risks or unvalidated paths

## Read On Demand

- Read `references/tomatovit-parallel-refactor-example.md` when the breakage looks like a config/distributed refactor that silently invalidated a recipe after merges.
- Read `references/vit-classification-baseline-example.md` when you want a healthy-recipe baseline for config, fake-data, and single-rank startup validation.
