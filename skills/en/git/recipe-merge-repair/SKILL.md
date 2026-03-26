---
name: recipe-merge-repair
description: Compare the current development branch against an upstream branch, identify shared-contract changes that affect the target recipe, adapt and validate the current branch, then complete the merge. Use when a base branch such as main changed substantially and an in-flight recipe branch needs to absorb those updates safely.
---

# recipe-merge-repair

> 中文版：`skills/zh-cn/git/recipe-merge-repair/SKILL.md`
> English version: `skills/en/git/recipe-merge-repair/SKILL.md`

## Goal

- Compare the current branch and upstream branch before running the merge.
- Turn the branch diff into a recipe breakage/adaptation map instead of resolving conflicts blindly.
- Adapt recipe-local code/configs, resolve conflicts, and validate on the current branch.
- Finish with the upstream branch successfully merged into the active development branch.
- Keep the fix inside `recipes/<recipe>/` unless the problem is clearly a shared-engine bug.

## Required Inputs

- The current development branch and the upstream branch to merge from, for example `main`.
- The target recipe path, config path, or engine/model files under `recipes/`.
- If known, the merge base, commit range, or high-priority commits/files to inspect.
- The validation path to run after the fix:
  - config validation only,
  - model/engine smoke test,
  - full training startup.
- Whether the working tree is clean; if not, decide whether local uncommitted changes are in scope for the merge.
- Repo runtime requirements when relevant:
  - source `~/.bashrc`
  - activate `.venv` with `source .venv/bin/activate`
  - use the local cluster GPU command or alias before GPU validation

## Workflow

### 1. Establish merge context and safety boundaries first

- Do not start with `git merge`. First confirm:
  - the current branch
  - the upstream branch
  - the merge base between them
  - whether the working tree has uncommitted changes
- Prefer:
  - `git branch --show-current`
  - `git status --short`
  - `git merge-base <current_branch> <upstream_branch>`
- If the tree is dirty, do not let the merge overwrite unclear local edits.
- Record the recipe target, config entrypoint, and validation goal before moving into diff analysis.

### 2. Compare the two branches before merging

- Separate three buckets:
  - what exists only on the upstream branch
  - what exists only on the current development branch
  - where both branches touch the same files or contracts
- Prefer:
  - `git log --left-right --graph --oneline <merge_base>...<upstream_branch>`
  - `git log --left-right --graph --oneline <merge_base>...<current_branch>`
  - `git diff --name-status <merge_base>..<upstream_branch>`
  - `git diff --name-status <merge_base>..<current_branch>`
  - `git diff --name-status <current_branch>...<upstream_branch>`
  - `git diff <merge_base>..<upstream_branch> -- mvp_engine/ recipes/<target_recipe>/`
  - `git diff <merge_base>..<current_branch> -- recipes/<target_recipe>/`
- If the current branch also changed shared code, optionally inspect:
  - `git range-diff <merge_base>..<upstream_branch> <merge_base>..<current_branch>`
- Group changes by contract type:
  - config schema/layout changes
  - engine lifecycle or method signature changes
  - distributed or checkpoint runtime changes
  - model wrapper or registry changes
  - dataset/dataloader interface changes

### 3. Map branch deltas to the recipe surface

- Read the target recipe’s:
  - `engine/*.py`
  - `model/**/*.py`
  - `configs/*.yaml`
  - recipe-local `configs/schema.py` if it exists
- Identify direct dependencies on both upstream contract changes and current-branch local edits.
- Do not stop at the first failure. Build a full breakage list and merge-hotspot list before editing.

Common patterns in this repo:

- core config refactor landed, but the recipe still relies on raw `DictConfig` access and has no recipe-local `ConfigClass`
- shared parallel APIs changed:
  - mesh keys renamed
  - `backend_kwargs` nesting changed
  - `parallelize_model(...)` signature changed
  - checkpoint helpers now infer backend from `DeviceMesh`
- tensor-parallel runtime was introduced or changed, but the recipe model has no `TP_MODULE_CONFIG`
- fields moved from nested config blocks to top-level blocks and the recipe YAML still uses the old structure

### 4. Make a merge/adaptation plan before resolving conflicts

- For each hotspot, decide the strategy first:
  - take the upstream version directly
  - keep the current branch’s recipe-local intent
  - compose both sides manually
  - treat it as a shared-layer bug and fix `mvp_engine/`
- Typical judgments:
  - if upstream changed the shared contract and the current branch changed recipe wiring, usually keep both and adapt at the recipe layer
  - if the current branch copied old shared logic and upstream now provides the new shared implementation, do not revive the old contract; migrate to the new one
  - do not resolve conflicts line-by-line without deciding what runtime contract should exist after the merge

### 5. Repair the recipe at the correct layer and resolve the merge

- Prefer recipe-local fixes first:
  - add or update `recipes/<recipe>/configs/schema.py`
  - set `ConfigClass` on the recipe engine
  - migrate recipe YAML keys to the current shared schema
  - update recipe-local engine/model code to the current shared runtime contract
- Only edit `mvp_engine/` when the merged code is itself wrong for all recipes.
- Do not over-abstract. Keep repair code explicit and local to the affected recipe.
- Once the change map is clear, run the actual merge. Prefer:
  - `git merge --no-commit --no-ff <upstream_branch>`
- Resolve conflicts using the hotspot map, then apply any required recipe-local adaptation.
- Do not assume the problem is solved just because the textual conflicts are gone; continue with contract-level checks.

### 6. Handle config-schema fallout explicitly

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

### 7. Handle distributed-runtime fallout explicitly

- Update recipe calls to shared helpers to match the current signatures.
- In this repo today:
  - `parallelize_model(...)` takes `model`, `device_mesh`, and `backend_kwargs`
  - checkpoint helpers take `mesh` first and infer DDP vs FSDP2 internally
- If the recipe previously used `self.parallel_backend`, replace that logic with mesh-derived routing or the new helper behavior.
- If TP is enabled anywhere in config, confirm the model class defines a valid `TP_MODULE_CONFIG` and add `TP_MODULE_POSTPROCESSORS` only when reshape metadata still assumes global dimensions.

### 8. Validate the merged result on the intended path

- Validate at the smallest level that proves the merge breakage is fixed:
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

### 9. Finish the merge and run closing checks

- Only complete the merge commit after conflict resolution and validation succeed.
- At minimum, run:
  - `git diff --check`
  - `git status --short`
- Confirm the final state means:
  - upstream features are now present on the current branch
  - current recipe-local development goals were not overwritten by upstream changes
  - validation covered the merged tree, not only a pre-merge snapshot

## Validation

- Confirm branch comparison happened before the merge and before repairs.
- Confirm you distinguished:
  - upstream-only changes after the merge base
  - current-branch-only changes after the merge base
  - overlapping files and contracts that became merge hotspots
- Confirm the recipe now matches current shared contracts in both code and YAML.
- Run targeted validation for every repaired breakage class.
- Confirm validation ran on the post-merge working tree.
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
  - current branch, upstream branch, and merge base
  - branch-only files or commits inspected
  - files or contracts identified as merge hotspots
  - breakages found in the target recipe
  - recipe-local fixes applied
  - how key merge conflicts were resolved
  - validation commands run and results
  - residual risks or unvalidated paths

## Read On Demand

- Read `references/tomatovit-parallel-refactor-example.md` when the upstream branch contains a large shared config/distributed refactor that the current recipe branch now needs to merge in.
- Read `references/vit-classification-baseline-example.md` when you want a healthy-recipe baseline for config, fake-data, and single-rank startup validation.
