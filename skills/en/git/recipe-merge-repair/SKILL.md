---
name: recipe-merge-repair
description: Systematically compare the current development branch against an upstream branch, identify shared changes and integration risks that affect the target recipe, turn those changes into a clear breakage/adaptation map, and complete conflict resolution, adaptation, validation, and the final merge on the current branch. Use when a base branch changed substantially and an in-flight recipe branch needs to absorb those updates safely.
---

# recipe-merge-repair

> 中文版：`skills/zh-cn/git/recipe-merge-repair/SKILL.md`
> English version: `skills/en/git/recipe-merge-repair/SKILL.md`

## Goal

- Safely merge the upstream branch into the current development branch, not just stop at diff analysis or textual conflict cleanup.
- Resolve the code, configuration, interface, runtime-path, and shared-contract conflicts exposed by the merge so the target recipe remains runnable and maintainable afterward.
- Adapt the current recipe to upstream implementations and behavior changes so local development goals are not broken, reverted, or silently invalidated by the merge.
- Prove through targeted validation that the repair actually works after the merge, rather than treating “the conflicts are gone” as success.
- Keep fixes inside `recipes/<recipe>/` whenever possible, and only modify shared code when the problem clearly belongs to the shared layer.

## Required Inputs

- The current development branch and the upstream branch to merge from, for example `main`.
- The target recipe path, entry config, key modules, or the relevant engine/model/data files under `recipes/`.
- If known, the merge base, commit range, or commits/files that deserve focused review.
- The intended validation scope and depth:
  - only basic viability or the minimum critical path,
  - smoke tests for the main runtime path,
  - or more complete startup, training, evaluation, inference, or end-to-end validation.
- Whether the working tree is clean; if not, decide whether local uncommitted changes are in scope for this merge.
- Any required environment, data, service, permission, or compute prerequisites for validation.
- If the user agrees, the user-specified way to request the resources needed for validation.

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
- If the tree is dirty, do not let the merge overwrite local edits whose ownership or intent is unclear.
- Before diff analysis, write down what this merge must preserve: the target recipe, its main entrypoints, its critical capabilities, and the validation goal.

### 2. Compare the two branches before merging

- Separate three kinds of change:
  - what was added or changed on the upstream branch
  - what was added or changed on the current development branch
  - where the two branches overlap on files, modules, or contracts
- Prefer:
  - `git log --left-right --graph --oneline <merge_base>...<upstream_branch>`
  - `git log --left-right --graph --oneline <merge_base>...<current_branch>`
  - `git diff --name-status <merge_base>..<upstream_branch>`
  - `git diff --name-status <merge_base>..<current_branch>`
  - `git diff --name-status <current_branch>...<upstream_branch>`
  - `git diff <merge_base>..<upstream_branch> -- mvp_engine/ recipes/<target_recipe>/`
  - `git diff <merge_base>..<current_branch> -- recipes/<target_recipe>/`
- If the current branch also changed shared code, inspect this when needed:
  - `git range-diff <merge_base>..<upstream_branch> <merge_base>..<current_branch>`
- Do not organize the comparison only by file names. Also organize it by capability surface and contract surface, for example:
  - entrypoints and configuration
  - API, class, function, or hook expectations
  - training, evaluation, and inference lifecycle behavior
  - shared capabilities such as data, model, registry, checkpoint, and logging
  - resources, devices, startup mode, or other runtime prerequisites

### 3. Map branch deltas to the recipe surface

- Read the target recipe’s main entrypoints and dependency surface instead of staring only at conflicted files. At minimum, cover:
  - the recipe’s engine, model, data, config, entry scripts, or registration points
  - the recipe’s call sites into shared-layer capabilities
  - recipe-local helpers, schema, tests, or validation scripts when they exist
- Identify the recipe’s direct dependencies on both upstream shared changes and current-branch local edits, and trace how those dependencies connect into real runtime paths.
- Do not stop at the first failure. Build the full breakage list and merge-hotspot list before editing.
- Breakage mapping should not only answer “which file conflicts.” It should answer:
  - which capabilities will break
  - why they will break
  - whether the issue belongs to recipe-local adaptation, a shared-layer defect, or a design mismatch that must be recomposed across both sides
- The goal is not to guess one familiar case in advance. The goal is to turn the branch diff into a clear recipe adaptation map.

### 4. Make a merge/adaptation plan before resolving conflicts

- For each hotspot, decide the strategy first:
  - take the upstream version directly
  - keep the current branch’s recipe-local design
  - manually compose both sides
  - treat it as a shared-layer issue and fix shared code
- Prioritize runtime semantics and final behavior over the textual shape of the conflict blocks.
- If upstream changed the shared contract and the current branch changed recipe wiring, usually keep both and adapt at the recipe layer.
- If the current branch only continues an old shared behavior while upstream now provides a new shared implementation, do not force the old logic back in. Decide whether to migrate to the new semantics or add the minimal compatibility layer that is actually needed.

### 5. Repair the recipe at the correct layer and resolve the merge

- Repair the problem at the smallest correct layer: prefer recipe-local adaptation first, then decide whether any shared-layer change is truly necessary.
- Only modify `mvp_engine/` or other shared code when the issue clearly belongs to the shared implementation and affects more than one recipe.
- Do not over-abstract, and do not turn this merge into a broad refactor. Repairs should stay direct, local, and easy to validate.
- Once the analysis is clear enough, perform the actual merge. Prefer:
  - `git merge --no-commit --no-ff <upstream_branch>`
- Resolve conflicts using the hotspot map, then apply the necessary adaptations and repairs.
- Do not assume the work is done just because the textual conflicts disappeared; keep checking whether behavior, contracts, and critical paths still hold.

### 6. Handle gaps between shared contracts and local assumptions explicitly

- Check, one by one, whether the recipe’s assumptions about the shared layer still hold, including but not limited to:
  - input/output expectations
  - configuration and parameter entrypoints
  - class, function, hook, registry, or helper call patterns
  - initialization, loading, restore, save, and evaluation lifecycle behavior
  - resource, device, environment, permission, or startup prerequisites
- Do not stop at fixing surface-level import errors or conflict blocks. Confirm that the contracts required by the real runtime path are fully aligned.
- If upstream changed shared behavior while the current recipe still depends on the old semantics, make an explicit choice: migrate to the new semantics, add recipe-local compatibility, or repair a real defect in the shared layer.
- For template-style, reusable, or externally consumed recipes, also verify that common entrypoints, default behavior, extension points, and override parameters still behave as intended, so the merge does not leave behind a recipe that technically runs but is practically broken to use.

### 7. Handle runtime and integration-path fallout explicitly

- Check whether the recipe still connects cleanly to surrounding systems, for example:
  - data loading and preprocessing
  - model construction, wrapping, weight loading, export, or restore
  - training, evaluation, inference, or tooling entrypoints
  - checkpoint, logging, metrics, and artifact output
  - shared helpers, registration, startup logic, or runtime features the recipe depends on
- If upstream changes affected call order, default behavior, error handling, resource assumptions, or edge conditions, repair those integration points as part of the merge instead of only patching one function signature.
- If the recipe has multiple runtime paths, confirm that both the main path and the common secondary paths were not silently broken by the merge.

### 8. Validate the merged result on the intended path

- Use the smallest sufficient validation level that proves the breakage is fixed, moving from lighter checks to heavier ones, for example:
  - static checks, imports, construction, or minimal initialization
  - smoke tests for critical modules or runtime paths
  - startup validation for the target task
  - more complete training, evaluation, inference, or end-to-end validation
- Validation must cover the paths actually affected by this merge, not just unrelated default checks.
- If real dependencies are unavailable, do not skip validation outright. Prefer the smallest viable fixture, stub, or temporary override that can still prove the core path is repaired.
- Temporary validation aids should not permanently weaken default recipe behavior; keep testing accommodations separate from the actual repair.

### 9. Finish the merge and run closing checks

- Only complete the merge commit after conflict resolution and validation succeed.
- At minimum, also run:
  - `git diff --check`
  - `git status --short`
- Confirm the final state means:
  - upstream changes are now present on the current development branch
  - the current recipe’s local development goals were not erased by upstream updates
  - the validation path matches the merged tree, not a pre-merge or partially edited state

## Validation

- Confirm that branch comparison happened before merge and before repair work.
- Confirm that you distinguished:
  - upstream-only changes after the merge base
  - current-branch-only changes after the merge base
  - overlapping hotspot files, capability surfaces, and shared contracts
- Confirm that the breakage list and merge-hotspot list cover the critical paths actually affected by this merge.
- Confirm that the recipe’s implementation, configuration, entrypoints, and dependency chain now align with the post-merge shared contracts, not just that textual conflicts disappeared.
- Run targeted validation for each repaired breakage class, and match the validation style to the capability that was actually affected.
- Validation may progress from lightweight checks to heavier runtime verification, but it must at least prove that the core path has recovered.
- Confirm that validation ran on the post-merge working tree, not on a pre-merge snapshot or a partially assembled state.
- If validation is only partial, state clearly:
  - which paths were validated
  - which paths were not validated
  - where the main residual risks remain
- If validation depends on extra environment, data, weights, services, or compute resources, state those prerequisites clearly and explain how they limit the strength of the conclusion.

## Output

- Report:
  - current branch, upstream branch, and merge base
  - branch-only files or commits inspected
  - files, modules, or contracts identified as merge hotspots
  - breakages found in the target recipe
  - recipe-local repairs or shared-layer repairs applied
  - how key merge conflicts were resolved
  - what validation ran and the results
  - remaining risks, limitations, or unvalidated paths

## Read On Demand

- Read `references/tomatovit-parallel-refactor-example.md` when the upstream branch contains a large shared config/distributed refactor that the current recipe branch now needs to merge.
- Read `references/vit-classification-baseline-example.md` when you want a healthy-recipe baseline for validation in this repository.
