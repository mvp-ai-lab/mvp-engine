---
name: recipe-merge-repair
description: Merge an upstream branch into a recipe development branch by
  comparing branch deltas, mapping shared-contract breakages, resolving
  conflicts, adapting the recipe, validating affected paths, and completing the
  merge safely.
---

# Recipe Merge Repair

## Goal

Safely absorb upstream changes into the current recipe branch:

- compare upstream and current-branch changes before merging;
- map branch deltas to recipe runtime surfaces;
- resolve textual conflicts and semantic contract conflicts;
- repair the recipe at the smallest correct layer;
- validate the post-merge tree before completing the merge.

Keep fixes inside `recipes/<recipe>/` whenever possible. Modify `mvp_engine/`
only when the issue clearly belongs to shared infrastructure.

## Required Inputs

Identify these before running a merge:

- current development branch;
- upstream branch, usually `main`;
- target recipe path and key entry config;
- known commits or files that deserve focused review;
- intended validation depth;
- working-tree state and whether local uncommitted changes are in scope;
- data, weights, compute, service, or permission prerequisites.

Ask the user before proceeding if the merge could overwrite unrelated dirty
files or if validation requires unavailable resources.

## Workflow

### 1. Establish Merge Context

Do not start with `git merge`. First inspect:

```bash
git branch --show-current
git status --short
git merge-base <current_branch> <upstream_branch>
```

Record what the merge must preserve: target recipe capabilities, entrypoints,
configs, and validation goal.

### 2. Compare Branch Deltas

Compare upstream-only, current-only, and overlapping changes:

```bash
git diff --name-status <merge_base>..<upstream_branch>
git diff --name-status <merge_base>..<current_branch>
git diff --name-status <current_branch>...<upstream_branch>
```

Read `references/merge_rules.md` for the full comparison checklist and hotspot
mapping rules.

### 3. Map Hotspots To Recipe Contracts

Inspect the target recipe's engine, model, data, config, registration, tests,
and shared-layer call sites.

Build a breakage map:

- capability affected;
- upstream change involved;
- current-branch assumption involved;
- repair layer: recipe-local, shared-layer, or design decision.

### 4. Merge And Repair

After the hotspot map is clear, merge without committing:

```bash
git merge --no-commit --no-ff <upstream_branch>
```

Resolve conflicts by final behavior, not conflict-block convenience. Preserve
recipe-local goals while adapting to new shared contracts.

### 5. Validate Post-Merge

Run the smallest validation that proves repaired paths work, then broader checks
when feasible. At minimum, run:

```bash
git diff --check
git status --short
```

Only complete the merge commit after conflicts are resolved and validation is
sufficient for the requested scope.

## Validation

### Soft Validation

Confirm:

- branch comparison happened before the merge;
- upstream-only, current-only, and overlapping changes were distinguished;
- breakage map covers affected recipe runtime paths;
- final recipe contracts align with post-merge shared behavior;
- unrelated local edits were not overwritten;
- residual risks name unvalidated paths and prerequisites.

### Hard Validation

Run targeted post-merge validation for each repaired breakage class. Depending
on scope, this may include imports, structure tests, smoke tests, startup,
training, evaluation, or recipe-specific checks.

Always run `git diff --check` before finalizing. Do not report the merge as
complete if validation ran on a pre-merge or partially resolved tree.

## Output

- Branches: current, upstream, merge base.
- Hotspots: files, modules, contracts inspected.
- Breakages: issue, cause, repair layer.
- Repairs: recipe-local and shared-layer changes applied.
- Conflict Resolution: important choices made.
- Validation: commands and results.
- Remaining Risks: unvalidated paths or blocked resources.

## Read On Demand

- `references/merge_rules.md`: branch comparison, hotspot mapping, repair, and
  validation checklist.
