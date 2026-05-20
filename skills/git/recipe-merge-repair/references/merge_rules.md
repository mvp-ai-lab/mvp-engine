# Recipe Merge Repair Rules

Use this reference when merging an upstream branch into an in-flight recipe
branch.

## Comparison Checklist

Before `git merge`, collect:

```bash
git branch --show-current
git status --short
git merge-base <current_branch> <upstream_branch>
git log --left-right --graph --oneline <merge_base>...<upstream_branch>
git log --left-right --graph --oneline <merge_base>...<current_branch>
git diff --name-status <merge_base>..<upstream_branch>
git diff --name-status <merge_base>..<current_branch>
git diff --name-status <current_branch>...<upstream_branch>
```

For target recipe work, also inspect:

```bash
git diff <merge_base>..<upstream_branch> -- mvp_engine/ recipes/<recipe>/
git diff <merge_base>..<current_branch> -- recipes/<recipe>/
```

## Hotspot Surfaces

Organize changes by behavior, not only by file:

- config schema and YAML defaults;
- engine lifecycle and hook order;
- model construction, wrapping, loading, and forward outputs;
- dataset, preprocessing, collator, and dataloader behavior;
- checkpoint save/load and export;
- logging, metrics, and artifact paths;
- test templates and recipe-local skill assertions;
- shared helpers called by the recipe.

## Repair Layer

Prefer this order:

1. recipe-local adaptation;
2. small compatibility shim in the recipe;
3. shared-layer fix only when the upstream contract is defective or multiple
   recipes are affected.

Do not restore old shared behavior just because it is familiar. If upstream
introduced a better shared contract, adapt the recipe to it unless that breaks
the branch goal.

## Conflict Resolution

Resolve conflicts by final runtime semantics:

- preserve recipe-local goals;
- preserve upstream bug fixes and shared-contract changes;
- avoid unrelated cleanup;
- do not delete local behavior without tracing what replaces it;
- re-read merged files after conflict markers are removed.

## Validation Selection

Match validation to repaired risk:

- import/config changes: import and config schema validation;
- recipe structure changes: `tests/test_structure.py`;
- training lifecycle changes: `tests/test_smoke.py` or startup smoke;
- data/model path changes: targeted unit or recipe-local impact validation;
- checkpoint/logging changes: save/load or artifact inspection.

Always finish with:

```bash
git diff --check
git status --short
```

State any resource blockers such as missing data, weights, GPU/NPU, or service
credentials.
