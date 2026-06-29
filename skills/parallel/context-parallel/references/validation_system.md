# Context Parallel Validation System

Use this file for test-layer policy and for mapping mechanism references to
executable assertion hooks.

This file owns:

- public validation before coding;
- canonical recipe-local assertion ownership;
- structure/contract/smoke/parity boundaries;
- blocked hard-validation semantics.

This file does not own:

- CP runtime facts;
- mechanism-specific invariants;
- implementation guidance for a model family.

## Public Validation Before Coding

For a fresh implementation task:

1. Copy `references/asserts.py` to
   `recipes/<recipe>/tests/skills/context-parallel/asserts.py` if missing.
2. Add `tests/test_contract.py` from
   `tests/templates/test_contract.py.template` when semantic CP checks are
   needed.
3. Fill recipe-local knobs in the copied assertions instead of changing generic
   skill prose for one recipe.
4. Run public tests on the baseline and confirm they fail for the missing CP
   behavior.
5. Give the coder the task, skill, and public tests. The coder implements, runs
   tests, and self-repairs from failures.

Do not derive public tests from a demo diff. Derive them from invariants.

## Canonical Assertion Surface

The copied `tests/skills/context-parallel/asserts.py` is canonical for that
recipe. Preserve it across rounds and edit it incrementally.

Allowed:

- add reusable assertions for newly discovered invariants;
- add recipe-local class names, field names, helper names, and thresholds;
- move repeated round-local checks into the canonical file.

Forbidden:

- clear the file each round;
- replace assertions with demo-specific expected code;
- delete assertions to make an implementation pass;
- treat comments, marker strings, or unused helpers as proof of dataflow.

Round-local tests may be regenerated, but they should import or copy from the
canonical template and fill recipe-local knobs.

## Mechanism Hook Map

When a mechanism reference applies, connect it to executable validation:

- packed attention topology:
  `assert_packed_topology_matches_flattened_qkv(...)`;
- VLM or model-family media ownership:
  `MODEL_FAMILY_SEQUENCE_FIELDS`,
  `MODEL_FAMILY_HELPER_NAMES`,
  `MODEL_FAMILY_NATIVE_LOCAL_FORWARD`,
  `assert_model_family_dataflow_contract(...)`, and
  `assert_cp_helper_outputs_drive_dataflow(...)`;
- custom attention dispatch:
  `CP_ATTENTION_CLASS_NAMES` and `assert_attention_dispatch_bound(...)`;
- auxiliary hidden layout:
  `AUXILIARY_HIDDEN_NAMES` and `assert_auxiliary_hidden_layout(...)`;
- gradient sync order:
  `assert_optimizer_order_contract(...)` and `assert_before_train_end(...)`;
- runtime parity:
  `assert_cp_parity_artifact(...)` and `tests/templates/test_parity.py.template`.

Do not leave a selected mechanism as prose-only guidance.

## Layer Boundaries

Structure:

- proves files, imports, configs, registry, and basic CP entrypoints exist;
- must not prove forward dataflow, runtime success, or parity.

Contract:

- proves cheap CP semantics with AST/source/config/runtime-light checks;
- may check helper dataflow, bound forward patches, attention dispatch, and
  optimizer-step order;
- must not call `engine.train()` or require accelerators.

Smoke:

- proves the CP-on runtime path can run one small step;
- may use hooks to assert observed runtime state;
- must not claim CP-off/CP-on numerical parity or memory impact.

Parity/Impact:

- validates artifacts from real CP-off/CP-on runs;
- checks loss, gradient, token ownership, memory, throughput, checkpoint, or
  other declared metrics;
- treats blocked/not-run artifacts as unresolved unless the test explicitly
  checks artifact format only.

## Runtime And Parity

If repository or user instructions explain how to access GPU/NPU/distributed
resources, hard validation must try those resources. Follow local instructions
such as `CUSTOM.md` exactly. If the attempt fails, report the command and
concrete failure.

Recommended CP parity artifact:

```json
{
  "status": "passed",
  "metrics": {
    "loss_cp_off": 0.0,
    "loss_cp_on": 0.0,
    "loss_abs_diff": 0.0,
    "grad_max_abs_diff": 0.0,
    "grad_mean_abs_diff": 0.0,
    "tokens_per_rank_min": 1,
    "tokens_per_rank_max": 1
  }
}
```

If hard validation cannot run:

```json
{
  "status": "blocked",
  "reason": "Concrete resource, dependency, or scheduler failure.",
  "metrics": {}
}
```

Blocked means unresolved. It is never evidence of CP correctness.

## Non-Invasive Metric Collection

Generate parity metrics without changing production recipe code by default:

- use recipe-local parity runners;
- use smoke hooks such as `assert_train_pre_step_end` or
  `assert_forward_step_end`;
- wrap engine instance methods inside the runner;
- record peak memory outside the engine with the accelerator API;
- add a generic observer surface only when hooks cannot expose required state.

Do not add CP-specific metrics fields to production engine/model code solely for
tests.
