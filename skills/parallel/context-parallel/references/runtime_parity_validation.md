# Runtime Parity Validation

Use whenever the final correctness claim depends on distributed execution,
FlashAttention, model-family kernels, or memory savings.

This file owns hard runtime and parity/impact validation. It does not own cheap
contract assertions, model implementation strategy, or runtime mesh semantics.

## Invariant

- Static and contract tests can block bad implementations, but they do not prove
  CP correctness.
- Resources available through user or repository instructions must be attempted.
- Blocked hard validation is reported as unresolved, not passed.

## Public Validation

- Include a CP-on smoke command or job.
- Include a CP-off/CP-on parity runner that records loss and gradient metrics,
  including max and mean gradient absolute differences.
- Generate metrics non-invasively with hooks, wrappers, or generic observers
  before changing production engine/model code.

## Assertion Hooks

Use `assert_cp_parity_artifact(...)` from `references/asserts.py` in
`tests/test_parity.py` or a skill-local impact test. Set
`PARITY_ARTIFACT_PATHS` in the recipe-local assertion copy only for parity or
impact tests that should require existing artifacts.

## Validation Targets

- Final report claims correctness from static tests only.
- Parity artifact says blocked/not-run while the result is treated as pass.
- Resource instructions exist but no Slurm or accelerator attempt was made.
