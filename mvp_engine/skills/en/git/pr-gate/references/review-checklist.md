# Review Checklist

## 1. Change scope

- Only files related to the target behavior are modified.
- No unrelated refactor mixed into the same change.

## 2. Correctness

- New logic covers edge cases.
- Failure paths are observable (errors/logging).

## 3. Maintainability

- Names are explicit and local reasoning is clear.
- No over-abstraction or unnecessary wrapping.

## 4. Regression risk

- Defaults and config behavior are still expected.
- Distributed/parallel paths are not silently broken.

## 5. Commit quality

- Commits are logically split.
- Commit messages explain intent independently.
