# Skills Directory Rules

- Follow repository-level `AGENTS.md` first; this file only defines additional rules for the `skills/` subtree.

## Decision Order

1. Check whether an existing skill already matches the task.
2. If not, check whether an existing skill can be extended without duplication.
3. Only create a new skill when the task is recurring, reusable, and not better represented in `mvp_engine/` or `recipes/`.

## Scope Boundaries

- Do not create a skill for one-off experiment logic that belongs in `recipes/`.
- Do not create a skill for stable generic runtime capabilities that belong in `mvp_engine/`.
- Create or extend a skill only when the task is a recurring agent workflow that does not fit a clean reusable API.
- Do not use a skill to restate repository-wide rules that belong in the root `AGENTS.md`.

## Required Skill Shape

- Every skill must contain a `SKILL.md`.
- `SKILL.md` is the primary source of truth for that skill.
- Every new skill must declare:
  - inputs
  - outputs
  - failure modes
- Every `SKILL.md` should clearly state when the skill should be used.
- Include references or examples when they are necessary for correct execution.

## Reuse Rules

- Before creating a new skill, check whether an existing skill already covers the same pattern.
- Prefer extending or reusing an existing skill instead of creating a duplicate skill.

## Authoring Constraints

- Do not hardcode paths outside this repository inside a skill.
- Keep examples, templates, and references inside the skill's own folder.
- Keep skill instructions implementation-oriented, reusable, and specific enough for an agent to execute.
