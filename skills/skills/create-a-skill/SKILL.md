---
name: create-a-skill
description: Create or update a repo skill using the canonical skill format. Use when the user asks to add a new skill, rewrite an existing skill, or normalize a skill directory.
---

# Create or Update a Skill

## Goal

Create a new skill or revise an existing one so it is easy for an agent to trigger and follow.
Keep the main `SKILL.md` focused on workflow, routing rules, and validation.
Move long examples, templates, and variant-specific detail into `references/` when needed.

## Required Inputs

- The target skill path, including category.
- For a new skill, the intended `<category>` and kebab-case `<skill-name>`.
- Whether this is a new skill or an update to an existing skill.
- The user task the skill should cover, including trigger conditions and expected output.
- Any repo-specific constraints that the skill must preserve.

## Workflow

### 1. Make the skill-fit decision internally first

- This check belongs only to the current `create-a-skill` run; it is not content for the target skill itself.
- Do not copy this decision, its rationale, or any "first explain why this should or should not be a skill" wording into the target `SKILL.md`.
- Use a skill when the task has a stable workflow but requires model-, recipe-, or context-specific adaptation.
- Do not create a skill for logic that should be an engine-core reusable code API.
- Do not create a skill for one-off experiment glue that should stay inside `recipes/`.
- If the answer is no, explain that briefly in the current response and stop without creating or rewriting the target skill files.

### 2. Define the skill contract first

- Choose the category and kebab-case skill name before writing files.
- Write a short `description` that states both capability and trigger context.
- Decide the expected output, validation bar, and whether the skill needs `references/` or `scripts/`.
- If this is an update, keep the existing `name` stable unless the user explicitly requests a rename.

### 3. Branch on starting state

- For a new skill:
  - create the skill directory under `skills/<category>/<skill-name>/`
  - scaffold `SKILL.md` from the base template
  - add `references/` or `scripts/` only if the workflow truly needs them
- For an existing skill:
  - read the current `SKILL.md` and any local `references/` or `scripts/`
  - identify structural gaps:
    - missing YAML front matter
    - mismatched `name`
    - duplicated detail that belongs in `references/`
    - missing validation or output guidance
  - preserve useful content, but remove drift, redundancy, and speculative guidance

### 4. Normalize the directory layout

- Default to `skills/<category>/<skill-name>/`.
- Require `SKILL.md`.
- Add `references/` only when the skill needs read-on-demand material.
- Add `scripts/` only when the workflow depends on deterministic helper scripts.
- Do not add unrelated files such as `README.md` or `CHANGELOG.md`.

### 5. Write or rewrite `SKILL.md` in the canonical format

- Start with YAML front matter and keep exactly these fields:
  - `name`
  - `description`
- Keep `name` aligned with the directory name.
- Make `description` cover both capability and trigger context.
- Use this section skeleton unless a section is genuinely unnecessary:
  - `Goal`
  - `Required Inputs`
  - `Workflow`
  - `Validation`
  - `Output`
  - `Read On Demand`
- Prefer direct instructions over background exposition.
- Keep the main document centered on decisions, steps, and pass criteria.
- The target skill should describe how to do the domain task after the skill triggers; do not add meta steps about deciding whether the task deserves to exist as a skill.

### 6. Split supporting material only when it helps

- Move base templates, long examples, test patterns, and variant-specific detail into `references/`.
- In `SKILL.md`, state exactly when each reference should be opened.
- Add scripts only when the step would otherwise be repeatedly hand-written or error-prone.
- If scripts are added, document their inputs, outputs, and when to run them.

## Validation

- Confirm the final `SKILL.md` has valid front matter with only `name` and `description`.
- Confirm a new skill has the required directory and `SKILL.md` file.
- Confirm the main document uses the canonical section names or intentionally omits only truly unnecessary sections.
- Confirm any `references/` entries are referenced from `Read On Demand`.
- Confirm any `scripts/` entries are explicitly invoked by the workflow and described with inputs and outputs.
- Confirm the generated or updated target skill does not contain a meta step about deciding whether something should be a skill, and does not ask a future agent to explain that internal decision to the user first.
- If you add or modify helper scripts, run them at least once.

## Output

- Report which skill paths were created or updated.
- State whether the task produced a new skill, an update to an existing skill, or both.
- Summarize the structural changes that matter, not every wording tweak.
- State what you validated and what you did not validate.
- If the skill still has intentional limitations or follow-up work, list them explicitly.

## Read On Demand

- Read [references/base-template.md](references/base-template.md) when you are creating a new skill from scratch or doing a clean rewrite of a weak existing one.
- Read existing files under the target skill's `references/` or `scripts/` directories only if the task requires updating them.
