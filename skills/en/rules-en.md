# Unified Skill Rules

This document defines the standard template that all newly added or modified skills should follow.

## Contents

- [Standard Skill Directory Layout](#skill-layout)
- [Required Metadata for `SKILL.md`](#skill-frontmatter)
- [Unified `SKILL.md` Section Skeleton](#skill-structure)
- [Standards for `references/`](#skill-references)
- [Standards for `scripts/`](#skill-scripts)
- [Writing Guidelines](#writing-guidelines)
- [Base Template](#base-template)

<a id="skill-layout"></a>

## Standard Skill Directory Layout

Each skill should, by default, maintain mirrored English and Chinese directories:

```text
skills/
├── en/<category>/<skill-name>/
│   ├── SKILL.md
│   ├── references/   # optional
│   └── scripts/      # optional
└── zh-cn/<category>/<skill-name>/
    ├── SKILL.md
    ├── references/   # optional
    └── scripts/      # optional
```

Constraints:

- `<category>` should continue using the existing top-level categories, such as `training`, `parallel`, `model`, and `git`
- `<skill-name>` must use lowercase English letters with hyphens, such as `gradient-checkpointing`
- `SKILL.md` is required
- Create `references/` only when there is actual read-on-demand material
- Create `scripts/` only when the skill workflow explicitly depends on helper scripts
- Do not create unrelated extra documents such as `README.md` or `CHANGELOG.md`
- Do not commit `__pycache__/`, temporary files, cache files, or generated artifacts

<a id="skill-frontmatter"></a>

## Required Metadata for `SKILL.md`

Every `SKILL.md` must start with YAML front matter, and it must contain exactly these two fields:

```md
---
name: <skill-name>
description: <One or two sentences describing what the skill does and when it should be triggered>
---
```

Requirements:

- `name` must match the directory name
- `description` must cover both scope of capability and trigger context
- Do not add other custom fields to the front matter
- `name` must be identical in the English and Chinese versions
- `description` may be written in different languages, but the meaning must stay aligned

Recommended writing pattern:

- State what the skill does first
- Then state when it should be used
- Make the `description` sufficient for an agent to decide whether the skill should trigger

<a id="skill-structure"></a>

## Unified `SKILL.md` Section Skeleton

All future skill documents should use this section skeleton. It may be trimmed by skill type, but section names should no longer drift.

```md
---
name: <skill-name>
description: <skill capability + trigger context>
---

# <Skill Title>

> 中文版：`skills/zh-cn/<category>/<skill-name>/SKILL.md`
> English version: `skills/en/<category>/<skill-name>/SKILL.md`

## Goal

Use 2 to 5 lines to describe the skill's goal, scope, and expected deliverable.

## Required Inputs

List the context, input files, configuration, or constraints that must be confirmed before using this skill.

## Workflow

### 1. <Step Name>

Explain what to do in the first step.

### 2. <Step Name>

Explain what to do in the second step.

## Validation

Explain how to validate the result, including minimal commands, checks, or pass criteria.

## Output

Explain what the final user-facing output must contain. If a fixed response format is required, place it here.

## Read On Demand

List which files under `references/` should be read and under what conditions.
```

Unified rules:

- Use `Goal` for the top-level goal section
- Use `Required Inputs` for the input section
- Use `Workflow` for the main process section
- Use `Validation` for the validation section
- Use `Output` for the output section
- Use `Read On Demand` for the additional-material entry point

If a skill does not need one of these sections, it may be removed, but do not rename it to a near-synonym. For example:

- Do not introduce `Steps`; fold it into `Workflow`
- Do not introduce `Review output template`; fold it into `Output`
- Do not introduce `Reference` or `Example` as the ending top-level section; fold them into `Read On Demand`

<a id="skill-references"></a>

## Standards for `references/`

`references/` should contain read-on-demand material only, not the main workflow itself.

Typical content for `references/`:

- reference implementations
- example configurations
- sample tests
- long-form explanations
- variant-specific material split by framework, model, or backend

Requirements:

- `SKILL.md` must clearly state when to read which reference file
- reference filenames should express their purpose, such as `vit_example.md` or `fsdp_notes.md`
- if a reference file is long, it should begin with a table of contents or quick navigation
- do not duplicate the same explanation in both `SKILL.md` and `references/`

<a id="skill-scripts"></a>

## Standards for `scripts/`

`scripts/` should contain only helper scripts that the skill explicitly depends on.

Suitable cases for `scripts/`:

- the same extraction or transformation logic would otherwise be rewritten repeatedly
- the step requires deterministic behavior and should not be generated ad hoc each time
- the script can significantly shorten the main document or reduce error rate

Requirements:

- `SKILL.md` must state when the script should run, what its input is, and what its output is
- script names should directly describe their purpose
- each script should keep a single clear responsibility
- any newly added script should be actually run at least once for validation
- if a script is only stored under one language version but shared logically, the other language version should reference the same script path and explain why

<a id="writing-guidelines"></a>

## Writing Guidelines

All future skill documents should follow these writing rules:

- Prefer imperative or instruction-style sentences instead of long background exposition
- Prioritize how and when to act; avoid repeating general knowledge the agent already knows
- Keep only the core workflow in the main document; move details, examples, templates, and sample configs into `references/`
- When the document grows beyond roughly `300` to `500` lines, prefer splitting content into `references/`
- If multiple variants exist, keep only the routing rules in the main document and split variant-specific details into separate reference files
- In general, do not go deeper than `###` heading level
- Use one consistent step numbering scheme; do not mix multiple numbering systems

<a id="base-template"></a>

## Base Template

```md
---
name: <skill-name>
description: <What this skill does. When this skill should be used.>
---

# <Skill Title>

> 中文版：`skills/zh-cn/<category>/<skill-name>/SKILL.md`
> English version: `skills/en/<category>/<skill-name>/SKILL.md`

## Goal

- State the goal
- State the boundaries
- State the expected deliverable

## Required Inputs

- List the required inputs
- List the necessary prerequisites

## Workflow

### 1. Gather Context

- What information to collect first
- Which files to inspect first

### 2. Make Changes

- What principles to follow
- How to branch when conditions differ

## Validate

### Validation Criteria / Test Cases

- Which commands to run
- Which results to check

## Output

- What the final response must contain
- If a fixed template is needed, put it here

## Read On Demand

- When to read `references/<file>.md`
- When to run `scripts/<script>.py`
```
