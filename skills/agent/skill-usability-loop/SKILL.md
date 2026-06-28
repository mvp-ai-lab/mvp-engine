---
name: skill-usability-loop
description: Optimize a repository skill by running a fixed six-role fresh
  subagent loop that improves both first-pass instructions and executable
  validation until a fresh agent can converge to correct code within the
  declared task scope. Use when a user provides a target skill, novice task,
  optional evaluation-only demo/reference, and round budget; demo/reference
  code is oracle material only and must not become novice implementation
  context or demo-specific skill guidance.
---

# Skill Usability Loop

## Goal

Improve a skill so a user agent can run it and converge to correct code within
the declared task and validation scope.

Optimize three surfaces together:

- **Instruction quality:** the skill should make a correct first implementation
  likely.
- **Validation quality:** the skill should require public tests or checks that
  expose wrong code and guide self-repair.
- **Generality:** skill edits must apply to all suitable recipes, model
  families, or frameworks, not only to the current demo.

Use this skill when the user asks to optimize, audit, or regression-test the
first-use and self-repair experience of a repository skill.

## User Prompt Contract

Expect the user to provide task-level inputs:

- target skill path, such as `skills/parallel/context-parallel/SKILL.md`;
- realistic one-sentence novice task;
- optional demo recipe or reference implementation, used only as an oracle for
  tester, reviewer, and diagnoser roles;
- optional round budget or correctness goal.

Do not require the user to write per-round instructions, reviewer checklists,
tester commands, failure taxonomy, or final output format. Derive those from
this loop plus the target skill's `Required Inputs`, `Workflow`, `Validation`,
and `Output` sections. Ask only when repository context cannot reveal the task,
correctness standard, or runtime resources.

## Required Inputs

Identify before starting:

- target skill path;
- novice task;
- optional demo/reference artifact and which non-novice roles may use it;
- correctness standard for final code, including static, runtime, parity,
  reviewer, and blocked-validation requirements;
- public validation that the novice may see and run;
- private acceptance checks that only tester/reviewer use;
- hard validation requirements from the target skill, especially runtime,
  numerical, parity, GPU, or distributed checks that must run when resources or
  resource instructions are available;
- stop criteria, usually two fresh novice runs with no unresolved reviewer or
  tester correctness findings, or a fixed round budget;
- runtime resources or scheduler parameters needed for hard validation;
- local resource paths needed by validation, such as `CUSTOM.md`, data
  directories, model/cache directories, credential mounts, or other untracked
  files referenced by repository/user instructions.

If the target skill or task is unclear, ask before spawning subagents. If only
the reviewer/tester checklist is unclear, read the target skill and derive it.

## Subagent Roles

Use exactly the six project-scoped Codex custom agents in `.codex/agents/`:

1. `.codex/agents/novice-coder.toml`
2. `.codex/agents/code-reviewer.toml`
3. `.codex/agents/tester.toml`
4. `.codex/agents/skill-diagnoser.toml`
5. `.codex/agents/skill-writer.toml`
6. `.codex/agents/writing-supervisor.toml`

Every subagent must be fresh with `fork_context=false`. Close each subagent
after its report is captured. Do not store subagent prompt files under this
skill; `.codex/agents/*.toml` is the prompt source of truth.

## Round Flow

### 1. Freeze Round

Record:

- target skill revision and changed reference files;
- baseline code revision;
- one-sentence novice task;
- demo/reference oracle, if any, and which non-novice roles may use it;
- stop criteria or round budget;
- validation budget and unavailable resources.

For every round that may produce code changes, create a fresh temporary git
worktree before spawning the novice. The worktree path must live on the same
shared filesystem as the main checkout, not under `/tmp`, so scheduler jobs and
multi-node workers can see the same files. Prefer a dedicated directory under
the main checkout's parent directory, for example
`<main-checkout-parent>/.skill-usability-worktrees/<repo-name>/<round-id>`,
unless the user provides another shared worktree root. The worktree must contain
the intended baseline code plus the current target skill revision for that
round.

After creating the worktree, freeze the local resource context before spawning
any subagent:

- Copy small local instruction overlays from the main checkout into the
  worktree. In particular, if `CUSTOM.md` exists in the main checkout, copy it
  to `<round-worktree>/CUSTOM.md` with the exact current contents.
- Link large or mutable resources instead of copying them. If repository/user
  instructions, target-skill validation, public tests, or smoke commands need
  paths such as `data/`, model/checkpoint/cache directories, credential mounts,
  or other untracked local resources, create symlinks in the worktree that point
  to the main checkout or shared resource paths.
- Do not copy data directories or other potentially large resources into the
  worktree.

Treat copied overlays and resource symlinks as frozen round context, not novice
implementation output: do not stage or edit them, and delete them with the
worktree. This is required even when those files are ignored or untracked,
because resource instructions, scheduler accounts, data paths, and existing
virtual-environment rules must be identical for tester, novice, and reviewer.

All novice code edits for that round must happen only inside that worktree. Do
not let probe code changes modify or dirty the main repository checkout. Pass
the same worktree path to `novice-coder`, `code-reviewer`, and `tester`. Remove
the worktree after all round reports are captured.

Use the main repository checkout's existing `uv` virtual environment for all
code execution, testing, and validation from the temporary worktree. Do not
create or sync a separate virtual environment inside the round worktree unless
the user explicitly asks.

When spawning subagents, include both paths in the prompt:

- main checkout path, for the existing `.venv` and any source-of-truth local
  instruction overlays;
- round worktree path, where all code edits and validation commands must run.

If `CUSTOM.md` exists in the main checkout but is absent from the round
worktree, or if required local resources are absent and not linked into the
worktree, stop setup and fix the worktree context before continuing. Do not let
any subagent infer that resource instructions or data are unavailable merely
because the temporary worktree omitted ignored local files.

Do not include user-provided reference implementations, completed patches,
expected diffs, solution walkthroughs, or private acceptance checks in the
novice-coder prompt or workspace.

Do not change the target skill during validation design, novice implementation,
review, testing, or diagnosis. Skill edits happen only in step 7.

### 2. Derive Validation Before Coding

Spawn `tester` before coding to derive a validation plan from the target skill,
task, repository context, and optional demo/reference oracle.

The tester must separate:

- **public validation:** checks the novice may see, run, and use for self-repair;
- **private acceptance:** checks hidden from the novice and used only for final
  tester/reviewer acceptance.

Public validation must encode general invariants, not demo-specific expected
diffs. It should be suitable for eventual absorption into the target skill as a
reusable test, assertion template, blocked-validation schema, or workflow step.
It may be delivered as a text plan or as public test files in the round
worktree. If files are created before novice coding, create only public
validation artifacts, list them explicitly, and keep private acceptance outside
the novice-visible workspace until post-coding validation.

Private acceptance may be demo-derived, but it must test behavior or invariants
rather than requiring the novice to copy demo code.

If the target skill's validation or output contract requires runtime,
numerical, parity, GPU, distributed, or end-to-end checks, the tester must make
those checks required private acceptance. When a GPU is locally visible or
repository/user instructions provide a path to GPU or other required resources,
the real resource-backed validation must be attempted. Do not replace it with
static, CPU, or surrogate checks. If the prescribed resource path fails, the
round is resource-blocked or failed, not passed.

### 3. Novice Implementation And Self-Repair

Spawn `novice-coder` with only:

- the one-sentence user task;
- the target skill name/path needed to trigger the skill;
- the temporary worktree path;
- public validation from step 2.

The novice may discover repository files naturally, but the parent agent must
not point it at demo/reference implementation details.

The novice must implement code, run public validation, and self-repair within
the same round until public validation passes or a blocker is clearly recorded.
It must not delete, weaken, skip, or fake public validation.

Its final report must list:

- files changed;
- first-pass failures or uncertainty before self-repair;
- public validation attempted and final result;
- fixes made because validation failed;
- decisions inferred from the skill;
- unresolved blockers or places it still had to guess.

### 4. Independent Correctness Review

Spawn `code-reviewer` to review the novice's final output in the same temporary
git worktree.

Pass the reviewer the correctness standard, public validation, and private
acceptance intent. The reviewer may use demo/reference oracle material.

The reviewer must classify each finding as exactly one of:

- `implementation_bug`
- `instruction_gap`
- `validation_gap`
- `debug_guidance_gap`
- `missing_scaffold`
- `misleading_example`
- `over_specific_to_demo`
- `not_skill_issue`

Do not stop only because a finding is an `implementation_bug`. A first-time
implementation bug is usability evidence unless it is outside task scope,
depends only on unavailable external resources, or is caused by unrelated dirty
state.

### 5. Runtime Or Acceptance Probe

Spawn `tester` after coding to run public validation and private acceptance on
the novice output in the same temporary git worktree, not on later skill edits.

If required hard validation needs GPU, data, credentials, a scheduler, or other
external resources, tester must first read repository/user resource instructions
and attempt the prescribed path when it exists. If a local GPU or prescribed
resource path exists, tester must run the real validation and must not downgrade
it to static, CPU, or surrogate checks. If the prescribed path fails, report the
exact command and error as resource-blocked or failed validation.

Only when no local resource is available and no repository/user instructions
explain how to access the required resource may tester run the strongest cheap
surrogate checks available, such as static, AST, config, import, py_compile, or
CPU shape/contract checks. Surrogate checks do not satisfy required hard
validation.

When expected output is code, tester should include a validation that would
have failed for each reviewer correctness finding when such a check can run
without unavailable resources.

### 6. Diagnose Failure

Spawn `skill-diagnoser` with:

- validation plan;
- novice first-pass and self-repair report;
- code review;
- tester acceptance report.

The diagnoser must explain whether failures came from instructions,
validation, missing scaffold, debug guidance, over-specific examples, unrelated
implementation errors, or blocked resources.

Classification rules:

- If public validation did not catch a correctness issue that a cheap check
  could catch, classify it as `validation_gap`.
- If public validation caught the issue but the novice could not repair it from
  the skill, classify it as `instruction_gap`, `debug_guidance_gap`, or
  `missing_scaffold`.
- If a proposed fix depends on current demo details, classify it as
  `over_specific_to_demo` and require a general invariant before editing the
  skill.
- Use `not_skill_issue` only for issues outside task scope, issues caused by
  unrelated dirty state, or hard checks blocked solely by missing external
  resources after exact blocked commands are recorded.

Only `instruction_gap`, `validation_gap`, `debug_guidance_gap`,
`missing_scaffold`, `misleading_example`, and `over_specific_to_demo` may drive
skill edits.

### 7. Edit Skill

Spawn `skill-writer` to make the smallest general-first edits from the
diagnoser's skill-caused findings.

Prefer improvements in this order:

- public validation;
- assertion templates;
- scaffold or blocked-validation schema;
- debug guidance;
- concise prose.

Rules for edits:

- keep `SKILL.md` as a short route map;
- put model-, recipe-, or framework-specific detail in `references/`, gated by
  applicability;
- avoid recording round history in `SKILL.md`;
- never write the current demo branch or implementation as the answer;
- never paste recipe-specific implementation code into the main `SKILL.md`;
- turn demo failures into general invariants, required actions, reusable tests,
  or conditional model-family guidance;
- if a fix applies only to one recipe, keep it recipe-local and out of the
  generic skill;
- if a fix applies to one model family, write it as a model-family rule rather
  than a demo-recipe rule;
- if a fix applies to all suitable recipes, promote it to the main workflow or
  validation section;
- make copyable snippets real APIs or clearly mark them as pseudocode;
- prefer deleting, moving, or tightening text over expanding the main skill.

### 8. Writing Review

Spawn `writing-supervisor` to review only skill, reference, and subagent/config
changes.

The supervisor must reject edits that are too demo-specific. It checks:

- guidance applies beyond the current demo recipe;
- examples are accurate, scoped, and clearly examples;
- model-family rules are conditional and scoped;
- validation checks encode invariants, not branch-specific diffs;
- public tests help a future recipe author find mistakes without knowing the
  demo;
- validation is executable or clearly marked as resource-blocked;
- references are read-on-demand rather than required up front.

If writing fails, return once to `skill-writer` with only the supervisor
findings. Do not restart novice implementation until writing passes.

### 9. Regression Or Continue

After writing passes, remove the round worktree, create a new clean worktree,
and start a new round from step 1 with a fresh `novice-coder`.

Do not reuse previous novice context. Reuse the same one-sentence task unless
intentionally testing a new user path.

If the current round does not meet the stop criteria or stop rationale, continue
the complete loop after the fresh `novice-coder` finishes new code output: run
correctness review, acceptance probe, diagnosis, skill edit, and writing review
before starting any later regression probe.

## Stop Criteria

Default stop criteria:

- two consecutive fresh novice runs pass public validation;
- private acceptance passes, including any required real GPU/runtime/parity
  validation when a local resource or prescribed resource path exists;
- reviewer/tester have no unresolved correctness findings;
- coder reached correctness without hidden demo implementation details;
- latest skill edits remain general and reusable.

Do not stop successfully with resource-blocked hard validation if repository or
user instructions provided a resource path. Stop as blocked/failed instead. Only
when no resource path exists may blocked hard validation remain as an explicit
unresolved risk.

Budget stop:

- If the user set a round budget, stop after the last round and report
  unresolved findings.
- Explain whether each unresolved issue is instruction still unclear,
  validation still weak, scaffold missing, runtime resource blocked, task needs
  code/kit support, or the attempted edit would be too demo-specific.

Do not stop merely because a remaining issue is labeled an implementation bug
if it is still a first-time output correctness failure within scope.

## Output

Return:

- Subagent Reports: one line per role and round with pass/fail and artifacts;
- First-Pass Result: whether the initial novice output was correct;
- Self-Repair Result: whether public validation led to a correct final output;
- Public Validation Coverage: what it caught and missed;
- Private Acceptance Result: pass/fail/resource-blocked;
- Failure Taxonomy: grouped by skill-caused vs not skill-caused;
- Skill Changes: files changed and why each change is general;
- Correctness Result: unresolved reviewer/tester findings within scope;
- Next Round Recommendation: stop, run another round, or escalate to code/kit
  changes.

## Read On Demand

- `../../../.codex/agents/*.toml`: canonical persisted Codex custom agent
  definitions.
