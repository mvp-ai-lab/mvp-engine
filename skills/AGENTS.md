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

## Recipe-Local Skill Tests

When a skill changes a user recipe, the tests for that skill should live with
that recipe, not under `skills/` and not under an unrelated demo recipe.

Use this layout:

```text
recipes/<recipe>/
└── skill_tests/
    ├── skill_manifest.yaml
    ├── test_structure.py
    ├── test_smoke.py
    └── <skill-id>/
        ├── asserts.py
        └── test_effectiveness.py  # optional, only when the skill declares effectiveness checks
```

- `skill_tests/skill_manifest.yaml` tracks installed recipe skills as a simple
  `skills` list. A skill name is written only after all required recipe-local
  test layers pass in a full skill-test run.
- `skill_tests/test_structure.py` and `skill_tests/test_smoke.py` are
  recipe-level cumulative tests. They should be created when the recipe is
  created and extended when new skills are applied.
- Each skill directory must contain `asserts.py` with `assert_structure(...)`
  and `assert_smoke(...)` hooks. Keep skill-specific structure and smoke
  assertions there.
- Recipe-level `test_structure.py` and `test_smoke.py` must load asserts in
  `skill_manifest.yaml` order, then run the current `--skill` asserts last so a
  skill being installed is validated before it is recorded in the manifest.
- A skill directory may also contain `test_effectiveness.py` when the skill
  declares effectiveness checks.
- The skill test runner discovers `structure` and `smoke` from the recipe-level
  files, and discovers optional `effectiveness` from the selected skill
  directory.
- The standard flow has two layers: `structure` and `smoke`. Skills that
  declare effectiveness checks add `effectiveness` as the third layer.
- A skill application is incomplete until every required layer passes through
  the skill test runner, or the exact environment limitation and command to run
  in a real environment are reported.
- Tests must exercise the user's real recipe/model entrypoints with a minimal
  recipe-owned config or batch, not a separate toy model unrelated to that
  recipe.
- Recipe-local skill tests should use:
  - `mvp_engine.test.recipe_probe.import_modules`
  - `mvp_engine.test.recipe_probe.load_config`
  - `mvp_engine.test.recipe_probe.build_engine`
  - `mvp_engine.test.recipe_probe.single_rank_distributed_env`
  - `mvp_engine.test.recipe_probe.multi_rank_distributed_env`
- Pass the recipe path to `import_modules` / `load_config`, then keep
  recipe-local code focused on the small config override and skill-specific
  assertions.
- When applying a new skill, preserve the recipe-level assert loader so
  validation for the new skill also covers previously installed skills.
- If a new skill conflicts with an existing assertion, the agent may update the
  assertion, but it must keep the smallest correct change and explain the
  conflict and resolution to the user.

### Standard Test Layers

- `skill_tests/test_structure.py`: verify recipe import, registry wiring,
  config schema validation, required slots, logger/checkpoint hooks, and all
  accumulated skill structure assertions.
- `skill_tests/test_smoke.py`: cover one real recipe-owned step: forward, loss,
  backward, optimizer step, logger write, checkpoint noop or temporary save,
  and all accumulated skill smoke assertions.

### Cumulative Smoke Config Pattern

- `test_smoke.py` should keep one standard recipe-owned smoke config override.
  This config is the shared validation target for all installed skills.
- When applying a new skill, update that standard config minimally so it
  satisfies both previously installed skills and the new skill. In most cases
  this means changing only `parallel.mesh`, backend kwargs, precision, or small
  model dimension values required by the capability.
- Do not add a separate per-skill smoke test that builds another engine just for
  that skill. Build one engine and one step context per rank from the shared
  config, then run every installed/current skill's `assert_smoke(...)` against
  that same engine, context, batch, log path, and checkpoint path.
- `test_smoke.py` should expose clear single-GPU and multi-GPU execution paths.
  Choose between them from the resolved config mesh world size, not from
  hardcoded skill names. Use `single_rank_distributed_env(...)` when the shared
  config resolves to one rank, and `multi_rank_distributed_env(...)` when it
  resolves to multiple ranks.
- If two skills require incompatible smoke configs, stop and explain the
  conflict instead of adding parallel smoke paths that hide the incompatibility.

### Distributed and Real-path Requirements

- If a skill needs distributed execution, use
  `mvp_engine.test.recipe_probe.multi_rank_distributed_env(...)` and configure
  the distributed mode to match the skill requirement or user preference, such
  as DDP, FSDP2 sharding, or tensor parallel.
- `test_smoke.py` must use the full real capability path for the skill: real
  recipe entrypoints, real engine wiring, and real logger/checkpoint behavior.
  Do not short-circuit it with monkeypatch-based fake engines, fake wrappers,
  fake process groups, fake device meshes, fake loggers, fake training steps, or
  similar test-only stand-ins.
- Skill-related validation should prefer the real GPU execution path whenever the
  skill or recipe meaningfully depends on accelerator, distributed, launcher,
  logger, checkpoint, or parallel behavior; do not default to CPU-only or
  single-process `gloo` as a convenience fallback. Do not silently replace a
  GPU-expected `runtime` or `smoke` test with a weaker CPU variant just to get a
  local pass. Only use a CPU fallback when the skill documentation explicitly
  allows it and the CPU path still validates the same required capability.
- If a recipe-local test really requires a real GPU, NPU, distributed launcher,
  or extra execution permission, it should fail with an actionable command for
  the user to run in that environment, not `skip`.
- If `test_smoke.py` or `test_effectiveness.py` is blocked by GPU availability,
  distributed-launch requirements, or permissions, the main agent should return
  the exact `python -m tests.test_skills ...` command and any required launcher
  command instead of asking the user to design the test flow.
- Do not swap in an unrelated tiny recipe or model. Use the user's real
  recipe/model entrypoints with the smallest recipe-owned config or batch that
  still exercises the skill landing points.

### Execution Workflow

- Before any skill validation, read `skills/README.md`, this section, and the
  target `SKILL.md`; then follow the stricter requirement when they differ.
- Run tests with
  `python -m tests.test_skills --recipe <recipe> --skill <skill-id>`.
- Recipe-local skill validation must run only in fresh subagents with
  `fork_context=false`. Do not run `python -m tests.test_skills ...` from the
  main agent's local terminal, background terminal sessions, or any other
  non-subagent shell fallback.
- Do not run a full local dry run of `structure`, `smoke`, or `effectiveness`
  before the subagent workflow. A local preflight is allowed only for the
  smallest syntax/import check needed to unblock validation.
- Run `--layer structure` in one fresh subagent, wait for it to pass, then run
  `--layer smoke` in a new fresh subagent, and finally run `--layer effectiveness`
  in a new fresh subagent when that layer exists.
- The user should not need to ask for these tests explicitly. When an agent
  applies a skill to a user recipe, it should add the matching recipe-local
  tests and try to run them by default.
- The agent should update `skill_tests/skill_manifest.yaml` automatically after
  all required recipe-local layers succeed. If a skill declares effectiveness
  checks, all three layers must pass before the skill name is recorded.
- If any required layer fails, do not mark the work complete and do not update
  `skill_tests/skill_manifest.yaml` as passing.
- The main agent should summarize the `structure` / `smoke` / `effectiveness`
  outcomes after those subagents finish, omitting effectiveness for skills that
  do not define it.
