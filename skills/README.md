# Skills

Skills are a **second kind of interface** in this repo, alongside code interfaces (functions, classes).

---

## Core idea: codebase for the agent era

This repo is built for a **coding-agent-first** workflow. The interface to the system is not only “code you run” but also **structured guidance an agent can follow**. Many training features have a clear, repeatable *pattern* (e.g. “wrap each encoder layer in checkpoint”), but the *implementation* depends on each model’s structure (encoder layout, layer types, forward signatures). Forcing everything into a single generic API leads to over-abstraction and hard-to-follow code; hand-writing each variant is repetitive and error-prone. **Skills** are the middle path: we document the pattern once (workflow, rules, examples, tests), and the agent generates the concrete code for each new model or recipe. So the codebase is “code + skills”: code where a single API fits, skills where the right move is “same pattern, different glue per case.” That keeps the repo simple, readable, and easy to extend without piling abstraction into the core engine.

---

## Why Skills Exist

Some capabilities in a training framework have **clear logic and fixed patterns but cannot be generalized into a single API**—because they must be adapted to each model’s structure.

Example: gradient checkpointing. The recipe is clear (wrap each layer’s forward in `torch.utils.checkpoint.checkpoint` inside the Encoder loop), but Encoder layout, layer types, and forward signatures differ per model. Forcing a single `apply_gradient_checkpointing(model)` leads to over-abstraction.

The usual options are either a brittle generic API (hard to maintain) or hand-writing per model (repetitive, error-prone). **Skills are a third path**: encode “how to do it” as structured guidance and let a coding agent generate the adapted code for each new model.

## Code vs Skill: When to Use Which

```
Can it be generalized into a single API? ─── yes ──→ Implement as code (functions/classes) in the relevant mvp_engine/ module
        │
        no
        │
        ▼
Is there a fixed pattern to follow? ─── yes ──→ Implement as a Skill
        │
        no
        │
        ▼
    Keep it in recipe/ as experiment-specific code
```

**Code interface** examples: checkpoint I/O, logging utilities, config parsing, distributed primitives.
**Skill interface** examples: gradient checkpointing adaptation, FSDP wrap policy, adding a new model, adding a new dataset.

## Directory Layout

```
skills/
├── README.md             ← overview (this repo)
├── training/             ← training technique skills
├── parallel/             ← distributed and parallelism skills
├── model/                ← model integration and conversion skills
├── data/                 ← data pipeline skills
├── debug/                ← debugging and performance analysis skills
├── recipe/               ← recipe setup skills
├── experiment/           ← experiment analysis skills
├── config/               ← config generation guidance
├── git/                  ← git, review, and merge workflow skills
└── skills/               ← skill authoring guidance
```

## Skill Structure

Each skill is a folder:

```
skill-name/
├── SKILL.md              # required — workflow, steps, caveats
└── references/          # optional — full examples, test templates
    ├── example-xxx.md
    └── test-patterns.md
```

- Keep **SKILL.md** under ~500 lines; only the core workflow.
- Put detailed examples and templates in **references/**; the agent loads them when needed.

## How to Use

Tell the coding agent what you need and point it at the relevant skill:

```
Use @skills/<category>/<skill-name>/SKILL.md
```

The agent will follow the skill and generate the adapted code and tests.

Example:

```
Use @skills/git/pr-gate/SKILL.md
```

## Recipe-Local Skill Tests

When a skill changes a user recipe, the tests for that skill should live with that
recipe, not under `skills/` and not under some unrelated demo recipe.

Use this layout:

```text
recipes/<recipe>/
└── skill_tests/
    ├── skill_manifest.yaml
    └── <skill-id>/
        ├── test_spec.yaml
        ├── test_structure.py
        ├── test_runtime.py
        └── test_smoke.py
```

- `skill_tests/skill_manifest.yaml` tracks recipe-relevant training skills with statuses such as
  `pending`, `applied`, `failed`, and `not_applicable`, plus per-layer validation
  results for each individual skill.
- `test_spec.yaml` declares which layers are required for that applied skill.
- Tests must exercise the user's real recipe/model entrypoints with a minimal
  recipe-owned config or batch, not a separate toy model unrelated to that recipe.
- `test_structure.py` should at least verify recipe import, registry wiring, config
  schema validation, required slots, and logger/checkpoint hooks.
- `test_runtime.py` should build dataset, collator, model, optimizer, scheduler, and
  engine successfully, but it does not need to run training.
- `test_smoke.py` should cover one real step: forward, loss, backward, optimizer
  step, logger write, and checkpoint noop or temporary save.
- Recipe-local skill tests should import recipe modules through the explicit
  repository package path, such as `from recipes.<recipe>.configs.schema import ...`.
  Do not create `recipes/<recipe>/skill_tests/conftest.py` just to mutate `sys.path`
  for short imports like `from <recipe>...`; the skill test runner executes from
  the repository root, which is sufficient for `mvp_engine` and `recipes.<recipe>`
  imports.
- Prefer starting from `tests/test_structure_template.py`,
  `tests/test_runtime_template.py`, and `tests/test_smoke_template.py`. Copy them
  into `recipes/<recipe>/skill_tests/<skill-id>/` and keep edits minimal,
  usually just the import block plus skill-specific assertions.
- If the skill needs distributed execution, the copied `test_smoke.py` should
  use `multi_rank_distributed_env(...)` from `tests/test_smoke_template.py` and
  configure the distributed mode to match the skill requirement or user
  preference, such as DDP, FSDP2 sharding, or tensor parallel.
- Run them with `python -m tests.test_skills --recipe <recipe> --skill <skill-id>`.
- Recipe-local skill validation must run only in fresh subagents with
  `fork_context=false`. Do not run `python -m tests.test_skills ...` from the
  main agent's local terminal, background terminal sessions, or any other
  non-subagent shell fallback.
- Run `--layer structure` in one fresh subagent, wait for it to pass, then run
  `--layer runtime` in a new fresh subagent, and only then run `--layer smoke`
  in another new fresh subagent.
- The user should not need to ask for these tests explicitly. When an agent applies
  a skill to a user recipe, it should also add the matching recipe-local tests and
  try to run them by default.
- The agent should also initialize or update `skill_tests/skill_manifest.yaml` automatically,
  and leave a skill as `applied` once that skill's recipe-local tests succeed.
- The main agent should summarize the `structure` / `runtime` / `smoke` outcomes
  after those subagents finish.
- If `test_smoke.py` needs GPU resources, distributed launch conditions, or higher
  execution permissions and those are not currently available, the main agent
  should report the exact command for the user instead of asking the user to
  design the test flow.
- If a recipe-local test really requires a real GPU or distributed environment, it
  should fail with an actionable command for the user to run in that environment,
  not `skip`.

## Skill List

- `parallel/fsdp2-prefetching`: [parallel/fsdp2-prefetching/SKILL.md](parallel/fsdp2-prefetching/SKILL.md)
- `parallel/tensor-parallel`: [parallel/tensor-parallel/SKILL.md](parallel/tensor-parallel/SKILL.md)
- `training/model-compile`: [training/model-compile/SKILL.md](training/model-compile/SKILL.md)
- `training/gradient-checkpointing`: [training/gradient-checkpointing/SKILL.md](training/gradient-checkpointing/SKILL.md)
- `model/model-migration`: [model/model-migration/SKILL.md](model/model-migration/SKILL.md)
- `recipe/new-recipe-template`: [recipe/new-recipe-template/SKILL.md](recipe/new-recipe-template/SKILL.md)
- `git/pr-gate`: [git/pr-gate/SKILL.md](git/pr-gate/SKILL.md)
- `git/pr-feedback`: [git/pr-feedback/SKILL.md](git/pr-feedback/SKILL.md)
- `git/pr-skill-review`: [git/pr-skill-review/SKILL.md](git/pr-skill-review/SKILL.md)
- `experiment/analysis`: [experiment/analysis/SKILL.md](experiment/analysis/SKILL.md)
- `git/recipe-merge-repair`: [git/recipe-merge-repair/SKILL.md](git/recipe-merge-repair/SKILL.md)

## Adding a New Skill

1. Create `skill-name/SKILL.md` under the right category.
2. Document: when to use it, step-by-step workflow, key rules, common pitfalls.
3. Add at least one verified full example under `references/`.
4. Add test templates under `references/` if applicable.
