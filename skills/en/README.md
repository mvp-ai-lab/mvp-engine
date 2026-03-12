# Skills

Skills are a **second kind of interface** in this repo, alongside code interfaces (functions, classes).
**中文：** [README.md](../zh-cn/README.md)

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
├── en/                   ← English docs
│   ├── README.md
│   ├── training/, parallel/, model/, data/, debug/, recipe/, config/, git/
│   └── ...
└── zh-cn/                   ← 中文文档
    ├── README.md
    ├── training/, parallel/, model/, data/, debug/, recipe/, config/, git/
    └── ...
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

Tell the coding agent what you need and point it at the skill (under `en/` or `zh-cn/` by language):

```
Use @skills/en/<category>/<skill-name>/SKILL.md
```

The agent will follow the skill and generate the adapted code and tests.

Example:

```
Use @skills/en/git/pr-gate/SKILL.md
```

## Skill List

- `model/model-migration`: [model/model-migration/SKILL.md](model/model-migration/SKILL.md)
- `git/pr-gate`: [git/pr-gate/SKILL.md](git/pr-gate/SKILL.md)
- `git/pr-feedback`: [git/pr-feedback/SKILL.md](git/pr-feedback/SKILL.md)
- `git/pr-skill-review`: [git/pr-skill-review/SKILL.md](git/pr-skill-review/SKILL.md)

## Adding a New Skill

1. Create `skill-name/SKILL.md` under the right category.
2. Document: when to use it, step-by-step workflow, key rules, common pitfalls.
3. Add at least one verified full example under `references/`.
4. Add test templates under `references/` if applicable.
