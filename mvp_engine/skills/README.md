# Skills

Skills are a **second kind of interface** in this repo, alongside code interfaces (functions, classes).  
**中文：** [README.zh-CN.md](README.zh-CN.md)

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
mvp_engine/skills/
├── README.md               ← this file
├── training/               ← training techniques (model-specific adaptation)
│   └── gradient-checkpointing/
│       ├── SKILL.md
│       └── references/
├── parallel/               ← distributed / parallelism strategies
├── model/                  ← model integration and conversion
├── data/                   ← data pipeline integration
├── debug/                  ← debugging and performance analysis
└── recipe/                 ← new experiment setup workflow
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

Tell the coding agent what you need and point it at the skill:

```
Add gradient checkpointing to MyNewViT using @mvp_engine/skills/training/gradient-checkpointing/SKILL.md
```

The agent will follow the skill and generate the adapted code and tests for your model.

## Adding a New Skill

1. Create `skill-name/SKILL.md` under the right category.
2. Document: when to use it, step-by-step workflow, key rules, common pitfalls.
3. Add at least one verified full example under `references/`.
4. Add test templates under `references/` if applicable.
