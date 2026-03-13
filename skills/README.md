# Skills

Structured task guidance for coding agents. Documentation is split by language:

- **English:** [en/README.md](en/README.md)
- **中文：** [zh-cn/README.md](zh-cn/README.md)

## Layout

```
skills/
├── README.md     ← this file (overview)
├── en/           ← English docs (README, training/, parallel/, model/, data/, debug/, recipe/, git/)
└── zh-cn/           ← 中文文档（结构同 en/）
```

Point the agent at a skill under `en/` or `zh-cn/` by language, e.g.
`@skills/en/<category>/<skill-name>/SKILL.md`

## Skill List

- `training/gradient-checkpointing`
  - English: [en/training/gradient-checkpointing/SKILL.md](en/training/gradient-checkpointing/SKILL.md)
  - 中文： [zh-cn/training/gradient-checkpointing/SKILL.md](zh-cn/training/gradient-checkpointing/SKILL.md)
- `model/model-migration`
  - English: [en/model/model-migration/SKILL.md](en/model/model-migration/SKILL.md)
  - 中文： [zh-cn/model/model-migration/SKILL.md](zh-cn/model/model-migration/SKILL.md)
- `git/pr-gate`
  - English: [en/git/pr-gate/SKILL.md](en/git/pr-gate/SKILL.md)
  - 中文： [zh-cn/git/pr-gate/SKILL.md](zh-cn/git/pr-gate/SKILL.md)
- `git/pr-feedback`
  - English: [en/git/pr-feedback/SKILL.md](en/git/pr-feedback/SKILL.md)
  - 中文： [zh-cn/git/pr-feedback/SKILL.md](zh-cn/git/pr-feedback/SKILL.md)
- `git/pr-skill-review`
  - English: [en/git/pr-skill-review/SKILL.md](en/git/pr-skill-review/SKILL.md)
  - 中文： [zh-cn/git/pr-skill-review/SKILL.md](zh-cn/git/pr-skill-review/SKILL.md)
