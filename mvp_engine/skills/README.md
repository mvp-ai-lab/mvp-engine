# Skills

Structured task guidance for coding agents. Documentation is split by language:

- **English:** [en/README.md](en/README.md)
- **中文：** [cn/README.md](cn/README.md)

## Layout

```
mvp_engine/skills/
├── README.md     ← this file (overview)
├── en/           ← English docs (README, training/, parallel/, model/, data/, debug/, recipe/)
└── cn/           ← 中文文档（结构同 en/）
```

Point the agent at a skill under `en/` or `cn/` by language, e.g.  
`@mvp_engine/skills/en/<category>/<skill-name>/SKILL.md`

**This branch:** adds the [gradient-checkpointing](en/training/gradient-checkpointing/SKILL.md) skill (en + cn).
