# Skills

Structured task guidance for coding agents. Documentation is split by language:

- **English:** [en/README.md](en/README.md)
- **中文：** [zh-cn/README.md](zh-cn/README.md)

## Layout

```
skills/
├── README.md     ← this file (overview)
├── en/           ← English docs (README, training/, parallel/, model/, data/, debug/, recipe/)
└── zh-cn/           ← 中文文档（结构同 en/）
```

Point the agent at a skill under `en/` or `zh-cn/` by language, e.g.  
`@skills/en/<category>/<skill-name>/SKILL.md`
