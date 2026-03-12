# Recipe Patterns

Read these precedents on demand instead of expanding every recipe at once.

## Default search command

```bash
rg -n "torch\\.compile|optim\\.compile|compile_backend|compile_mode" recipes
```

## Precedent 1: `vit_classification`

- File: `recipes/vit_classification/engine/vit_classification_engine.py`
- Pattern: compile first, then parallelize the model.
- Use `model.compile()`.
