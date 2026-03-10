# Docstring Rules

## Must-update cases

- New public function/class/module is introduced.
- Function signature changes (args/defaults/returns).
- New side effects are introduced (file writes, network IO, state mutation).

## Suggested Python format

```python
def build_loader(cfg: LoaderConfig) -> DataLoader:
    """Build a dataloader from runtime config.

    Args:
        cfg: Runtime loader configuration.

    Returns:
        Configured PyTorch dataloader.
    """
```

## Anti-patterns

- Empty text such as "do something".
- Docstring drifting from actual behavior or missing side effects.
