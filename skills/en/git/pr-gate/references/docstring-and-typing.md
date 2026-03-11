# Docstring And Typing

## Must-update cases

- A touched public function or class is missing a docstring.
- A touched public function is missing parameter or return type hints.
- Signature changes affect arguments, defaults, nullability, container shape, return values, or side effects.
- Existing docstrings or annotations in touched code are stale, incorrect, or broader than necessary.

## Expected cleanup scope

- Limit edits to files, functions, classes, and modules already touched by the branch.
- Prefer concrete standard types such as `list[str]`, `dict[str, int]`, and `Path`.
- Use `Any` only when the contract is genuinely unknown or intentionally dynamic.
- Private trivial helpers may skip docstrings unless the logic is non-obvious.

## Keep in sync

- Docstrings, type hints, defaults, return values, and side effects should describe the same contract.
- Avoid filler docstrings such as "do something".
- If behavior changes but typing is intentionally broad, explain why in code comments or PR notes.

## Suggested Python format

```python
def build_loader(
    dataset: Dataset,
    batch_size: int,
    image_size: tuple[int, int],
) -> DataLoader:
    """Build a dataloader for image classification.

    Args:
        dataset: Dataset whose samples contain image tensors shaped [3, H, W]
            and integer class labels.
        batch_size: Number of samples per batch.
        image_size: Target image size as (height, width).

    Returns:
        A PyTorch dataloader that yields batches with:
            images: float32 tensor of shape [B, 3, H, W]
            labels: int64 tensor of shape [B]
    """
```
