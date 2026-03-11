# Docstring And Typing

## 必须更新的场景

- 已触达的公有函数或类缺少 docstring。
- 已触达的公有函数缺少参数或返回值类型标注。
- 签名变化影响了参数、默认值、可空性、容器结构、返回值或副作用。
- 已触达代码中的 docstring 或类型标注过时、错误，或比实际需要更宽泛。

## 期望修正范围

- 只处理本分支已经触达的文件、函数、类、模块。
- 优先使用具体标准类型，如 `list[str]`、`dict[str, int]`、`Path`。
- 仅在契约确实未知或有意保持动态时使用 `Any`。
- 私有且简单的 helper 可以不写 docstring，除非逻辑不直观。

## 保持一致

- docstring、类型标注、默认值、返回值与副作用应描述同一份行为契约。
- 避免写 "do something" 这类无信息描述。
- 若行为已变化但类型仍故意保持宽泛，应在代码注释或 PR 说明里交代原因。

## 推荐 Python 格式

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
