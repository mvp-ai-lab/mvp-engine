# Docstring Rules

## 必须补齐的场景

- 新增公有函数/类/模块。
- 函数签名（参数、默认值、返回值）发生变化。
- 增加副作用（写文件、网络 IO、状态修改）。

## 推荐格式（Python）

```python
def build_loader(cfg: LoaderConfig) -> DataLoader:
    """Build a dataloader from runtime config.

    Args:
        cfg: Runtime loader configuration.

    Returns:
        Configured PyTorch dataloader.
    """
```

## 反例

- 仅写 "do something" 这类无信息描述。
- 参数与代码不一致，或遗漏异常/副作用说明。
