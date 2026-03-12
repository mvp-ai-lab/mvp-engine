# Recipe Patterns

按需读取这些先例，不要一次性把所有 recipe 都展开。

## 默认搜索命令

```bash
rg -n "torch\\.compile|optim\\.compile|compile_backend|compile_mode" recipes
```

## 先例 1: `vit_classification`

- 文件：`recipes/vit_classification/engine/vit_classification_engine.py`
- 模式：先 compile ，后并行化模型。
- 使用 `model.compile()`
