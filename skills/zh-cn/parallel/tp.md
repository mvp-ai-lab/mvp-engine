# TP 模块配置手册（中文）

## 目标
为 `recipes/` 下新增模型生成 `<MODEL_NAME>_TP_MODULE_CONFIG`，并在模型类上绑定 `TP_MODULE_CONFIG = <MODEL_NAME>_TP_MODULE_CONFIG`。

## 本仓库运行时约束
- 运行入口：`mvp_engine/distributed/tp.py`。
- 必须是：`dict[str, object]`，映射关系为 `module.__class__.__name__ -> plan`。
- plan 结构：`dict[子模块线性层名, "col" | "row"]`。
- 子模块名必须与目标类 `named_children()` 返回一致。

## 分析步骤
1. 打开目标模型文件：`recipes/**/model/**/modeling_*.py`。
2. 找到训练实际使用的顶层模型类。
3. 找到重复计算块（Attention、MLP、分支 MLP、Projector 等）对应的类。
4. 在这些类的 `__init__` 中提取直接 `nn.Linear` 子层名称。
5. 按以下规则分配 TP 模式：
- 输入扩展类投影使用 `"col"`：`q_proj`、`k_proj`、`v_proj`、`qkv`、`fc1`、`up_proj`、`gate_proj`，以及 `_a/_b` 分支变体。
- 输出回收类投影使用 `"row"`：`out_proj`、`o_proj`、`proj_out`、`fc2`、`down_proj`、`wo`，以及 `_a/_b` 分支变体。
- 拿不准时，前置投影优先 `"col"`，回到 hidden size 的最后一层优先 `"row"`。

## 编辑模板
```python
<MODEL_NAME>_TP_MODULE_CONFIG: dict[str, object] = {
    "<AttentionClass>": {
        "q_proj": "col",
        "k_proj": "col",
        "v_proj": "col",
        "out_proj": "row",
    },
    "<MLPClass>": {
        "fc1": "col",
        "fc2": "row",
    },
}

class <TopModelClass>(...):
    TP_MODULE_CONFIG = <MODEL_NAME>_TP_MODULE_CONFIG
```

## 校验清单
- 配置里的类名与运行时 `module.__class__.__name__` 一致。
- 每个 plan key 都真实存在于该类子模块中。
- plan 的值仅使用 `"col"` 或 `"row"`。
- 顶层模型类包含 `TP_MODULE_CONFIG` 赋值。

## TomatoViT 参考
可直接参考 `recipes/tomatovit/model/tomato_vit/modeling_tomatovit.py`：
- 常量：`TOMATOVIT_TP_MODULE_CONFIG`
- 绑定：`TP_MODULE_CONFIG = TOMATOVIT_TP_MODULE_CONFIG`
