---
name: model-migration
description: 将外部模型迁移到 mvp-engine 的 recipe 中，并保持行为一致、checkpoint 严格兼容，以及可选的 NPU 支持。适用于把 modeling 和 configuration 代码迁入 recipes/。
---

# Model Migration

## Goal

- 在不改变数学逻辑和参数命名的前提下，把源模型迁移到 `recipes/<recipe>/model/`。
- 保持 checkpoint 兼容性，使现有权重能通过 `strict=True` 严格加载且没有 key mismatch。
- 把一致性测试放到 recipe-local 的 `tests/` 下，而不是依赖全局 `tests/`。

## Required Inputs

- 源 `modeling_*.py`、`configuration_*.py` 以及需要兼容的 checkpoint 文件。
- 目标 recipe 路径 `recipes/<recipe>/`。
- 会实例化迁移后模型的运行时入口或 builder。
- 是否需要额外提供 NPU 版本。
- 能运行一致性测试和严格加载校验的环境。

## Workflow

### 1. 定位并指纹校验源资产

- 在修改前先找到源 modeling、configuration 和 checkpoint 目录。
- 当存在多个候选副本时，用 hash 和 diff 确认哪个才是权威实现。

```bash
sha256sum SOURCE_MODEL.py CHECKPOINT_DIR/modeling_*.py TARGET_MODEL.py
diff -u SOURCE_MODEL.py CHECKPOINT_DIR/modeling_*.py
```

- 如果两份文件字节级一致，就把该实现视为权威来源。

### 2. 先迁移 CPU 或 GPU 版本

- 以最小接入改动把源 `configuration_*.py` 和 `modeling_*.py` 迁入目标 recipe。
- 除非 recipe 接入边界强制要求，否则模块名、类名和参数名都保持不变。
- 只有在基础模型已经能编译后，再更新 `__init__.py` 导出和 builder。
- 迁移阶段不要顺手做重构；先保证接入正确，再谈优化。

### 3. 仅在需要时添加 NPU 版本

- 如果用户需要 NPU 支持，从 CPU 或 GPU 实现近似复制出 `modeling_*_npu.py`。
- NPU 特有替换应限制在尽可能小的代码块里，例如 fused rotary、norm 或 attention。
- 保持参数名和模块结构一致，确保同一份 checkpoint 可以加载到两套实现上。
- 优先使用带 fallback 的 `torch_npu` 导入，并且只有在 tensor 真正在 NPU 上时才走 fused op。
- 非 NPU 路径必须保留严格一致的数学 fallback。

### 4. 添加 recipe-local 一致性测试

- 把迁移测试放到 `recipes/<recipe>/tests/`。
- 至少覆盖：
  - 源模型和迁移模型在支持输入上的一致性
  - 如果存在 NPU 版本，共享权重下 CPU 或 GPU 类与 NPU 类的一致性
- 当迁移要求严格一致时，使用 `torch.equal` 这类严格断言，而不是宽松比较。

### 5. 校验 checkpoint 兼容性

- 对每个迁移后的类执行 `load_state_dict(..., strict=True)` 检查。
- 对 checkpoint 目录执行 `from_pretrained(...)` 冒烟测试。
- 如果严格加载失败，就对比模型 `state_dict().keys()` 与 checkpoint keys，并优先修复迁移模型中的命名或结构问题，而不是先改 checkpoint。

### 6. 达到验收标准后再停止

- 不能只停在“代码能 import 或能编译”。
- 只有在行为一致、checkpoint 兼容和 recipe-local 验证都通过，或剩余缺口被明确写出后，才算完成。

## Validation

- 已验证源 modeling 和 configuration 的身份；若有偏差，已记录原因。
- 目标 recipe 下存在一致性测试，并在当前可用环境中通过。
- 严格加载通过，没有 missing 或 unexpected keys。
- `from_pretrained(...)` 能成功加载迁移类。
- 已运行覆盖迁移文件的 lint 或目标检查。

## Output

- 说明迁移或新建了哪些文件。
- 说明是否新增了 NPU 版本。
- 总结一致性测试和严格加载校验结果。
- 说明是否还存在环境缺口，例如没有 NPU 硬件导致无法做完整验证。

## Read On Demand

- 这个 skill 没有附带的本地参考文件。需要时直接读取源 modeling 和 configuration 文件；只有做 NPU 特定算子替换时，再查 Ascend PyTorch for NPU 文档。
