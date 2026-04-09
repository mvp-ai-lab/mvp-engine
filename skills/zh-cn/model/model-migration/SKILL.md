---
name: model-migration
description: 将外部模型迁移到 mvp-engine 的 recipe 中，保证数学与输出严格一致，支持可选 NPU 版本，并进行严格 checkpoint 兼容性校验。适用于迁移 modeling/configuration 代码、验证 state_dict 键对齐、补充 recipe 内部一致性测试。
---

# Model Migration

## 目标

在不改变行为的前提下，将源模型迁移到 `recipes/<recipe>/model/`。

必须同时满足：
- 保持相同的数学逻辑和参数命名。
- 在相同输入与权重下产出完全一致的输出。
- 使用 `strict=True` 加载现有 checkpoint，且无 key mismatch。
- 迁移测试放在 recipe 目录下，而不是全局 `tests/`。

## 工作流

### 1. 定位并指纹校验源资产

- 找到源 `modeling_*.py`、`configuration_*.py` 与 checkpoint 文件（`.safetensors`/`.bin`）。
- 迁移前先通过 hash 与 diff 校验一致性。

```bash
sha256sum SOURCE_MODEL.py CHECKPOINT_DIR/modeling_*.py TARGET_MODEL.py
diff -u SOURCE_MODEL.py CHECKPOINT_DIR/modeling_*.py
```

如果文件字节级一致，视源模型代码为权威实现。

### 2. 先迁移 CPU/GPU 版本

- 根据用户要求将源 `configuration_*.py` 和 `modeling_*.py` 迁移到 recipe 的 model 目录。
- 除非是 recipe 接入所必需，模块/类名与参数名保持不变。
- 仅在基础模型可编译后，再更新 `__init__.py` 与 builder 导出。

硬性规则：迁移阶段不要做重构，只做最小接入修改。

### 3. 以最小改动添加 NPU 版本（可选）

- 可询问用户是否需要 NPU 支持，但不应作为迁移前置条件。
- 基于 CPU/GPU 版本近似复制出 `modeling_*_npu.py`。
- 仅在小范围隔离代码块中做 NPU 替换（例如 fused rotary/norm/attention）。
- 保持参数命名与模块结构和 CPU/GPU 对齐，确保单个 checkpoint 可同时加载。
- PyTorch for NPU 文档参考：https://www.hiascend.com/document/detail/zh/Pytorch/730/index/index.html

推荐模式：
- 以 fallback 方式导入 `torch_npu`。
- 仅在 tensor 设备为 NPU 时走 fused op。
- 非 NPU 设备严格走原始数学 fallback 路径。

### 4. 添加 recipe 内部一致性测试

测试放置于：
- `recipes/<recipe>/skill_tests/model-migration/`

至少补齐：
- `test_spec.yaml`：声明这个 skill 在该 recipe 上要求哪些测试层级。
- `test_structure.py`：至少验证 recipe import、registry 接线、config schema 校验、
  required slots，以及 logger/checkpoint hooks；还必须验证迁移后的 recipe 入口
  与迁移后的模型类接线存在。
- `test_runtime.py`：至少成功构建 dataset、collator、model、optimizer、
  scheduler 和 engine，且不直接启动训练；还必须验证与一致性相关的关键运行时
  路径能通过迁移后的 recipe 入口被触发。
- `test_smoke.py`：覆盖 1 个真实、recipe-owned 的 single step：forward、loss、
  backward、optimizer step、logger write，以及 checkpoint noop 或临时保存；
  还必须验证源实现 vs 迁移实现的一致性，以及通过迁移后 recipe 入口完成严格
  checkpoint load 覆盖。
- `test_smoke.py` 必须走该 skill 的完整真实能力路径：真实迁移后 recipe 入口、
  真实一致性校验，以及真实 checkpoint-load / logger / checkpoint 接线；禁止用
  monkeypatch、fake migrated model、fake load path 或类似测试桩把要验证的能力
  短路掉。
- 如果该 recipe 的 full-capability single-step 只能在 GPU、NPU 或分布式环境下
  成立，就把 smoke test 写成真实 launcher 测试，并在 `test_spec.yaml` 里把
  `gpu_preferred` 设为 `true`；不要为了在更弱环境里跑通而退化成 fake 逻辑。

至少覆盖：
- 源模型 vs 迁移模型在全部支持输入上的一致性。
- CPU/GPU 类 vs NPU 类（fallback 路径）在共享权重下的一致性。
- 通过迁移后 recipe 入口做严格 checkpoint load 覆盖。

当你在用户 recipe 上执行这个 skill 时，应默认自动补齐这些测试，不要要求用户自己再描述测试布局。
如果因为设备资源或执行权限限制而无法运行，直接把准确的 `python -m tests.test_skills` 命令
以及环境相关的启动命令返回给用户。

如果环境允许，分别在 CPU/GPU 与 NPU 设备上运行测试，验证实现间一致性。

一致性断言标准：
- 需要严格相等时使用 `torch.equal`。

### 5. 校验 checkpoint 兼容性

对两个类都执行严格加载测试。

```python
state = load_file(".../model.safetensors")
res = model.load_state_dict(state, strict=True)
assert len(res.missing_keys) == 0
assert len(res.unexpected_keys) == 0
```

同时执行：
- `ModelClass.from_pretrained(<checkpoint_dir>)` 冒烟测试。

若严格加载失败：
- 对比 `state_dict().keys()` 与 checkpoint keys。
- 修复迁移模型中的命名/结构不一致（除非绝对必要，不要修改 checkpoint）。

### 6. 最终验收清单

仅在以下都通过时交付：
- 已确认源 modeling/config 一致性（或已记录有理由的偏差）。
- 一致性测试通过。
- 迁移类与 NPU 类 strict load 通过。
- 无 missing/unexpected keys。
- Lint/测试通过。

## 常用命令

```bash
# 运行 recipe 内部测试
python -m tests.test_skills --recipe <recipe> --skill model-migration

# lint 迁移相关文件
uv run --with ruff ruff check recipes/<recipe>/model recipes/<recipe>/skill_tests/model-migration

# 查看改动文件
git status --short --untracked-files=all
```
