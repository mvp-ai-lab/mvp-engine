# ViT Classification 基线示例

当你需要一个“当前仓库里已经健康对齐的 recipe”作为对照时，优先看这个例子。

## 为什么它适合作为基线

- `recipes/vit_classification/` 已经具备：
  - recipe-local `ConfigClass`
  - 可离线 smoke 的 fake data 支持
  - 当前版本的 `parallelize_model(...)` 调用方式
  - 当前版本的顶层 `checkpoint` 配置结构

因此，当另一个 recipe 在 merge 之后疑似损坏时，它是很好的对照样本。

## 验证中发现了什么

1. 这个 recipe 的代码路径本身已经兼容当前 engine/config 契约。
2. 但模板默认 mesh 不适合单卡 smoke：
   - `parallel.mesh.replicate: -1`
   - `parallel.mesh.shard: 8`
   - `parallel.mesh.tensor: 1`
3. 在 `WORLD_SIZE=1` 的验证里，这会推导出 `replicate=0`，并在 `DeviceMesh` 初始化阶段直接失败，还没执行到 recipe 逻辑。

## 修复形态

- 让模板默认值可以直接跑 smoke：
  - 把 `parallel.mesh.shard` 从 `8` 改成 `1`
  - 同时保留简短注释，说明多卡 FSDP2 训练时应主动把 `shard` 调大
- 让常用调参在 Hydra struct mode 下更顺手：
  - 在 `train.yaml` 中显式写出 `patch_size`、`num_channels`、`hidden_size`、`intermediate_size`、`num_hidden_layers`、`num_attention_heads`
  - 这样 smoke test 时可以直接写 `model.hidden_size=192` 这类普通 override

## 验证形态

- `python -m compileall recipes/vit_classification`
- 对 `train.yaml` 做 config/schema 组合验证
- GPU smoke 包括：
  - fake data
  - 本地构造模型（`load_pretrained_weights: false`）
  - 单卡 mesh
  - engine 启动和一次训练前向

## 这个 skill 应该吸收的经验

- 不是每个 recipe 都需要写修复代码。
- 有时真正的问题只是“默认验证路径和当前 world size 不兼容”。
- skill 应该区分三类问题：
  - 共享契约真的变了
  - recipe 逻辑真的坏了
  - 只是 smoke 用的 mesh / 默认值不匹配
