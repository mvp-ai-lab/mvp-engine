# ViT Classification 控制组示例

当你需要一个“已经基本对齐当前共享层”的 recipe 作为控制组，来判断 breakage 到底来自共享层还是来自目标 recipe 本身时，优先参考这个例子。

## 为什么它适合作为控制组

`recipes/vit_classification/` 在当前仓库里具备几个很有价值的特征：

- 有清晰的 recipe-local schema：`recipes/vit_classification/configs/schema.py`
- engine 显式设置了 `ConfigClass`
- dataset / model / engine 三层分工清楚，入口比较短
- `parallelize_model(...)` 的调用方式已经对齐当前共享 helper
- 配置里已经使用当前顶层 `checkpoint` 布局
- dataset 提供 `use_fake_data` 路径，便于在缺真实数据时做轻量验证

这使它很适合当作“当前共享契约下，健康 recipe 大致应该长什么样”的参照物。

## 这个例子在 merge-repair 里怎么用

当另一个 recipe merge 后坏掉时，可以把 `vit_classification` 当成控制组去问三类问题：

1. 共享层的当前用法是什么。
   - 例如 engine 如何拿到经过校验的 config
   - model 如何走当前的并行入口
   - dataset 如何提供一个可验证的最小路径
2. 目标 recipe 偏离控制组的地方在哪里。
   - 是入口、schema、helper 调用方式不同
   - 还是确实因为 recipe 业务逻辑更复杂
3. 当前失败到底是 merge breakage，还是验证路径本身选得不好。
   - 如果控制组都跑不起来，优先怀疑共享层或验证环境
   - 如果控制组能跑、目标 recipe 不行，就继续缩小到 recipe-local hotspot

## 从仓库现状里可以提炼出的具体经验

- `recipes/vit_classification/engine/vit_classification_engine.py` 展示了当前共享 engine 契约下比较干净的 prepare pattern：先 build dataset/model，再调用共享 parallel helper。
- `recipes/vit_classification/dataset/imagenet.py` 展示了一个很有用的验证策略：在真实依赖不可用时，保留 fake-data 路径，让 smoke test 不被外部数据阻塞。
- `recipes/vit_classification/configs/train.yaml` 说明“健康 recipe”也未必默认适合所有验证环境；例如当前 mesh 默认值更偏向多卡模板，在单卡 smoke 时往往需要临时 override。

这点很重要：控制组不是“永远零问题”的样板，而是一个能帮助你区分问题来源的参考系。

## skill 应该从这个例子吸收什么

- 当目标 recipe 很复杂时，先找一个健康、简单、已对齐共享层的 recipe 做对照，能大幅减少盲修。
- 一个好的控制组应该覆盖：
  - 当前共享配置入口
  - 当前共享运行时入口
  - 最小可行验证路径
- 如果控制组和目标 recipe 的差异只出现在 recipe-local 扩展层，那么 merge repair 的重点就应该回到这些扩展层，而不是继续怀疑整个 shared stack。

## 这个例子最有价值的经验

- 不是每个 merge breakage 都需要直接去翻复杂 recipe；有时先读一个健康 recipe，能更快看清“当前标准姿势”是什么。
- 控制组的意义不是替代目标 recipe，而是帮助 agent 判断：当前看到的是共享层回归、recipe-local 漂移，还是验证路径选择错误。
