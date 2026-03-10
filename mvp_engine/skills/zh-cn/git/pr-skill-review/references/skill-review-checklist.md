# Skill 评审清单

用于评审本训练框架仓库中的 **skill**（结构化指南：Markdown + references）。Skill 由编码 agent 按“固定模式、无单一实现”的场景使用，例如梯度 checkpoint：模式是“用 `torch.utils.checkpoint.checkpoint` 包装每个 encoder 层”，但不同模型的 encoder 结构、层类型和 forward 签名不同。Skill 只写一次模式；agent 为每个模型生成具体代码。

---

## 1. 准确性

- [ ] **步骤与代码片段**在该领域内正确（如 PyTorch API、配置键）。
- [ ] 无错误或过时的 API 用法（import、函数名、签名）。
- [ ] **边界情况**在适用处已覆盖（如 checkpoint 场景下的 `output_attentions`、`use_reentrant`）。
- [ ] **顺序**正确（若 skill 隐含先后关系，如 freeze → checkpointing → FSDP/DDP）。

## 2. 完整性

- [ ] 仅凭本 skill，agent（或人）即可在新模型上加入该能力。
- [ ] 无遗漏步骤（如配置接入、在训练循环中开启）。
- [ ] **常见坑**有说明（如“不要对整模型做 checkpoint”“保持 tuple/list 返回值”等）。
- [ ] **参考资料**足够：完整示例、测试模板或指向现有实现的引用。

## 3. 清晰与一致

- [ ] 流程易跟随（步骤编号、清晰的“先做 A 再做 B”）。
- [ ] **术语**在 skill 与 references 中一致（如 Encoder、layer、`custom_forward`）。
- [ ] **中英文版本**（若同时存在）表述一致，无矛盾或单边多出的步骤。

## 4. 与 skill 理念一致

- [ ] Skill **不**规定单一通用 API 以适配所有模型。
- [ ] 侧重 **模式 + 按模型适配**：写清模式一次，指向示例/测试，由 agent 生成模型相关代码。
- [ ] **不**过度抽象（例如避免一个“apply_checkpointing(model)”包住所有差异）。

## 5. 测试指引

- [ ] **测试模板**（若有）足够：开关状态、该能力的实际使用、以及适用的数值/梯度检查。
- [ ] 无误导或不完整的测试步骤（如只说“跑一下测试”而不说明要断言什么）。
- [ ] 缺口有说明（如“增加测试：开启 checkpoint 并与未开启时对比梯度”）。

## 6. Skill 位置（便于引用）

评论时可引用：

- **主流程：** `mvp_engine/skills/<category>/<skill-name>/SKILL.md`（及 `SKILL.zh-CN.md`）
- **完整示例：** `references/example-*.md`（及 `.zh-CN.md`）
- **测试模板：** `references/test-patterns.md`（及 `.zh-CN.md`）

---

## 给作者的输出

- 列出**具体问题与建议**，并注明文件/章节。
- 若有**可保留**之处，简要说明以便作者知道哪些无需改。
