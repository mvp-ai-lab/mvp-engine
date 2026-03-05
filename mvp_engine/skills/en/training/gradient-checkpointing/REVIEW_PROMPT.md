# Prompt: Review the Gradient Checkpointing Skill

Use the text below as the **system or user prompt** when asking an AI to review this repo’s gradient checkpointing skill. You can append “Review the following files:” and paste or @-mention the skill files, or rely on the AI having access to the repo.

---

## Prompt (English)

You are reviewing a **skill** in a training-framework repo. Skills are structured guides (Markdown + references) that a coding agent follows to implement a feature for each new model. They are used when the feature has a **fixed pattern but no single implementation**—e.g. gradient checkpointing: “wrap each encoder layer in `torch.utils.checkpoint.checkpoint`” is the pattern, but encoder layout, layer types, and forward signatures differ per model. So the skill documents the pattern once; the agent generates the concrete code per model.

**What to review**

1. **Accuracy**  
   Are the steps and code snippets correct for PyTorch gradient checkpointing? Any wrong or outdated API usage, missing edge cases (e.g. `output_attentions`, `use_reentrant`), or incorrect ordering (freeze → checkpointing → FSDP/DDP)?

2. **Completeness**  
   Can an agent (or human) add gradient checkpointing to a new model using only this skill? Missing steps, missing “common pitfalls,” or missing references (e.g. test templates, full example)?

3. **Clarity and consistency**  
   Is the workflow easy to follow? Are terms used consistently (e.g. Encoder, layer, `custom_forward`)? Do the English and Chinese versions say the same thing?

4. **Fit with the skill philosophy**  
   Does it avoid prescribing a single generic API? Does it focus on “pattern + per-model adaptation” and point to examples/tests rather than abstracting everything into one function?

5. **Test guidance**  
   Are the test templates sufficient (enable/disable state, actual checkpoint usage, gradient numerical match)? Any gaps or misleading steps?

**Skill location (for reference)**  
- Main workflow: `mvp_engine/skills/training/gradient-checkpointing/SKILL.md` (and `SKILL.zh-CN.md`)  
- Full example: `references/example-tomatovit.md` (and `.zh-CN.md`)  
- Test templates: `references/test-patterns.md` (and `.zh-CN.md`)  

Please list concrete issues and suggestions (with file/section references). If something is good as-is, say so briefly so the author knows what to keep.

---

## Prompt (中文)

你在审阅一个训练框架仓库里的 **skill**。Skill 是给 coding agent 用的结构化指南（Markdown + 参考资料），用来为每个新模型实现某个功能。适用于**模式固定、但无法写成一个通用实现**的情况——例如 gradient checkpointing：「在 Encoder 逐层循环里对每层包一层 `torch.utils.checkpoint.checkpoint`」是模式，但不同模型的 encoder 结构、layer 类型、forward 签名都不同，所以 skill 只写清模式一次，由 agent 针对每个模型生成具体代码。

**请从下面几个维度审阅：**

1. **正确性**  
   步骤和代码片段是否符合 PyTorch gradient checkpointing 的用法？有无错误或过时的 API、遗漏的边界情况（如 `output_attentions`、`use_reentrant`），或顺序错误（freeze → checkpointing → FSDP/DDP）？

2. **完整性**  
   仅凭这份 skill，agent（或人）能否为一个新模型加上 gradient checkpointing？是否缺少步骤、缺少「常见陷阱」或缺少参考资料（如测试模板、完整示例）？

3. **清晰与一致**  
   工作流是否容易跟随？术语是否一致（如 Encoder、layer、`custom_forward`）？中英文内容是否一致？

4. **与 skill 理念的契合**  
   是否避免了「强行做成一个通用 API」？是否突出「模式 + 按模型适配」，并以示例和测试为主，而不是把所有逻辑塞进一个抽象函数？

5. **测试指导**  
   测试模板是否足够（enable/disable 状态、是否真的调用了 checkpoint、梯度数值一致性）？有无遗漏或容易误导的步骤？

**Skill 位置（供你定位）**  
- 主工作流：`mvp_engine/skills/training/gradient-checkpointing/SKILL.md`（及 `SKILL.zh-CN.md`）  
- 完整示例：`references/example-tomatovit.md`（及 `.zh-CN.md`）  
- 测试模板：`references/test-patterns.md`（及 `.zh-CN.md`）  

请给出具体问题和改进建议（注明文件/段落）。若某部分已经很好，也简要指出，方便作者保留。
