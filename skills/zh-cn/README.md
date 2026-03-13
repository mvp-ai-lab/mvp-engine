# Skills

Skills 是本仓库的一种**新接口形式**，与代码接口（函数、类）并列存在。
**English:** [README.md](../en/README.md)

---

## 核心思路：面向 Agent 的代码仓

本仓库按 **coding agent 优先** 的方式设计。对外暴露的不只是「可执行的代码」，还有 **agent 能按步骤执行的、结构化的工作流**。很多训练相关能力都有清晰、可重复的**模式**（例如「在 Encoder 逐层循环里对每层做 checkpoint 包装」），但**具体实现**依赖每个模型的结构（encoder 怎么组织、有哪些 layer 类型、forward 签名如何）。若全部压成一个通用 API，容易过度抽象、难读难维护；若每个模型都手写一遍，又重复且易错。**Skill** 是折中：把「模式」写清楚一次（步骤、规则、示例、测试模板），由 agent 针对新模型或新 recipe 生成具体代码。于是代码仓 = **代码 + skills**：能抽象成单一 API 的用代码，适合「同一模式、不同胶水」的用 skill。这样核心引擎保持简单、可读、易扩展，而不必把各种变体都塞进封装里。

---

## 为什么需要 Skills

训练框架中有些功能**逻辑清晰、模式固定，但无法写成通用函数**——因为它们必须深入到每个模型的具体结构中去适配。

典型例子：gradient checkpointing。做法很明确（在 Encoder 逐层循环里包 `torch.utils.checkpoint.checkpoint`），但不同模型的 Encoder 结构、layer 类型、forward 参数签名都不同，强行抽象成一个 `apply_gradient_checkpointing(model)` 只会造成过度封装。

传统做法是要么硬写通用接口（复杂、难维护），要么每次手写（重复、易错）。**Skill 是第三条路**：把「怎么做」的知识结构化沉淀下来，交给 coding agent 按需为每个新模型生成适配代码。

## Code vs Skill 的判断标准

```
能泛化为统一 API？ ─── 是 ──→ 写成代码（函数/类），放在 mvp_engine/ 对应模块
        │
        否
        │
        ▼
有固定模式可遵循？ ─── 是 ──→ 写成 Skill
        │
        否
        │
        ▼
    保留在 recipe/ 中作为实验特定代码
```

**代码接口**示例：checkpoint 文件存取、日志工具、config 解析、分布式通信原语。
**Skill 接口**示例：gradient checkpointing 适配、FSDP wrap 策略、新模型接入、新 dataset 接入。

## 目录结构

```
skills/
├── README.md             ← 本说明（仓库总览）
├── en/                   ← 英文文档
│   ├── README.md
│   ├── training/, parallel/, model/, data/, debug/, recipe/, config/, git/
│   └── ...
└── zh-cn/                   ← 中文文档
    ├── README.md
    ├── training/, parallel/, model/, data/, debug/, recipe/, config/, git/
    └── ...
```

## Skill 的结构

每个 skill 是一个独立文件夹，在 en/ 与 zh-cn/ 下各有一份：

```
skill-name/
├── SKILL.md
└── references/
    ├── example-xxx.md
    └── test-patterns.md
```

- **SKILL.md** 控制在约 500 行以内，只写核心工作流。
- 详细示例和模板放在 **references/**，agent 按需读取。

## 如何使用

对 coding agent 说明需求并引用对应 skill（任选 en 或 zh-cn 路径）：

```
参考 @skills/zh-cn/<分类>/<skill-name>/SKILL.md
或 @skills/en/<category>/<skill-name>/SKILL.md
```

Agent 会按 skill 工作流生成适配代码和测试。

示例：

```
参考 @skills/zh-cn/git/pr-gate/SKILL.md
```

## Skill 列表

- `training/gradient-checkpointing`：[training/gradient-checkpointing/SKILL.md](training/gradient-checkpointing/SKILL.md)
- `model/model-migration`：[model/model-migration/SKILL.md](model/model-migration/SKILL.md)
- `git/pr-gate`：[git/pr-gate/SKILL.md](git/pr-gate/SKILL.md)
- `git/pr-feedback`：[git/pr-feedback/SKILL.md](git/pr-feedback/SKILL.md)
- `git/pr-skill-review`：[git/pr-skill-review/SKILL.md](git/pr-skill-review/SKILL.md)

## 如何新增 Skill

1. 在对应分类目录下，在 **en/** 与 **zh-cn/** 中分别创建 `skill-name/SKILL.md`。
2. 写明：适用场景、分步工作流、关键规则、常见陷阱。
3. 在 `references/` 中放至少一个已验证的完整示例。
4. 若有测试模板，放在 `references/`（如 test-patterns.md）。
