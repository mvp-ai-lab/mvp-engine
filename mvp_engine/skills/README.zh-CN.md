# Skills

Skills 是本仓库的一种**新接口形式**，与代码接口（函数、类）并列存在。  
**English:** [README.md](README.md)

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
mvp_engine/skills/
├── README.md / README.zh-CN.md
├── training/               ← 训练技巧（需按模型适配）
│   └── gradient-checkpointing/
│       ├── SKILL.md / SKILL.zh-CN.md
│       └── references/
├── parallel/               ← 分布式/并行策略
├── model/                  ← 模型接入与转换
├── data/                   ← 数据管线接入
├── debug/                  ← 调试与性能分析
└── recipe/                 ← 新实验搭建流程
```

## Skill 的结构

每个 skill 是一个独立文件夹，中英文各一份：

```
skill-name/
├── SKILL.md              # 英文（必需）
├── SKILL.zh-CN.md        # 中文（可选）
└── references/
    ├── example-xxx.md
    ├── example-xxx.zh-CN.md
    ├── test-patterns.md
    └── test-patterns.zh-CN.md
```

- **SKILL.md / SKILL.zh-CN.md** 控制在约 500 行以内，只写核心工作流。
- 详细示例和模板放在 **references/**，agent 按需读取。

## 如何使用

对 coding agent 说明需求并引用对应 skill（中英文任选其一）：

```
给 MyNewViT 加 gradient checkpointing，参考 @mvp_engine/skills/training/gradient-checkpointing/SKILL.zh-CN.md
Add gradient checkpointing to MyNewViT using @mvp_engine/skills/training/gradient-checkpointing/SKILL.md
```

Agent 会按 skill 工作流为你的模型生成适配代码和测试。

## 如何新增 Skill

1. 在对应分类目录下创建 `skill-name/SKILL.md`（及可选 `SKILL.zh-CN.md`）。
2. 写明：适用场景、分步工作流、关键规则、常见陷阱。
3. 在 `references/` 中放至少一个已验证的完整示例（可同时提供中英文）。
4. 若有测试模板，放在 `references/`（可同时提供 test-patterns.md 与 test-patterns.zh-CN.md）。
