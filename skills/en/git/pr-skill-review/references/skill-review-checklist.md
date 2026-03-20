# Skill Review Checklist

Use this checklist when reviewing a **skill** (structured guide: Markdown + references) in this training-framework repo. Skills are followed by a coding agent to implement a feature for each new model when the feature has a fixed pattern but no single implementation—e.g. gradient checkpointing: "wrap each encoder layer in `torch.utils.checkpoint.checkpoint`" is the pattern, but encoder layout, layer types, and forward signatures differ per model. The skill documents the pattern once; the agent generates the concrete code per model.

---

## 1. Accuracy

- [ ] **Steps and code snippets** are correct for the domain (e.g. PyTorch APIs, config keys).
- [ ] No wrong or outdated API usage (imports, function names, signatures).
- [ ] **Edge cases** are covered where relevant (e.g. `output_attentions`, `use_reentrant` for checkpointing).
- [ ] **Ordering** is correct (e.g. freeze → checkpointing → FSDP/DDP) if the skill implies a sequence.

## 2. Completeness

- [ ] An agent (or human) could add the feature to a new model using **only** this skill.
- [ ] No missing steps (e.g. config wiring, enabling in the training loop).
- [ ] **Common pitfalls** are called out (e.g. "do not checkpoint the whole model," "preserve tuple/list returns").
- [ ] **References** are sufficient: full example, test templates, or pointers to existing implementations where needed.

## 3. Clarity and consistency

- [ ] Workflow is easy to follow (numbered steps, clear "do this then that").
- [ ] **Terms** are used consistently (e.g. Encoder, layer, `custom_forward`) across the skill and references.
- [ ] **English and Chinese versions** (if both exist) say the same thing; no contradictory or extra steps in one locale.

## 4. Fit with skill philosophy

- [ ] The skill **avoids** prescribing a single generic API that tries to fit every model.
- [ ] It focuses on **pattern + per-model adaptation**: document the pattern once, point to examples/tests, let the agent generate model-specific code.
- [ ] It does **not** over-abstract (e.g. one "apply_checkpointing(model)" that hides all variation).

## 5. Test guidance

- [ ] **Test templates** (if any) are sufficient: enable/disable state, actual usage of the feature, and where applicable numerical/gradient checks.
- [ ] No misleading or incomplete test steps (e.g. "just run the test" without saying what to assert).
- [ ] Gaps are called out (e.g. "add a test that enables checkpointing and compares gradients to non-checkpoint run").

## 6. Skill location (for reference)

When commenting, you can refer to:

- **Main workflow:** `skills/<category>/<skill-name>/SKILL.md` (and `SKILL.zh-CN.md`)
- **Full example:** `references/example-*.md` (and `.zh-CN.md`)
- **Test templates:** `references/test-patterns.md` (and `.zh-CN.md`)

---

## Output for the author

- List **concrete issues and suggestions** with file/section references.
- If something is **good as-is**, say so briefly so the author knows what to keep.
