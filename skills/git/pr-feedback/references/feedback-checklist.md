# Checklist when replying to PR feedback

When the PR under review is about a **skill** (or touches skill files), use the dimensions below to ensure your replies address what reviewers care about. Same dimensions as the skill review checklist; here they are used to **respond** to feedback, not to perform the review.

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

- **Main workflow:** `skills/<category>/<skill-name>/SKILL.md`
- **Full example:** `references/example-*.md`
- **Test templates:** `references/test-patterns.md`

---

## Output for the author

- List **concrete issues and suggestions** with file/section references.
- If something is **good as-is**, say so briefly so the author knows what to keep.
