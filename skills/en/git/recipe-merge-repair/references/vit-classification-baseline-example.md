# ViT Classification Control-Group Example

Use this example when you need a recipe that is already mostly aligned with the current shared layer, so you can use it as a control group and determine whether a failure comes from shared infrastructure or from the target recipe itself.

## Why it works well as a control group

`recipes/vit_classification/` has several properties that are especially useful in this repository:

- a clear recipe-local schema in `recipes/vit_classification/configs/schema.py`
- an engine that explicitly sets `ConfigClass`
- a clean split across dataset / model / engine layers
- current `parallelize_model(...)` usage that matches the shared helper
- current top-level `checkpoint` layout in config
- a `use_fake_data` path that makes lightweight validation possible without real data

That makes it a strong reference point for “what a healthy recipe looks like under current shared contracts.”

## How to use this example during merge repair

When another recipe breaks after a merge, use `vit_classification` as a control group to ask three questions:

1. What does the current shared-layer usage look like?
   - How does the engine receive validated config?
   - How does the model enter the current parallelization path?
   - How does the dataset provide a minimal validation path?
2. Where does the target recipe diverge from the control group?
   - Is the difference in entrypoints, schema, or helper usage?
   - Or is the target recipe genuinely more complex in its local logic?
3. Is the failure really merge breakage, or is the chosen validation path itself bad?
   - If even the control group does not run, suspect shared infrastructure or the validation environment first.
   - If the control group runs and the target recipe does not, keep narrowing to recipe-local hotspots.

## Concrete lessons from the current repository state

- `recipes/vit_classification/engine/vit_classification_engine.py` shows a relatively clean prepare pattern under the current shared engine contract: build dataset/model first, then call the shared parallel helper.
- `recipes/vit_classification/dataset/imagenet.py` shows a very useful validation tactic: preserve a fake-data path so smoke tests are not blocked by unavailable external data.
- `recipes/vit_classification/configs/train.yaml` also shows that even a healthy recipe is not automatically perfect for every validation environment; for example, the checked-in mesh defaults are more naturally aimed at a multi-rank template, so a single-rank smoke run often needs a temporary override.

That last point matters: a control group is not a “zero-problem sample.” It is a reference frame that helps you classify where the real problem lives.

## What the skill should learn from this example

- When the target recipe is complex, starting from a healthy, simpler, already-aligned recipe can dramatically reduce blind repair work.
- A good control group should cover:
  - the current shared configuration entrypoint
  - the current shared runtime entrypoint
  - a minimum viable validation path
- If the difference between the control group and the target recipe lives mostly in recipe-local extensions, then the merge-repair effort should move back to those extensions rather than continue doubting the entire shared stack.

## The most important lesson from this example

- Not every merge breakage should start with the most complex recipe. Sometimes reading a healthy recipe first is the fastest way to recover the current “correct posture.”
- The role of a control group is not to replace the target recipe. It is to help the agent decide whether the current failure is a shared-layer regression, recipe-local drift, or simply a bad validation path.
