# MLLMModelKit Freeze Policy

`apply_freeze_policy()` is prefix based. Defaults target Qwen-style components:

- vision: `model.visual.patch_embed.`, `model.visual.blocks.`;
- projector: `model.visual.merger.`, `model.visual.deepstack_merger_list.`;
- language: `model.language_model.`, `lm_head.`.

For another model family, inspect `named_parameters()` and pass explicit
`vit_prefixes`, `projector_prefixes`, and `llm_prefixes`. Keep groups
deterministic and non-overlapping. Freeze before optimizer construction,
trainable-parameter dtype upcast, and distributed wrapping.
