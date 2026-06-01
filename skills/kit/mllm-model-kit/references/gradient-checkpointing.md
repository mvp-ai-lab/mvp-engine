# MLLMModelKit Gradient Checkpointing

Use `apply_gradient_checkpointing(model, use_reentrant=False, mode="hf")` when
the model supports Hugging Face checkpointing.

Use `mode="custom"` or `"hf_with_custom"` only when the recipe passes
`target_modules` entries shaped as `ParentClass:child_name`. Custom wrapping uses
PyTorch checkpoint wrappers on matched child modules.

Checkpointing should run after model construction and recipe patches, and before
distributed wrapping. Validate with runtime smoke when possible because memory
and graph behavior are accelerator dependent.
