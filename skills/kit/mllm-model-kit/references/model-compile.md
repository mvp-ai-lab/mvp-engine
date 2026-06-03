# MLLMModelKit Compile

Use `apply_model_compile(model, backend, mode)` for the standard VLM compile
path. It disables the Qwen-style visual forward before compiling the model, then
calls `model.compile(backend=..., mode=...)`.

Use recipe-local compile code only when:

- the safe compile target is a submodule rather than the top-level model;
- the model has non-Qwen visual graph-break requirements;
- compile must happen after a wrapper for a documented reason.

Compile before FSDP/DDP/TP wrapping by default.
