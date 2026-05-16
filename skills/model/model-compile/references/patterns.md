# Model Compile Patterns

Use this reference when choosing compile scope or handling graph breaks.

## Top-Level Compile

Use top-level compile when `forward()` is mostly tensor/model math and all
required preprocessing is trace-friendly:

```python
if self.config.model.compile:
    model.compile(
        backend=self.config.model.compile_backend,
        mode=self.config.model.compile_mode,
    )
```

Place this after model loading, recipe-local patches, checkpointing hooks, and
freeze policy, but before distributed wrapping.

## Submodule Compile

Compile a submodule when the top-level forward performs Python-heavy work such
as tokenization glue, data-dependent sequence construction, packed metadata
creation, custom object assembly, or dynamic multimodal routing:

```python
if self.config.model.compile:
    model.backbone.compile(
        backend=self.config.model.compile_backend,
        mode=self.config.model.compile_mode,
    )
```

Keep the target large enough to matter. Compiling many tiny modules usually adds
complexity without improving throughput.

## Graph Breaks

If a known unsupported region is small and isolated, keep it eager:

```python
if self.config.model.compile:
    model.visual.forward = torch.compiler.disable(model.visual.forward)
    model.compile(
        backend=self.config.model.compile_backend,
        mode=self.config.model.compile_mode,
    )
```

Use graph breaks sparingly and document why the eager section is necessary.

## Placement Rules

- Compile before FSDP, DDP, tensor parallel, or recipe `parallelize_model(...)`
  unless the recipe has a documented reason to do otherwise.
- Do not change checkpoint keys, parameter names, or public model outputs for
  compile.
- Keep compile controlled by config so smoke tests can force it on or off.
- Track first-step compile latency separately from steady-state throughput when
  measuring performance.
