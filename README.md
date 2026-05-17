<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/banner.png">
    <source media="(prefers-color-scheme: light)" srcset="./assets/banner-light.png">
    <img alt="MVP Engine" src="./assets/banner.png">
  </picture>
</p>

<p align="center">
  <h1 align="center">MVP Engine</h1>
  <p align="center" style="font-weight: bold;">
    The Next-Generation Framework for Multimodal Model Training with Agents
  </p>
  <p align="center">
    by MVP Lab.
  </p>
</p>

## Overview

> **We HATE over-abstraction**.   

MVP Engine is a lightweight training engine for multimodal model research. Its core principle is simple: keep stable orchestration in `mvp_engine/`, and keep experiment-specific model, data, optimizer, scheduler, and training logic in `recipes/`.

Most training frameworks become heavily abstracted because they need to support every model family, data format, parallel strategy, and training trick through one reusable API surface. That pressure is real, but the result is often a deep stack of config switches, adapters, hooks, and indirection that makes simple experiments hard to read and hard to modify.

MVP Engine resolves this tension with **skills**. The core engine stays small and boring: launch, config merge, distributed setup, logging, checkpointing, and the training loop. Reusable but model-dependent patterns, such as tensor parallelism, gradient checkpointing, freeze policies, packing, loss guards, or migration steps, live as agent-facing `skills/` instructions. A coding agent applies those skills directly to the target recipe, generating concrete code where it belongs instead of forcing every variation into the core runtime.

## Design at a Glance

- **Engine as the orchestration layer**: `mvp_engine/engine/engine.py` defines the base `Engine` class and the
  train workflow (`before_train -> do_train -> after_train`). Subclasses implement `prepare_*` methods and
  step hooks such as `train_pre_step` and `forward_step`.
- **Core-only shared package**: common code in `mvp_engine/` should stay generic, minimal, and stable.
- **Hydra configuration**: `mvp_engine/launch.py` merges default config with recipe configs and launches the
  requested workflow (`train`, `evaluate`, or custom).
- **Logging system**: metrics are aggregated and dispatched to terminal/file backends; additional backends can
  be added with minimal changes.
- **Skills**: reusable code patterns that a coding agent can apply to recipes, such as parallelism, freeze policies. Skills are not part of the core engine but are available for recipe customization.
- **Recipe**: other stuffs such as dataset loading and preprocessing live in each recipe, so task-specific
  formats can evolve without adding brittle abstractions to the core engine.

## Agentic Workflow

1. Keep the core engine minimal and reusable (`mvp_engine/`).
2. Place task-specific model/data/training logic in `recipes/`.
3. Use a coding AI to execute relevant `skills/` (parallel, model, data, debug, recipe, etc.).
4. Let the AI assemble or modify recipe code/configs for your target training objective.

## Project Layout

- `mvp_engine` — core orchestration logic and Engine base class, tools such as logging, distributed helpers, training utilities
- `recipes/` — experiment-specific configs and custom engine/model/data definitions
- `skills/` — reusable agent skills used by coding AI to implement recipe customization patterns
- `outputs/` — run outputs, logs, and checkpoints


## Getting Started

```bash
uv venv --python=3.12
source .venv/bin/activate
uv sync

# Recipes that use `flash_attention_2` may require a FlashAttention wheel that
# matches the local Python, CUDA, PyTorch, and C++ ABI versions.

# Demo Training Command
torchrun --nproc_per_node=1 -m mvp_engine.launch --config ./recipes/magic_transformer/configs/train.yaml
```

## Development

```
uv pip install pre-commit
pre-commit install
```
