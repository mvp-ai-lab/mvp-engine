# Skills

Skills are agent-facing operating manuals for this repository. They sit next to
code interfaces such as `mvp_engine.kit`.

## Code, Kit, Skill

Use the simplest layer that fits:

```text
Stable reusable behavior?          -> implement or call a kit/API in mvp_engine/
Same task, variable glue?          -> document the workflow as a skill
One-off experiment behavior?       -> keep it in recipes/<recipe>/
```

A kit is a user/agent callable suite of APIs. A skill should explain how to use
that suite, what each API does, and where extension or override points live. A
skill should not ask the agent to reimplement behavior already covered by a kit.

Examples:

- `MLLMDataKit`, `MLLMSampleKit`, `MLLMMediaKit`, and `PackingOptions` cover the
  standard MLLM data pipeline.
- `MLLMModelKit` covers common MLLM model loading, patching, freeze,
  checkpointing, and compile wiring.
- `TokenNormedLossKit`, `MFUKit`, and `OptimKit` cover standard training
  utilities.
- TP plans, FSDP2 prefetch edges, model migration, and loss spike guards still
  rely on feature skills because they are model/recipe-specific patterns rather
  than stable generic kit APIs.

## Directory Layout

```text
skills/
├── kit/                  # how to use and extend mvp_engine.kit APIs
├── training/             # training technique workflows and kit-aware feature guides
├── parallel/             # distributed and parallelism workflows
├── model/                # model integration and kit-aware model feature guides
├── data/                 # data feature entrypoints that route to kit APIs
├── recipe/               # recipe setup workflows
├── experiment/           # run analysis workflows
└── git/                  # review, PR, and merge workflows
```

## Skill Structure

Each skill folder contains:

```text
skill-name/
├── SKILL.md
└── references/           # optional, loaded only when needed
```

Keep `SKILL.md` concise. Put detailed examples, formulas, and legacy fallback
patterns in `references/`.

## Current Skills

Kit APIs:

- `kit/mllm-data-kit`: [kit/mllm-data-kit/SKILL.md](kit/mllm-data-kit/SKILL.md)
- `kit/mllm-model-kit`: [kit/mllm-model-kit/SKILL.md](kit/mllm-model-kit/SKILL.md)
- `kit/token-loss-kit`: [kit/token-loss-kit/SKILL.md](kit/token-loss-kit/SKILL.md)
- `kit/mfu-kit`: [kit/mfu-kit/SKILL.md](kit/mfu-kit/SKILL.md)
- `kit/optim-kit`: [kit/optim-kit/SKILL.md](kit/optim-kit/SKILL.md)

Data:

- `data/vlm-data-pipeline`: [data/vlm-data-pipeline/SKILL.md](data/vlm-data-pipeline/SKILL.md)
- `data/vlm-packing`: [data/vlm-packing/SKILL.md](data/vlm-packing/SKILL.md)

Model:

- `model/gradient-checkpointing`: [model/gradient-checkpointing/SKILL.md](model/gradient-checkpointing/SKILL.md)
- `model/model-compile`: [model/model-compile/SKILL.md](model/model-compile/SKILL.md)
- `model/model-migration`: [model/model-migration/SKILL.md](model/model-migration/SKILL.md)
- `model/vlm-freeze-policy`: [model/vlm-freeze-policy/SKILL.md](model/vlm-freeze-policy/SKILL.md)

Training:

- `training/loss-spike-guard`: [training/loss-spike-guard/SKILL.md](training/loss-spike-guard/SKILL.md)
- `training/model-flops-utilization`: [training/model-flops-utilization/SKILL.md](training/model-flops-utilization/SKILL.md)
- `training/token-normalized-loss`: [training/token-normalized-loss/SKILL.md](training/token-normalized-loss/SKILL.md)

Parallel:

- `parallel/fsdp2-prefetching`: [parallel/fsdp2-prefetching/SKILL.md](parallel/fsdp2-prefetching/SKILL.md)
- `parallel/context-parallel`: [parallel/context-parallel/SKILL.md](parallel/context-parallel/SKILL.md)
- `parallel/sequence-parallel`: [parallel/sequence-parallel/SKILL.md](parallel/sequence-parallel/SKILL.md)
- `parallel/tensor-parallel`: [parallel/tensor-parallel/SKILL.md](parallel/tensor-parallel/SKILL.md)

Recipe, experiment, and git:

- `recipe/new-recipe-template`: [recipe/new-recipe-template/SKILL.md](recipe/new-recipe-template/SKILL.md)
- `experiment/experiment-analysis`: [experiment/experiment-analysis/SKILL.md](experiment/experiment-analysis/SKILL.md)
- `git/pr-feedback`: [git/pr-feedback/SKILL.md](git/pr-feedback/SKILL.md)
- `git/pr-gate`: [git/pr-gate/SKILL.md](git/pr-gate/SKILL.md)
- `git/recipe-merge-repair`: [git/recipe-merge-repair/SKILL.md](git/recipe-merge-repair/SKILL.md)

## Adding Or Updating A Skill

1. Check whether an existing kit API should be used first.
2. Keep feature skills as natural-language trigger points.
3. Link to the kit skill for the authoritative API contract.
4. Put model/recipe-specific fallback details in references.
