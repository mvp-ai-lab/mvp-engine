# Training

Skills for **training techniques** that require model-specific adaptation (e.g. gradient checkpointing, custom loss wiring). The pattern is consistent but the code touches each model’s structure.

- `model-compile`: add or adjust `model.compile` support for a recipe, including placement choice, extra-module coverage, and minimal validation.
Available:
- `gradient-checkpointing/SKILL.md`: Gradient checkpointing integration guide.
