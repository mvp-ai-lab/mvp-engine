## Skills

`skills` are the alchemist's ingredients.

Users can use skills to add features to their experiments. A skill is useful when the pattern is common, but each model or recipe needs slightly different code.

## Recipes

`recipes` are the actual experiment code. just like the potion formulas.

Each recipe contains the experiment-specific data, model wiring, configs, and training workflow. If the logic belongs only to one experiment, it should usually live in `recipes/<recipe-name>/`.

![overview figure](/01-overview-fig-cutout.png)