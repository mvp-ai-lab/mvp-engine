## Directory Layout

Recipe code should stay under `recipes/<name>/`. The core engine should not need changes for one experiment.

A typical layout is:

```text
recipes/vit_classification/
|-- configs/
|   |-- train.yaml
|   `-- schema.py
|-- dataset/
|   |-- imagenet.py
|   `-- sampler.py
|-- engine/
|   `-- vit_classification_engine.py
|-- model/
|   `-- vit.py
`-- README.md
```

## Responsibilities

- `configs/` defines knobs and defaults.
- `dataset/` builds samples, transforms, samplers, and loaders.
- `model/` builds the network and model-local helpers.
- `engine/` connects data, model, optimizer, scheduler, and step hooks.

This layout keeps experiments isolated while still reusing the shared launcher, logging, checkpointing, and distributed utilities.
