# Chapter V: Recipe

A recipe is the experiment boundary. It contains the code and config that make one training workflow concrete.

## Config System

Each recipe usually has two config layers:

- `configs/train.yaml` stores runnable values.
- `configs/schema.py` validates those values with Pydantic.

The YAML chooses the engine:

```yaml
engine: ViTClassificationEngine
```

The schema extends the shared base config:

```python
class ViTClassificationConfig(BaseEngineConfig):
    data: ViTDataConfig
    model: ViTModelConfig
```

The launcher loads YAML with Hydra, merges shared defaults, validates the final config, then builds the registered engine.

This means common fields stay consistent while each recipe can add its own `data`, `model`, or training options.
