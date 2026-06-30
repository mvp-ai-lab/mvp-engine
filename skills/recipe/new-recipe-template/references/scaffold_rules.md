# New Recipe Scaffold Rules

Use this reference when creating a recipe scaffold manually.

## Naming

For `recipe_name`:

- directory: `recipes/<recipe_name>`;
- engine file: `engine/<recipe_name>_engine.py`;
- engine class: `<RecipeNamePascalCase>Engine`;
- config class: `<RecipeNamePascalCase>Config`;
- default config: `configs/train.yaml` unless the user chooses another name.

## Minimal Files

Required layout:

```text
README.md
__init__.py
configs/__init__.py
configs/schema.py
configs/<config_name>.yaml
dataset/__init__.py
engine/__init__.py
engine/<recipe_name>_engine.py
model/__init__.py
```

When tests are requested, copy and customize:

```text
tests/templates/test_structure.py.template -> recipes/<recipe>/tests/test_structure.py
tests/templates/test_smoke.py.template -> recipes/<recipe>/tests/test_smoke.py
```

Add these only when the recipe or a skill needs the corresponding layer:

```text
tests/templates/test_contract.py.template -> recipes/<recipe>/tests/test_contract.py
tests/templates/test_parity.py.template -> recipes/<recipe>/tests/test_parity.py
```

## Config Schema

Start from `BaseEngineConfig`:

```python
from pydantic import BaseModel, Field

from mvp_engine.config.schema import BaseEngineConfig


class <RecipeNamePascalCase>DataConfig(BaseModel):
    pass


class <RecipeNamePascalCase>ModelConfig(BaseModel):
    pass


class <RecipeNamePascalCase>Config(BaseEngineConfig):
    data: <RecipeNamePascalCase>DataConfig = Field(default_factory=<RecipeNamePascalCase>DataConfig)
    model: <RecipeNamePascalCase>ModelConfig = Field(default_factory=<RecipeNamePascalCase>ModelConfig)
```

Keep fields minimal until the task-specific implementation needs more.

## Engine Stub

Register the engine and leave unknown behavior explicit:

```python
from mvp_engine.engine import ENGINE_REGISTRY, Engine

from ..configs.schema import <RecipeNamePascalCase>Config


@ENGINE_REGISTRY.register()
class <RecipeNamePascalCase>Engine(Engine):
    ConfigClass = <RecipeNamePascalCase>Config

    def prepare_model(self):
        raise NotImplementedError("Implement recipe model construction.")
```

Implement every abstract method, but do not invent dataset/model logic.

## Test Customization

In `test_structure.py`, set:

- `RECIPE_IMPORT_PATH`;
- `CONFIG_CLASS_NAME`;
- `EXPECTED_FILES`;
- config filename if not `train.yaml`.

In `test_smoke.py`, set:

- `CONFIG_NAME`;
- `WORLD_SIZE`;
- `CONFIG_OVERRIDES`;
- timeout.

Smoke is expected to fail until real training behavior exists. Structure should
pass for a valid scaffold.

In `test_contract.py`, keep checks fast and semantic: AST/source/config/dataflow
invariants, executable binding evidence, and recipe-local skill contracts. Do
not call `engine.train()` or require accelerator resources.

In `test_parity.py`, validate metrics from real runs. Blocked/not-run artifacts
are unresolved states, not correctness passes.

Generate parity artifacts with recipe-local test runners that use existing
hooks, method wrappers, or generic observation surfaces. Do not modify
production recipe engine/model code only to collect parity metrics.

## Validation

Run:

```bash
python3 -m compileall recipes/<recipe>
pytest recipes/<recipe>/tests/test_structure.py -q
```

Run smoke only after replacing task-specific stubs.
