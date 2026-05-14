"""Recipe-level cumulative structure tests for installed skills."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from mvp_engine.engine import ENGINE_REGISTRY
from mvp_engine.test.recipe_probe import import_modules, load_config
from mvp_engine.utils import skill_testing_util
from recipes.magic_transformer.configs.schema import MagicTransformerConfig

RECIPE_PATH = Path("recipes/magic_transformer")


def test_recipe_structure_matches_installed_skill_assertions() -> None:
    raw_config = load_config(RECIPE_PATH)
    config = MagicTransformerConfig.model_validate(OmegaConf.to_container(raw_config, resolve=True))

    import_modules(RECIPE_PATH)
    engine_cls = ENGINE_REGISTRY.get("MagicTransformerEngine")

    for skill_id, asserts in _load_skill_asserts():
        assert_structure = asserts.get("assert_structure")
        if assert_structure is None:
            raise AssertionError(f"{skill_id} asserts.py must define assert_structure.")
        assert_structure(recipe_root=RECIPE_PATH, config=config, engine_cls=engine_cls)


def _load_skill_asserts() -> tuple[tuple[str, dict[str, Any]], ...]:
    return tuple(
        (skill_id, runpy.run_path(str(asserts_path)))
        for skill_id, asserts_path in skill_testing_util.get_ordered_skill_asserts(RECIPE_PATH)
    )
