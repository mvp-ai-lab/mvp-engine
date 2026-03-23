from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from recipes.minimal_vlm.configs.schema import MinimalVLMConfig


def test_minimal_vlm_train_config_validates() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"
    config = OmegaConf.load(config_path)

    validated = MinimalVLMConfig.model_validate(OmegaConf.to_container(config, resolve=True))

    assert validated.engine == "MinimalVLMEngine"
    assert validated.data.loader_prefetch_factor == 2
    assert validated.data.packing is False
    assert validated.parallel.mesh.replicate == -1


def test_minimal_vlm_schema_accepts_streaming_packing_buffer_size() -> None:
    validated = MinimalVLMConfig.model_validate(
        {
            "engine": "MinimalVLMEngine",
            "data": {"packing_buffer_size": -1},
        }
    )

    assert validated.data.packing_buffer_size == -1


def test_minimal_vlm_schema_rejects_removed_optimal_packing_strategy() -> None:
    with pytest.raises(ValidationError):
        MinimalVLMConfig.model_validate(
            {
                "engine": "MinimalVLMEngine",
                "data": {"packing_selection_strategy": "optimal"},
            }
        )
