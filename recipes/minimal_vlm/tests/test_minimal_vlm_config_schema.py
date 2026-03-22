from pathlib import Path

from omegaconf import OmegaConf

from recipes.minimal_vlm.configs.schema import MinimalVLMConfig


def test_minimal_vlm_train_config_validates() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"
    config = OmegaConf.load(config_path)

    validated = MinimalVLMConfig.model_validate(OmegaConf.to_container(config, resolve=True))

    assert validated.engine == "MinimalVLMEngine"
    assert validated.data.loader_prefetch_factor == 2
    assert validated.parallel.mesh.replicate == -1
