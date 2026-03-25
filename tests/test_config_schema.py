from pathlib import Path

from omegaconf import OmegaConf

from mvp_engine.config.schema import BaseEngineConfig


def test_default_config_validates_with_base_engine_schema():
    config_path = Path(__file__).resolve().parents[1] / "mvp_engine" / "config" / "default.yaml"
    config = OmegaConf.load(config_path)

    validated = BaseEngineConfig.model_validate(OmegaConf.to_container(config, resolve=True))

    assert validated.engine == "Engine"
    assert validated.parallel.backend_kwargs.fsdp2.reshard_after_forward is True
    assert validated.parallel.backend_kwargs.ddp.model_dump() == {}
