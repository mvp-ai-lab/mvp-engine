import argparse
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from mvp_engine.engine import ENGINE_REGISTRY


@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    default_config = OmegaConf.to_container(
        OmegaConf.load(Path(__file__).parent / "config" / "default.yaml"),
        resolve=True,
    )
    config = OmegaConf.merge(OmegaConf.create(default_config), config)

    engine = ENGINE_REGISTRY.get(config.engine)(config)

    workflow = config.get("workflow", "train")
    if workflow == "train":
        engine.train()
    elif workflow == "evaluate":
        engine.evaluate()
    else:
        if hasattr(engine, workflow):
            getattr(engine, workflow)()
        else:
            raise ValueError(f"Unknown workflow: {workflow}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args, remaining = parser.parse_known_args()

    config_path = Path(args.config).resolve()
    config_dir = str(config_path.parent)
    config_name = config_path.stem

    recipe_dir = config_path.parent.parent
    if recipe_dir.exists() and recipe_dir.is_dir():
        import importlib.util
        for py_file in recipe_dir.glob("**/*.py"):
            if py_file.name.startswith("_"):
                continue
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

    sys.argv = [
        sys.argv[0],
        f"--config-path={config_dir}",
        f"--config-name={config_name}",
    ] + remaining
    sys.argv.extend(
        [
            "hydra.run.dir=.",
            "hydra.output_subdir=null",
            "hydra/hydra_logging=disabled",
            "hydra/job_logging=disabled",
        ]
    )
    main()
