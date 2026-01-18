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
        # Add recipe_dir's parent to sys.path so relative imports work
        recipe_parent = str(recipe_dir.parent)
        if recipe_parent not in sys.path:
            sys.path.insert(0, recipe_parent)
        
        recipe_name = recipe_dir.name
        import importlib
        for py_file in sorted(recipe_dir.glob("**/*.py")):
            if py_file.name.startswith("_"):
                continue
            # Build the full module name relative to recipe_parent
            relative_path = py_file.relative_to(recipe_dir.parent)
            module_name = str(relative_path.with_suffix("")).replace("/", ".")
            importlib.import_module(module_name)

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
    try:
        main()
    except Exception as e:
        from mvp_engine.utils.log import simple_info

        simple_info(f"Exception occurred: {e}")
        raise e
    finally:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

