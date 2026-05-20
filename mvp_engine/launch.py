import argparse
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from mvp_engine.engine import ENGINE_REGISTRY
from mvp_engine.patches import apply_all_patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECIPES_ROOT = PROJECT_ROOT / "recipes"
_RUNTIME_PATCHES_APPLIED = False


def _apply_runtime_patches() -> None:
    """Apply process-wide runtime patches before importing/wrapping training code."""

    global _RUNTIME_PATCHES_APPLIED
    if _RUNTIME_PATCHES_APPLIED:
        return
    apply_all_patches()
    _RUNTIME_PATCHES_APPLIED = True


def _find_recipe_dir(config_path: Path) -> Path | None:
    """Resolve the top-level recipe package for a config path under ``recipes/``."""
    config_path = config_path.resolve()
    try:
        relative_path = config_path.relative_to(RECIPES_ROOT)
    except ValueError:
        return None

    if len(relative_path.parts) < 2:
        return None

    recipe_dir = RECIPES_ROOT / relative_path.parts[0]
    if not recipe_dir.is_dir():
        return None

    return recipe_dir


def _import_recipe_modules(recipe_dir: Path) -> None:
    """Import all Python modules under a recipe so its engine/model registries are populated."""
    has_package_entry = (RECIPES_ROOT / "__init__.py").exists() and (recipe_dir / "__init__.py").exists()
    import_root = PROJECT_ROOT if has_package_entry else recipe_dir.parent
    relative_root = PROJECT_ROOT if has_package_entry else recipe_dir.parent

    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

    import importlib

    py_files = sorted(
        py_file
        for py_file in recipe_dir.glob("**/*.py")
        if not py_file.name.startswith("_") and py_file.name != "conftest.py" and "tests" not in py_file.parts
    )
    py_files = _filter_gitignored_paths(py_files)

    for py_file in py_files:
        relative_path = py_file.relative_to(relative_root)
        module_name = ".".join(relative_path.with_suffix("").parts)
        importlib.import_module(module_name)


def _filter_gitignored_paths(paths: list[Path]) -> list[Path]:
    """Drop paths ignored by Git so recipe auto-import follows .gitignore rules."""
    if not paths:
        return []

    project_root = PROJECT_ROOT.resolve()
    try:
        relative_paths = [path.resolve().relative_to(project_root).as_posix() for path in paths]
    except ValueError:
        return paths

    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        input="\n".join(relative_paths),
        capture_output=True,
        cwd=project_root,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        return paths

    ignored_paths = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return [
        path for path, relative_path in zip(paths, relative_paths, strict=True) if relative_path not in ignored_paths
    ]


@hydra.main(version_base=None)
def main(config: DictConfig) -> None:
    _apply_runtime_patches()

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
    _apply_runtime_patches()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args, remaining = parser.parse_known_args()

    config_path = Path(args.config).resolve()
    config_dir = str(config_path.parent)
    config_name = config_path.stem

    recipe_dir = _find_recipe_dir(config_path)
    if recipe_dir is not None:
        _import_recipe_modules(recipe_dir)

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
