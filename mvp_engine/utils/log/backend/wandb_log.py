from pathlib import Path
from typing import Mapping, Optional, Union

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    from mvp_engine.utils.log import simple_info

    simple_info("Wandb is not installed!")
    _WANDB_AVAILABLE = False

from numbers import Number

from omegaconf import DictConfig, OmegaConf

from mvp_engine.utils.distributed.utils import is_main_process

from .backend import Backend


class WandbBackend(Backend):
    """Log metrics, configs, and messages to Weights & Biases.

    Args:
        id: Unique identifier for the run (display name).
        project: Name of the wandb project.
        entity: Wandb username or team name.
        config: Dictionary or DictConfig to initialize the run with.
        path: Directory for wandb local files.
    """

    def __init__(
        self,
        id: str,
        project: str,
        entity: Optional[str] = None,
        config: Optional[Union[dict, DictConfig]] = None,
        path: Optional[Path] = None,
    ) -> None:
        self.id = id
        self.enable = is_main_process() and _WANDB_AVAILABLE

        if self.enable:
            # init wandb run
            wandb.init(
                project=project,
                entity=entity,
                name=id,
                config=OmegaConf.to_container(config, resolve=True)
                if isinstance(config, DictConfig)
                else config,
                dir=str(path) if path else None,
                resume="allow",
            )

    def log_config(self, config: DictConfig) -> None:
        """Update wandb config with provided DictConfig."""
        if self.enable:
            conf_dict = OmegaConf.to_container(config, resolve=True)
            wandb.config.update(conf_dict, allow_val_change=True)

    def log_metrics(
        self,
        metrics: Mapping[str, Union[float, str]],
        step: int,
        epoch: Optional[int] = None,
    ) -> None:
        """Log metrics to wandb.

        Args:
            metrics: Mapping of metric names to values.
            step: Current training step (used as the x-axis).
            epoch: Optional epoch index (added as a metric).
        """
        if not self.enable or len(metrics) == 0:
            return

        log_dict = {}
        if epoch is not None:
            log_dict["epoch"] = epoch

        for k, v in metrics.items():
            if k == "eta":
                continue

            if isinstance(v, Number):
                log_dict[k] = v
            # torch.Tensor or np.ndarray to scalar
            elif hasattr(v, "item"):
                try:
                    log_dict[k] = v.item()
                except (ValueError, RuntimeError):
                    continue
        if log_dict:
            wandb.log(log_dict, step=step)

    def destroy(self) -> None:
        """Finish the wandb run."""
        if self.enable:
            wandb.finish()
