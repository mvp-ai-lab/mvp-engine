"""File-based metric and message backend.

Writes logs and metrics into plain text files under the provided
`path` directory. Metrics values can be numeric or string; string
metrics are recorded as their last observed value.
"""

from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Union

from omegaconf import DictConfig, OmegaConf

from mvp_engine.distributed.utils import is_main_process

from .backend import Backend


class FileBackend(Backend):
    """Persist logs and metrics to files.

    Args:
        id: Identifier used in log lines and file names.
        path: Directory where logs/configs are stored.
    """

    def __init__(
        self,
        id: str,
        path: Path,
    ) -> None:
        self.path = path
        self.id = id

        self.enable = is_main_process()
        self.log_file = None

        if self.enable:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            self.log_file = open(path / f"log_{id}.log", "w")

    def log_config(self, config: DictConfig) -> None:
        """Write a YAML dump of `config` to a file."""
        if self.enable:
            with open(self.path / f"config_{self.id}.yaml", "w") as f:
                f.write(OmegaConf.to_yaml(config))

    def log_metrics(
        self,
        metrics: Mapping[str, Union[float, str]],
        step: int,
        epoch: Optional[int] = None,
    ) -> None:
        """Append aggregated metrics to the log file.

        Args:
            metrics: Mapping of metric names to aggregated values.
            step: Training step.
            epoch: Optional epoch index.
        """
        if not self.enable or self.log_file is None:
            return
        if len(metrics) == 0:
            return
        eta = metrics.pop("eta", None)
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        eta_str = f" | ETA {eta}" if eta is not None else ""
        epoch_str = f"Epoch {epoch} - " if epoch is not None else ""
        log_str = f"{date_str} | {self.id}{eta_str} | {epoch_str}Step {step:>8} || "
        for key, value in metrics.items():
            log_str += f"{key}: {value} | "

        self.log_file.write(log_str + "\n")
        self.log_file.flush()

    def destroy(self) -> None:
        """Close the log file if opened."""
        if self.enable and self.log_file is not None:
            self.log_file.close()

    def debug(self, message: str) -> None:
        """Write a debug-level message to the log file."""
        if self.enable and self.log_file:
            self.log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {self.id} | DEBUG | {message}\n")
            self.log_file.flush()

    def info(self, message: str) -> None:
        """Write an info-level message to the log file."""
        if self.enable and self.log_file:
            self.log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {self.id} | INFO  | {message}\n")
            self.log_file.flush()

    def warning(self, message: str) -> None:
        """Write a warning message to the log file."""
        if self.enable and self.log_file:
            self.log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {self.id} | WARN  | {message}\n")
            self.log_file.flush()

    def error(self, message: str) -> None:
        """Write an error message to the log file."""
        if self.enable and self.log_file:
            self.log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {self.id} | ERROR | {message}\n")
            self.log_file.flush()
