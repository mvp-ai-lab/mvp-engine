from typing import Mapping, Optional, Union


class Backend:
    """Abstract backend interface for logging and metrics.

    Implementations should override the public methods below. Metrics
    values may be numeric (float) or string (for last-value metrics).
    """

    def log_config(self, config: dict) -> None:
        """Persist or print configuration dictionary.

        Args:
            config: Configuration mapping (usually from Hydra/OmegaConf).
        """
        raise NotImplementedError

    def log_metrics(self, metrics: Mapping[str, Union[float, str]], step: int, epoch: Optional[int] = None) -> None:
        """Log aggregated metric values.

        Args:
            metrics: Mapping from metric name to aggregated value (float or str).
            step: Training step associated with the metrics.
            epoch: Optional epoch number.
        """
        raise NotImplementedError

    def destroy(self) -> None:
        """Optional cleanup when the backend is no longer needed."""

    def info(self, message: str) -> None:
        """Log an informational message (backend-specific)."""

    def warning(self, message: str) -> None:
        """Log a warning message (backend-specific)."""

    def error(self, message: str) -> None:
        """Log an error message (backend-specific)."""