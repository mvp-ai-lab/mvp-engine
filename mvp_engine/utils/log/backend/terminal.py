import inspect
from datetime import datetime
from typing import Mapping, Optional

from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from mvp_engine.distributed.utils import is_main_process

from .backend import Backend


def _get_caller_info(depth: int = 2) -> str:
    """Return a short ``filename:function:line`` string for a caller.

    Args:
        depth: How many frames to walk back from the current frame.

    Returns:
        Truncated caller location string, or ``"unknown:unknown:0"`` when unavailable.
    """
    frame = inspect.currentframe()
    try:
        for _ in range(depth):
            if frame is not None:
                frame = frame.f_back
        if frame is None:
            return "unknown:unknown:0"

        filename = frame.f_code.co_filename.split("/")[-1]
        funcname = frame.f_code.co_name
        lineno = frame.f_lineno
        location = f"{filename}:{funcname}:{lineno}"

        # Truncate to 30 chars, keeping the tail
        if len(location) > 30:
            location = "..." + location[-27:]
        return location
    finally:
        del frame


class TerminalBackend(Backend):
    """Console logger using Rich with optional caller location display."""

    def __init__(
        self,
        id: str,
    ) -> None:
        """Initialize a terminal backend.

        Args:
            id: Identifier printed alongside log messages.
        """
        self.id = id
        self.enable: bool = is_main_process()
        self.console: Optional[Console] = None

        if self.enable:
            self.console = Console(color_system="auto")

    def log_config(self, config: DictConfig) -> None:
        """Pretty-print configuration when running on main process."""
        if self.enable:
            self.info("=" * 80)
            self.info("Configurations:")
            for line in f"{OmegaConf.to_yaml(config)}".splitlines():
                self.info(" " * 4 + line)
            self.info("=" * 80)

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: int,
        epoch: Optional[int] = None,
    ) -> None:
        """Print metric values with optional epoch and ETA.

        Args:
            metrics: Mapping of metric names to values; ``eta`` is stripped if present.
            step: Current global step.
            epoch: Optional epoch number.
        """
        if not self.enable or not metrics:
            return

        metrics_dict = dict(metrics)
        eta = metrics_dict.pop("eta", None)
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        eta_str = f" | [bold]ETA[/bold] {eta}" if eta is not None else ""
        epoch_str = f"[bold]Epoch[/bold] {epoch} - " if epoch is not None else ""

        parts = [
            f"[bold]{date_str}[/bold] | [bright_yellow]{self.id}[/bright_yellow]{eta_str} | {epoch_str}[bold]Step[/bold] {step:>8} ||",
        ]

        for key, value in metrics_dict.items():
            if abs(value) <= 0.0001 and value != 0:
                parts.append(f"[bold]{key}[/bold] {value:.4e} |")
            else:
                parts.append(f"[bold]{key}[/bold] {value:.4f} |")

        self.console.print(" ".join(parts), soft_wrap=True)

    def destroy(self) -> None:
        """Tear down backend resources (no-op for terminal backend)."""
        return None

    def _print_with_location(self, message: str, level: str, level_color: str, location: str) -> None:
        """Print message with caller location right-aligned at the end."""
        if self.console is None:
            return None
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        terminal_width = self.console.width or 120

        # Build the main content (without location)
        if level_color == "cyan":
            main_content = f"[bold]{date_str}[/bold] | [bright_yellow]{self.id}[/bright_yellow] | [{level_color}]{level: <5}[/{level_color}] | {message}"
        else:
            main_content = f"[bold]{date_str}[/bold] | [bright_yellow]{self.id}[/bright_yellow] | [{level_color}]{level: <5}[/{level_color}] | [{level_color}]{message}[/{level_color}]"

        # Calculate plain text length (without markup)
        plain_len = len(date_str) + len(self.id) + len(level) + len(message) + 15  # separators
        location_part = f"[dim cyan]{location}[/dim cyan]"
        location_len = len(location)

        # Calculate padding needed
        padding = terminal_width - plain_len - location_len - 1
        if padding > 0:
            self.console.print(f"{main_content}{' ' * padding}{location_part}", soft_wrap=True)
        else:
            self.console.print(f"{main_content} {location_part}", soft_wrap=True)

    def debug(self, message: str) -> None:
        """Log a debug message."""
        if self.enable:
            location = _get_caller_info(depth=3)
            self._print_with_location(message, "DEBUG", "dim", location)

    def info(self, message: str) -> None:
        """Log an informational message."""
        if self.enable:
            location = _get_caller_info(depth=3)
            self._print_with_location(message, "INFO", "cyan", location)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        if self.enable:
            location = _get_caller_info(depth=3)
            self._print_with_location(message, "WARN", "yellow", location)

    def error(self, message: str) -> None:
        """Log an error message."""
        if self.enable:
            location = _get_caller_info(depth=3)
            self._print_with_location(message, "ERR ", "red", location)
