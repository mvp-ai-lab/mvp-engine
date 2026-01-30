from datetime import datetime
from typing import Optional

from rich.console import Console

from .backend.backend import Backend
from .logger import Logger


class LoggerProxy:
    """A proxy that forwards attribute access to the actual logger instance.

    This allows `from mvp_engine.utils.log import logger` to work correctly
    even when the logger is initialized after the import.
    """

    _instance: Optional[Logger] = None

    def __getattr__(self, name):
        if self._instance is None:
            raise RuntimeError("Logger not initialized. Call init_logger() first.")
        return getattr(self._instance, name)

    def __bool__(self):
        return self._instance is not None


logger = LoggerProxy()


def get_logger() -> Optional[Logger]:
    """Get the current logger instance, or None if not initialized."""
    return logger._instance


def init_logger(backends: list[Backend], interval: int = 20) -> Logger:
    """Initialize the global logger with the given backends.

    Args:
        backends: List of logging backends to use.
        interval: Logging interval in iterations.

    Returns:
        The initialized Logger instance.
    """
    if logger._instance is not None:
        logger._instance.destroy()
    logger._instance = Logger(backends, interval)
    return logger._instance


def simple_info(message: str) -> None:
    """Log an info message using the global logger."""
    if get_logger() is not None:
        logger.info(message)

    console = Console(color_system="auto")
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"[bold]{date_str}[/bold] | [cyan]INFO[/cyan] | {message}", soft_wrap=True)
