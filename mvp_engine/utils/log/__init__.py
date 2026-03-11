import os
from datetime import datetime
from typing import Optional

from rich.console import Console

from .backend.backend import Backend
from .logger import Logger, LogLevel, parse_log_level


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


def _warn_invalid_log_level(invalid_level: str) -> None:
    """Warn and fallback to info when ``LOG_LEVEL`` is invalid."""
    console = Console(color_system="auto")
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(
        f"[bold]{date_str}[/bold] | [yellow]WARN[/yellow] | Invalid LOG_LEVEL '{invalid_level}', fallback to 'info'.",
        soft_wrap=True,
    )


def _parse_env_log_level() -> LogLevel:
    """Parse LOG_LEVEL from env with safe fallback to ``info``."""
    raw_level = os.getenv("LOG_LEVEL", "info")
    try:
        return parse_log_level(raw_level)
    except ValueError:
        _warn_invalid_log_level(raw_level)
        return LogLevel.INFO


def init_logger(
    backends: list[Backend], interval: int = 20, level: Optional[str] = None
) -> Logger:
    """Initialize the global logger with the given backends.

    Args:
        backends: List of logging backends to use.
        interval: Logging interval in iterations.
        level: Optional log level override. Falls back to ``LOG_LEVEL`` env var, then ``info``.

    Returns:
        The initialized Logger instance.
    """
    if level is None:
        parsed_level = _parse_env_log_level()
    else:
        parsed_level = parse_log_level(level)

    if logger._instance is not None:
        logger._instance.destroy()

    logger._instance = Logger(backends, interval, parsed_level)
    return logger._instance


def simple_info(message: str, level: str = "info") -> None:
    """Log a message using the global logger and fallback console output.

    Args:
        message: Message content.
        level: One of ``debug``, ``info``, ``warn``/``warning``, or ``error``.
    """
    parsed_level = parse_log_level(level)
    logger_instance = get_logger()

    if logger_instance is not None:
        if parsed_level == LogLevel.DEBUG:
            logger.debug(message)
        elif parsed_level == LogLevel.INFO:
            logger.info(message)
        elif parsed_level == LogLevel.WARNING:
            logger.warning(message)
        else:
            logger.error(message)
        return

    configured_level = _parse_env_log_level()
    if parsed_level < configured_level:
        return

    level_name = "WARN" if parsed_level == LogLevel.WARNING else level.upper()
    level_color = {
        LogLevel.DEBUG: "dim",
        LogLevel.INFO: "cyan",
        LogLevel.WARNING: "yellow",
        LogLevel.ERROR: "red",
    }[parsed_level]

    console = Console(color_system="auto")
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(
        f"[bold]{date_str}[/bold] | [{level_color}]{level_name.upper()}[/{level_color}] | {message}",
        soft_wrap=True,
    )
