"""Global logging proxy and lazy logging exports."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from types import ModuleType
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .backend.backend import Backend
    from .logger import Logger, LogLevel
    from .logger import parse_log_level as parse_log_level


class LoggerProxy:
    """A proxy that forwards attribute access to the actual logger instance.

    This allows `from mvp_engine.utils.log import logger` to work correctly
    even when the logger is initialized after the import.
    """

    _instance: Optional["Logger"] = None

    def __getattr__(self, name):
        """Forward attribute access to the initialized logger instance."""
        if self._instance is None:
            raise RuntimeError("Logger not initialized. Call init_logger() first.")
        return getattr(self._instance, name)

    def __bool__(self):
        """Return whether the global logger has been initialized."""
        return self._instance is not None


_LOGGER_PROXY = LoggerProxy()
logger = _LOGGER_PROXY


class _LogPackage(ModuleType):
    """Module wrapper that keeps the exported logger proxy stable."""

    def __getattribute__(self, name: str):
        """Return the stable logger proxy for module-level logger reads."""
        if name == "logger":
            return ModuleType.__getattribute__(self, "_LOGGER_PROXY")
        return ModuleType.__getattribute__(self, name)


sys.modules[__name__].__class__ = _LogPackage


def get_logger() -> Optional["Logger"]:
    """Get the current logger instance, or None if not initialized."""
    return _LOGGER_PROXY._instance


def _load_logger_symbols():
    """Import logger implementation symbols on demand."""
    from .logger import Logger, LogLevel, parse_log_level

    globals()["logger"] = _LOGGER_PROXY
    return Logger, LogLevel, parse_log_level


def _warn_invalid_log_level(invalid_level: str) -> None:
    """Warn and fallback to info when ``LOG_LEVEL`` is invalid."""
    from rich.console import Console

    console = Console(color_system="auto")
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(
        f"[bold]{date_str}[/bold] | [yellow]WARN[/yellow] | Invalid LOG_LEVEL '{invalid_level}', fallback to 'info'.",
        soft_wrap=True,
    )


def _parse_env_log_level() -> LogLevel:
    """Parse LOG_LEVEL from env with safe fallback to ``info``."""
    _, LogLevel, parse_log_level = _load_logger_symbols()

    raw_level = os.getenv("LOG_LEVEL", "info")
    try:
        return parse_log_level(raw_level)
    except ValueError:
        _warn_invalid_log_level(raw_level)
        return LogLevel.INFO


def init_logger(
    backends: list["Backend"],
    interval: int = 20,
    accumulation_size: int = 20,
) -> "Logger":
    """Initialize the global logger with the given backends.

    Args:
        backends: List of logging backends to use.
        interval: Logging interval in iterations.
        accumulation_size: Default metric smoothing window.

    Returns:
        The initialized Logger instance.
    """
    Logger, _, _ = _load_logger_symbols()

    parsed_level = _parse_env_log_level()

    if _LOGGER_PROXY._instance is not None:
        _LOGGER_PROXY._instance.destroy()

    _LOGGER_PROXY._instance = Logger(
        backends=backends,
        interval=interval,
        accumulation_size=accumulation_size,
        level=parsed_level,
    )
    globals()["logger"] = _LOGGER_PROXY
    return _LOGGER_PROXY._instance


def simple_info(message: str, level: str = "info") -> None:
    """Log a message using the global logger and fallback console output.

    Args:
        message: Message content.
        level: One of ``debug``, ``info``, ``warn``/``warning``, or ``error``.
    """
    from rich.console import Console

    _, LogLevel, parse_log_level = _load_logger_symbols()

    parsed_level = parse_log_level(level)
    logger_instance = get_logger()

    if logger_instance is not None:
        if parsed_level == LogLevel.DEBUG:
            _LOGGER_PROXY.debug(message)
        elif parsed_level == LogLevel.INFO:
            _LOGGER_PROXY.info(message)
        elif parsed_level == LogLevel.WARNING:
            _LOGGER_PROXY.warning(message)
        else:
            _LOGGER_PROXY.error(message)
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


def __getattr__(name: str):
    """Lazily resolve logging exports that would otherwise import backends eagerly."""
    if name in {"Logger", "LogLevel", "parse_log_level"}:
        Logger, LogLevel, parse_log_level = _load_logger_symbols()
        exports = {
            "Logger": Logger,
            "LogLevel": LogLevel,
            "parse_log_level": parse_log_level,
        }
        return exports[name]
    if name == "Backend":
        from .backend.backend import Backend

        return Backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
