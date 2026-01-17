from datetime import datetime

from rich.console import Console

from .backend.backend import Backend
from .logger import Logger

logger: Logger = None


def init_logger(backends: list[Backend], interval: int = 20):
    global logger
    if logger is not None:
        logger.destroy()
    logger = Logger(backends, interval)
    return logger

def simple_info(message: str) -> None:
    """Log an info message using the global logger."""
    if logger is not None:
        logger.info(message)

    console = Console(color_system="auto")
    date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    console.print(
        f"[bold]{date_str}[/bold] | [cyan]INFO[/cyan] | "
        f"{message}",
        soft_wrap=True
    )


