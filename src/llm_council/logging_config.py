"""Package-only logging; embedding applications retain their root configuration."""

from __future__ import annotations

import logging
import threading

from rich.console import Console
from rich.logging import RichHandler

_configuration_lock = threading.Lock()


def configure_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("llm_council")
    with _configuration_lock:
        logger.setLevel(level)
        # Respect an embedding application's handlers and avoid duplicate output on repeat runs.
        if not logger.hasHandlers():
            handler = RichHandler(
                console=Console(stderr=True),
                show_time=False,
                show_level=True,
                show_path=False,
                markup=False,
                rich_tracebacks=True,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
    return logger
