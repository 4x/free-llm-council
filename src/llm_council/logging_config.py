"""Package-only logging; embedding applications retain their root configuration."""

from __future__ import annotations

import logging
import threading

_configuration_lock = threading.Lock()


def configure_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("llm_council")
    with _configuration_lock:
        logger.setLevel(level)
        # Respect an embedding application's handlers and avoid duplicate output on repeat runs.
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
            logger.addHandler(handler)
    return logger
