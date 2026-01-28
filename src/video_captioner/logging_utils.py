"""Logging helpers for video_captioner."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def configure_logging(*, log_file: Path, level: str = "INFO") -> None:
    """Configure loguru to log to stderr + a file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=level, backtrace=False, diagnose=False)
    logger.add(str(log_file), level=level, backtrace=False, diagnose=False)

