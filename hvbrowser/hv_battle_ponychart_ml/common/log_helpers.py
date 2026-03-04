"""Logging helpers for structured report output."""

from __future__ import annotations

import logging
from typing import Any


def log_section(
    logger: logging.Logger,
    title: str,
    *args: Any,
    width: int = 90,
) -> None:
    """Log a section header: blank line, = separator, title, = separator.

    Usage::

        log_section(logger, "RECOMMENDATION", width=90)
        log_section(logger, "RESULTS (%d images)", n, width=80)
    """
    logger.info("")
    logger.info("=" * width)
    logger.info(title, *args)
    logger.info("=" * width)
