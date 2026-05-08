"""Centralized logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """Configure root logger with console (and optional file) handler.

    Args:
        level: Logging level for all handlers.
        log_file: Optional path to write logs to file.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)
