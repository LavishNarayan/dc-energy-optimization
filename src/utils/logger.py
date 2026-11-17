from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]


def get_logger(name: str, log_file: str = "logs/system.log", level: int = logging.INFO) -> logging.Logger:
    """Returns a structured console + rotating file logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    full_log_path = ROOT_DIR / log_file
    full_log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        full_log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
