"""
Emind Logger — 统一日志模块
替代各处 print，支持文件/控制台输出、颜色、分级
"""
import os
import sys
import logging
from pathlib import Path
from typing import Optional


_LOG_CONFIGURED = False

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def setup_logger(
    name: str = "emind",
    level: str = "info",
    log_file: Optional[str] = None,
    console: bool = True,
    log_dir: str = "logs",
) -> logging.Logger:
    global _LOG_CONFIGURED

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS.get(level, logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    if console:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if log_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            Path(log_dir) / log_file, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOG_CONFIGURED = True
    return logger


def get_logger(name: str = "emind") -> logging.Logger:
    if not _LOG_CONFIGURED:
        # Auto-configure with defaults on first access
        level = os.environ.get("EMIND_LOG_LEVEL", "info")
        log_file = os.environ.get("EMIND_LOG_FILE")
        return setup_logger(name, level=level, log_file=log_file)
    return logging.getLogger(name)
