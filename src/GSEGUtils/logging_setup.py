# GSEGUtils – General utility functions and classes for GSEG research/projects
#
# Copyright (c) 2025–2026 ETH Zurich
# Department of Civil, Environmental and Geomatic Engineering (D-BAUG)
# Institute of Geodesy and Photogrammetry
# Geosensors and Engineering Geodesy
#
# Authors:
#   Nicholas Meyer
#   Jon Allemand
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import logging.config
import tempfile
from pathlib import Path
from typing import Literal, Optional

LOGGING_LEVELS = Literal[
    "CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG", "NOTSET"
]


def setup_logging(
    logfile_path: Optional[Path | str] = None,
    console_level: LOGGING_LEVELS = "WARNING",
    file_level: LOGGING_LEVELS = "DEBUG",
) -> Path:
    if logfile_path is None:
        logfile_path = Path(tempfile.mkdtemp()) / "debug.log"
    elif isinstance(logfile_path, str):
        logfile_path = Path(logfile_path)

    if not logfile_path.parent.exists():
        logfile_path.parent.mkdir(parents=True, exist_ok=True)

    logging_mapping = logging.getLevelNamesMapping()
    numeric_console: int = logging_mapping.get(console_level.upper(), 0)
    numeric_file: int = logging_mapping.get(file_level.upper(), 0)
    numeric_root = min(numeric_console, numeric_file)
    numeric_root = 30 if numeric_root == 0 else numeric_root

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {"format": "[%(levelname)s]: %(message)s"},
            "detailed": {
                "()": "colorlog.ColoredFormatter",  # This tells dictConfig to instantiate ColoredFormatter
                "format": "%(log_color)s[%(levelname)s|%(name)s|L%(lineno)d] %(asctime)s: %(message)s",
                # "format": "%(log_color)s%(levelname)s:%(name)s:%(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
            "file": {
                "format": "[%(levelname)s|%(name)s|L%(lineno)d] %(asctime)s: %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "detailed",
                "level": console_level,
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "file",
                "filename": logfile_path,
                "level": file_level,
                "mode": "a",  # append mode
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "root": {"level": numeric_root, "handlers": ["stdout", "file"]},
            "matplotlib": {
                "level": "WARNING",
                "handlers": ["stdout"],
                "propagate": False,
            },
            "PIL": {
                "level": "WARNING",
                "handlers": ["stdout"],
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(log_config)
    return logfile_path
