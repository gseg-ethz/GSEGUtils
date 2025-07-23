import logging.config
from pathlib import Path

def setup_logging(log_file_path: Path|str ="debug.log"):
    if isinstance(log_file_path, Path) and not log_file_path.parent.exists():
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "[%(levelname)s]: %(message)s"
            },
            "detailed": {
                "()": "colorlog.ColoredFormatter",  # This tells dictConfig to instantiate ColoredFormatter
                "format": "%(log_color)s[%(levelname)s|%(name)s|L%(lineno)d] %(asctime)s: %(message)s",
                # "format": "%(log_color)s%(levelname)s:%(name)s:%(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z"
            },
            "file": {
                "format": "[%(levelname)s|%(name)s|L%(lineno)d] %(asctime)s: %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z"
            }
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "detailed",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "file",
                "filename": log_file_path,
                "level": "DEBUG",
                "mode": "a",  # append mode
                "encoding": "utf-8"
            }
        },
        "loggers": {
            "root": {
                "level": "DEBUG",
                "handlers": [
                    "stdout",
                    "file"
                ]
            },
            'matplotlib': {
                'level': 'WARNING',
                'handlers': ['stdout'],
                'propagate': False,
            },
            'PIL': {
                'level': 'WARNING',
                'handlers': ['stdout'],
                'propagate': False,
            },
        }
    }

    logging.config.dictConfig(log_config)