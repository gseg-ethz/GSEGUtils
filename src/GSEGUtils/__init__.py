__all__ = [
    "__author__", "__email__",
    "__version__", "version", "__version_tuple__", "version_tuple",
    "base_arrays", "base_types", "config", "constants",
    "generate_init_stubs", "logging_setup", "singleton",
    "util", "validators"
]

__author__ = "Nicholas Meyer"
__email__ = "meyernic@ethz.ch"

from ._version import __version__, version, __version_tuple__, version_tuple
from . import (
    base_arrays, base_types, config, constants,
    generate_init_stubs, logging_setup, singleton,
    util, validators
)
