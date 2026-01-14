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

__all__ = [
    "__author__",
    "__email__",
    "__version__",
    "version",
    "__version_tuple__",
    "version_tuple",
    "base_arrays",
    "base_types",
    "config",
    "constants",
    "generate_init_stubs",
    "logging_setup",
    "singleton",
    "util",
    "validators",
]

__author__ = "Nicholas Meyer"
__email__ = "meyernic@ethz.ch"

from . import (
    base_arrays,
    base_types,
    config,
    constants,
    generate_init_stubs,
    logging_setup,
    singleton,
    util,
    validators,
)
from ._version import __version__, __version_tuple__, version, version_tuple
