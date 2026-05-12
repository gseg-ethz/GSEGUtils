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

"""Thread-safe singleton metaclass shared across GSEG packages."""

import threading
from typing import ClassVar


class SingletonMeta(type):
    """Thread-safe singleton metaclass.

    Classes that set ``metaclass=SingletonMeta`` are instantiated at most once per
    process; subsequent calls return the originally-constructed instance. The
    instance registry and the construction lock are class-level attributes so the
    same metaclass governs every singleton class in the process.

    Attributes
    ----------
    _instances : dict[type, object]
        Mapping from each singleton class to its sole live instance.
    _lock : threading.RLock
        Re-entrant lock guarding ``_instances`` reads/writes during construction.
    """

    _instances: ClassVar[dict[type, object]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()

    def __call__(cls, *args, **kwargs):
        """Return the singleton instance for ``cls``, constructing it on first call.

        Parameters
        ----------
        *args
            Positional arguments forwarded to ``cls.__init__`` on first construction.
        **kwargs
            Keyword arguments forwarded to ``cls.__init__`` on first construction.

        Returns
        -------
        object
            The single live instance of ``cls``.
        """
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]
