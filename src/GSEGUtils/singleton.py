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
        r"""Return the singleton instance for ``cls``, constructing it on first call.

        Notes
        -----
        The fast path (post-first-init) is lock-free: a single ``dict.get`` on
        ``cls._instances`` under CPython's GIL is atomic, so the common case
        returns the cached instance without acquiring ``cls._lock``. The slow
        path -- taken on the first call and on any race window before the
        first construction completes -- acquires ``cls._lock``, re-checks the
        registry (the double-check), and only then runs ``super().__call__``.

        Under free-threaded Python (PEP 703 / 3.13t) the GIL is removed and
        the fast-path read of ``cls._instances`` becomes a data race; the
        slow-path lock alone is no longer sufficient. PROJECT.md pins Python
        3.12 only, so this is forward-looking documentation rather than a
        runtime concern. Revisit this pattern if 3.13t is ever added as a
        supported runtime.

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
        instance = cls._instances.get(cls)
        if instance is not None:
            return instance
        with cls._lock:
            instance = cls._instances.get(cls)
            if instance is None:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return instance
