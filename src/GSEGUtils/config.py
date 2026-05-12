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

"""Per-context default settings for :class:`GSEGUtils.lazy_disk_cache.LazyDiskCache`.

Backed by :class:`contextvars.ContextVar` so the defaults can be overridden inside
a single async / threading context without leaking to siblings.
"""

from contextvars import ContextVar
from typing import Required, TypedDict, Unpack


class CacheDefaults(TypedDict, total=False):
    """TypedDict of overridable defaults for the disk-cache layer.

    Attributes
    ----------
    preset_automatic_offloading : bool
        Whether newly-constructed caches should default to automatic offload-on-GC.
    """

    preset_automatic_offloading: Required[bool]


_DEFAULTS: ContextVar[CacheDefaults] = ContextVar(
    "_DEFAULTS",
    # NOTE: a TypedDict is structurally a dict (mutable); B039 warns this can leak
    # across contexts.  The `configure` / `get_defaults` API explicitly copies-on-write
    # via `.get()` → `.set()`, so the shared default object is read-only in practice.
    # Hardening (frozen-dict default) is queued under FRAG (Phase 4).
    default=CacheDefaults(preset_automatic_offloading=True),  # noqa: B039
)


def configure(**defaults: Unpack[CacheDefaults]) -> None:
    """Update the cache defaults for the current context.

    Parameters
    ----------
    **defaults : Unpack[CacheDefaults]
        Keyword arguments matching :class:`CacheDefaults` keys; values overwrite
        the current per-context entry.
    """
    current = _DEFAULTS.get()
    current.update(defaults)
    _DEFAULTS.set(current)


def get_defaults() -> CacheDefaults:
    """Return the current effective :class:`CacheDefaults` for this context.

    Returns
    -------
    CacheDefaults
        Live reference (not a copy) to the currently-stored defaults.
    """
    return _DEFAULTS.get()
