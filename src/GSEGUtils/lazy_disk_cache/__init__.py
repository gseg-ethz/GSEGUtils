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

"""Lazy disk-backed cache primitives.

Exposes :class:`LazyDiskCache` (per-array offload-to-disk wrapper),
:class:`LazyDiskCacheConfig` / :class:`LazyDiskCacheKw` (configuration helpers),
:class:`DiskBackedNDArray` (single-pickle-file ndarray proxy), and
:class:`DiskBackedStore` (collection of named :class:`LazyDiskCache` entries
sharing a cache directory).

Also exposes the **store key contract**: :func:`is_valid_store_key`, the
predicate deciding whether a string may become a filename inside a cache
directory, and the exception types it is enforced with —
:exc:`StoreKeyError` (a refused key) and :exc:`StoreContainmentError` (a
resolved path that would land outside the cache directory).
"""

__all__ = [
    "LazyDiskCache",
    "LazyDiskCacheKw",
    "LazyDiskCacheConfig",
    "DiskBackedNDArray",
    "DiskBackedStore",
    "register_lazy_disk_cache_class",
    "StoreKeyError",
    "StoreContainmentError",
    "is_valid_store_key",
]

from .disk_backed_ndarray import DiskBackedNDArray
from .disk_backed_store import DiskBackedStore, register_lazy_disk_cache_class
from .lazy_disk_cache import LazyDiskCache, LazyDiskCacheConfig, LazyDiskCacheKw
from .paths import StoreContainmentError, StoreKeyError, is_valid_store_key
