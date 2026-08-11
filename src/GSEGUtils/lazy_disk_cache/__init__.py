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
resolved path that would land outside the cache directory) — together with the
three artefact path builders :func:`get_npy_path`, :func:`get_meta_path` and
:func:`get_legacy_pickle_path`, each of which validates the key lexically and
then verifies that the path it returns lands inside the cache directory.
Phase 15 adds the two refusals raised by :meth:`DiskBackedStore.purge`:
:exc:`StorePurgeRefusedError` (a purge issued from a process that did not
construct the store) and :exc:`StorePurgeIncompleteError` (one or more artefacts
survived the unlink), both published because the migration note tells downstream
callers to catch them by name.

Those eight names — the six key-contract names above plus the two purge refusal
types — are the **published key-and-purge contract surface**. That figure counts
the contract, not the module: ``__all__`` itself carries fourteen names, the
remaining six being the four classes, the ``LazyDiskCacheKw`` typed dict and the
:func:`register_lazy_disk_cache_class` registration helper, which are not part of
the contract this paragraph is about. The raising validator behind
:func:`is_valid_store_key`, and the four internal-construction builders the store
writes through — the two temporary codec names and the two memmap names —
stay module-level in ``GSEGUtils.lazy_disk_cache.paths`` and are deliberately
unpublished; see the *Store key contract* page for the reasoning.
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
    "StorePurgeRefusedError",
    "StorePurgeIncompleteError",
    "is_valid_store_key",
    "get_npy_path",
    "get_meta_path",
    "get_legacy_pickle_path",
]

from .disk_backed_ndarray import DiskBackedNDArray
from .disk_backed_store import (
    DiskBackedStore,
    StorePurgeIncompleteError,
    StorePurgeRefusedError,
    register_lazy_disk_cache_class,
)
from .lazy_disk_cache import LazyDiskCache, LazyDiskCacheConfig, LazyDiskCacheKw
from .paths import (
    StoreContainmentError,
    StoreKeyError,
    get_legacy_pickle_path,
    get_meta_path,
    get_npy_path,
    is_valid_store_key,
)
