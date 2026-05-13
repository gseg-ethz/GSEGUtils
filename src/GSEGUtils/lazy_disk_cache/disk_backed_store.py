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

"""``MutableMapping`` of named :class:`LazyDiskCache` entries sharing a cache dir.

Provides :class:`DiskBackedStore`, the multi-entry container that complements
:class:`DiskBackedNDArray` (single-entry) and supports pickling the whole store
via :meth:`__getstate__` / :meth:`__setstate__`.

Phase 2 (Plan 02-01) hardening: the on-disk format is a constrained
``<key>.npy`` + ``<key>.meta.json`` pair written via
``np.save(..., allow_pickle=False)`` and ``json.dump``. The legacy ``pickle``
codec is gone (SEC-01); writes are atomic via ``tmp + flush + fsync +
os.replace + (POSIX) dir-fsync`` (FRAG-04); subclass names in the JSON sidecar
are resolved through an explicit allow-list dict (no ``importlib``).
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterator,
    MutableMapping,
    Optional,
    Protocol,
    TypeGuard,
    Unpack,
    cast,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, validate_call

from .disk_backed_ndarray import DiskBackedNDArray
from .lazy_disk_cache import LazyDiskCache, LazyDiskCacheConfig, LazyDiskCacheKw

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase-2 codec constants (D-02 / D-03)
# ---------------------------------------------------------------------------

_SCHEMA_VERSION: int = 1
_LAZY_DISK_CACHE_CLASS_REGISTRY: dict[str, type[LazyDiskCache]] = {
    "DiskBackedNDArray": DiskBackedNDArray,
}


def _resolve_lazy_disk_cache_class(name: str) -> type[LazyDiskCache]:
    """Resolve a class name to a registered :class:`LazyDiskCache` subclass.

    Implements D-02's explicit allow-list: only names baked into
    :data:`_LAZY_DISK_CACHE_CLASS_REGISTRY` at source-edit time are accepted.
    There is no ``importlib`` fallback, so a hand-crafted ``.meta.json`` cannot
    coerce the loader into instantiating an arbitrary subclass.
    """
    try:
        return _LAZY_DISK_CACHE_CLASS_REGISTRY[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown lazy_disk_cache_class {name!r}; allowed: {sorted(_LAZY_DISK_CACHE_CLASS_REGISTRY)}"
        ) from e


# type Array = _NDArray[np.generic]

# @runtime_checkable
# class SupportsOffload(Protocol):
#     def offload(self) -> None: ...


# type Factory[T: LazyDiskCache.rst] = Callable[[_NDArray, Unpack[LazyDiskCacheKw]], T]
@runtime_checkable
class Factory[T: LazyDiskCache](Protocol):
    """Protocol for callables that construct a :class:`LazyDiskCache` subtype from raw data."""

    def __call__(self, data: NDArray, **kwargs: Unpack[LazyDiskCacheKw]) -> T:
        """Construct a new cache entry of type ``T`` wrapping ``data``."""
        ...


type Validator[T] = Callable[[object], TypeGuard[T]]


class DiskBackedStore[T: LazyDiskCache](MutableMapping[str, T]):
    """Mapping of string keys to :class:`LazyDiskCache` entries with shared offload directory.

    Parameters
    ----------
    config : LazyDiskCacheConfig, optional
        Shared cache configuration (cache dir, caching flag, offload policy,
        purge-on-gc policy). Defaults to ``LazyDiskCacheConfig()``.
    factory : Factory[T]
        Callable used to wrap raw arrays into the concrete cache subtype ``T``
        when :meth:`add_data_to_store` is called.
    value_type : type[T] or tuple of type[T], optional
        If set, every value inserted must be an instance of this type / one of
        these types.
    validator : Validator[T], optional
        Additional runtime check executed on every insert.

    Notes
    -----
    Threading: this class has no instance lock; per-entry writes get their
    atomicity from :class:`LazyDiskCache`'s own :class:`threading.RLock` plus
    the ``os.replace`` semantics of :meth:`offload`. Single-PCD multi-thread
    mutation is unsupported (see PROJECT.md threading constraint).
    """

    _DBNDArrayFileExt = ".npy"
    _DBNDArrayMetaExt = ".meta.json"
    _LegacyPickleExt = ".pkl"

    _store: dict[str, Optional[T]]
    _cache_dir: Path
    _enable_caching: bool
    _automatic_offloading: bool
    _purge_disk_on_gc: bool

    _factory: Factory[T]
    _value_type: Optional[type[T] | tuple[type[T], ...]]
    _validator: Optional[Validator[T]]

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        *,
        config: LazyDiskCacheConfig = LazyDiskCacheConfig(),  # noqa: B008  # LazyDiskCacheConfig is a frozen pydantic dataclass — safe as default.
        factory: Factory[T],
        value_type: Optional[type[T] | tuple[type[T], ...]] = None,
        validator: Optional[Validator[T]] = None,
    ) -> None:

        self._store = {}
        self._enable_caching = config.enable_caching
        if config.cache_path is None or config.cache_path.is_file():
            self._cache_dir = Path(tempfile.mkdtemp())
        else:
            self._cache_dir = config.cache_path
        self._automatic_offloading = config.automatic_offloading and config.cache_path is not None
        self._purge_disk_on_gc = config.purge_disk_on_gc

        self._factory = factory
        self._value_type = value_type
        self._validator = validator

        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            # Scan for existing files. We track any key that has a Phase-2 codec
            # pair (.npy + .meta.json); legacy .pkl files are intentionally NOT
            # registered here so __getitem__ surfaces them as a cache miss with
            # the D-05 INFO log via _load_entry.
            available_files = [f for f in self._cache_dir.glob(f"*{self._DBNDArrayFileExt}") if f.is_file()]
            for f in available_files:
                self._store[f.stem] = None

    def _check_T(self, value: object) -> T:
        if not isinstance(value, LazyDiskCache):
            raise TypeError(f"value must be LazyDiskCache; got {type(value)}")

        if self._value_type is not None and not isinstance(value, self._value_type):
            raise TypeError(f"value must be {self._value_type}; got {type(value)}")

        if self._validator is not None and not self._validator(value):
            raise TypeError(f"value rejected by validator; got {type(value)}")

        return cast(T, value)

    def __getitem__(self, key: str) -> T:
        """Return the entry for ``key``, loading it from disk on a cache miss.

        Raises
        ------
        KeyError
            If no in-memory entry and no on-disk codec pair exist for ``key``,
            or if a legacy ``.pkl`` is present (refused without invoking
            the legacy pickle reader).
        ValueError
            If the on-disk JSON sidecar has an unsupported ``schema_version``
            or an unknown ``lazy_disk_cache_class``.
        """
        obj = self._store.get(key, None)
        if obj is not None:
            return obj

        loaded_obj = self._load_entry(key)
        self._store[key] = loaded_obj
        return loaded_obj

    def __setitem__(self, key: str, value: T) -> None:
        """Validate ``value`` and store it under ``key`` in memory."""
        self._store[key] = self._check_T(value)

    def __delitem__(self, key: str) -> None:
        """Remove ``key`` from the in-memory store."""
        del self._store[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys currently tracked by the store."""
        return iter(self._store)

    def __contains__(self, key):
        """Return ``True`` if ``key`` is tracked (in memory or on disk)."""
        return self._store.__contains__(key)

    def __len__(self) -> int:
        """Return the number of tracked keys."""
        return len(self._store)

    def __repr__(self) -> str:
        """Return a debug representation listing the currently-tracked keys."""
        return f"<DiskBackedStore({list(self._store.keys())})>"

    def _get_npy_path(self, feature: str) -> Path:
        """Return the on-disk ``.npy`` path for ``feature``."""
        return self._cache_dir / f"{feature}{self._DBNDArrayFileExt}"

    def _get_meta_path(self, feature: str) -> Path:
        """Return the on-disk JSON sidecar path for ``feature``."""
        return self._cache_dir / f"{feature}{self._DBNDArrayMetaExt}"

    def _get_legacy_pickle_path(self, feature: str) -> Path:
        """Return the legacy pre-Phase-2 ``.pkl`` path for ``feature`` (refused on read)."""
        return self._cache_dir / f"{feature}{self._LegacyPickleExt}"

    def add_data_to_store(
        self,
        key: str,
        data: NDArray,
        *,
        enable_caching_override: Optional[bool] = None,
        automatic_offloading_override: Optional[bool] = None,
        purge_disk_on_gc_override: Optional[bool] = None,
    ) -> None:
        """Wrap ``data`` via the configured factory and insert it under ``key``.

        Parameters
        ----------
        key : str
            Key under which the new cache entry is registered.
        data : NDArray
            Raw array to be wrapped.
        enable_caching_override : bool, optional
            Per-entry override for the store-level caching flag.
        automatic_offloading_override : bool, optional
            Per-entry override for the store-level auto-offload flag.
        purge_disk_on_gc_override : bool, optional
            Per-entry override for the store-level purge-on-gc flag.

        Raises
        ------
        KeyError
            If ``key`` is already present in the store.
        """
        if key in self:
            raise KeyError(f"Key {key} already exists in store.")

        enable_caching = enable_caching_override if enable_caching_override is not None else self._enable_caching
        cache_path = self._get_npy_path(key) if self._cache_dir else None
        automatic_offloading = (
            automatic_offloading_override if automatic_offloading_override is not None else self._automatic_offloading
        )
        purge_disk_on_gc = (
            purge_disk_on_gc_override if purge_disk_on_gc_override is not None else self._purge_disk_on_gc
        )

        new_container = self._factory(
            data,
            enable_caching=enable_caching,
            cache_path=cache_path,
            automatic_offloading=automatic_offloading,
            purge_disk_on_gc=purge_disk_on_gc,
        )

        self._store[key] = self._check_T(new_container)

    @property
    def store(self) -> dict[str, Optional[T]]:
        """Return the internal mapping of keys to in-memory entries (``None`` if offloaded)."""
        return self._store

    @property
    def cache_dir(self) -> Path:
        """Return the directory where offloaded codec pairs are written."""
        return self._cache_dir

    def keys(self) -> list[str]:
        """Return a list of all tracked keys."""
        return list(self._store.keys())

    def values(self) -> Iterator[Optional[T]]:
        """Iterate over the current in-memory entries (``None`` where offloaded)."""
        return iter(self._store.values())

    def items(self) -> Iterator[tuple[str, Optional[T]]]:
        """Iterate over ``(key, value)`` pairs (``value`` is ``None`` where offloaded)."""
        return iter(self._store.items())

    def offload(self, keys: Optional[str | list[str]] = None, pickle_container: bool = False) -> None:
        """Offload selected entries to disk.

        When no keys are provided every cached entry is considered. Items with
        ``cache_enabled=False`` are skipped. When ``pickle_container`` is ``True``
        (the legacy parameter name, kept for backward compatibility) the entire
        container entry is offloaded via the Phase-2 codec (``.npy`` + JSON
        sidecar, no actual pickling), the in-memory reference is cleared, and
        the next access reloads it lazily via :meth:`_load_entry`.

        Parameters
        ----------
        keys : str or list[str], optional
            Specific keys to offload. Defaults to every tracked key.
        pickle_container : bool, optional
            When ``True`` write the wrapping container via the codec; when
            ``False`` (default) delegate to each entry's own :meth:`offload`
            method. The name is retained for API stability — no pickle is used.
        """
        if keys is None:
            keys = self.keys()
        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            obj = self._store[key]
            if obj is None:
                continue
            if not obj.cache_enabled:
                logger.debug("Skipping offload for %s because caching is disabled.", key)
                continue
            if pickle_container:
                self._store_entry(key, obj)
                self._store[key] = None
                logger.debug(
                    "Wrote codec pair for %s under %s and cleared in-memory reference.",
                    key,
                    self._get_npy_path(key),
                )
                del obj
            else:
                obj.offload()

    def _store_entry(self, key: str, entry: LazyDiskCache) -> None:
        """Atomically write a cache entry as ``.npy`` + ``.meta.json`` pair (D-04 + Pitfall 4).

        Write order: ``.npy.tmp`` → flush+fsync → ``.meta.json.tmp`` → flush+fsync
        → ``os.replace(.npy.tmp → .npy)`` → ``os.replace(.meta.json.tmp → .meta.json)``
        → POSIX dir-fsync. A torn write leaves only ``.tmp`` files which the reader
        treats as cache miss. Disk-full / permission errors are re-raised after
        best-effort ``.tmp`` cleanup.
        """
        npy_final = self._get_npy_path(key)
        json_final = self._get_meta_path(key)
        npy_tmp = npy_final.with_suffix(".npy.tmp")
        json_tmp = self._cache_dir / f"{key}.meta.json.tmp"
        try:
            # _describe_buffer returns (shape, dtype, in_memory_array). We
            # serialise the live buffer; np.save writes the ndarray with its
            # full shape + dtype header (allow_pickle=False rejects object
            # dtypes per Pitfall 3).
            _shape, _dtype, in_memory_array = entry._describe_buffer()  # type: ignore[attr-defined]
            arr = np.asarray(in_memory_array)
            with open(npy_tmp, "wb") as f:
                np.save(f, arr, allow_pickle=False)
                f.flush()
                os.fsync(f.fileno())
            meta = {
                "schema_version": _SCHEMA_VERSION,
                "lazy_disk_cache_class": type(entry).__name__,
                "shape": list(arr.shape),
                "dtype": np.dtype(arr.dtype).str,
                "purge_disk_on_gc": entry.purge_disk_on_gc,
                "automatic_offloading": entry.automatic_offloading,
                "enable_caching": entry.cache_enabled,
            }
            with open(json_tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(npy_tmp), str(npy_final))
            os.replace(str(json_tmp), str(json_final))
            # POSIX dir-fsync so the rename itself is crash-durable (Pitfall 4).
            # Windows: os.open on a directory with O_RDONLY raises; guard.
            if os.name == "posix":
                dir_fd = os.open(str(self._cache_dir), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        except Exception:
            for p in (npy_tmp, json_tmp):
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
            raise

    def _load_entry(self, key: str) -> T:
        """Load a cache entry from the ``<key>.npy + <key>.meta.json`` pair.

        Refuses legacy ``.pkl`` files with a single INFO log (D-05).
        Raises ``KeyError`` on cache miss, ``ValueError`` on schema-version mismatch
        or unknown ``lazy_disk_cache_class``.

        Per W-5: the reconstructed instance's ``cache_path`` field is populated
        to the ``<key>.npy`` file path so the Plan-02-04 finalizer's
        :meth:`LazyDiskCache.enable_purge` reaches the registration branch
        instead of silently no-op-ing on ``if not self._cache_path: return``.
        Note that :meth:`LazyDiskCache._init_from_config` re-suffixes the
        provided ``cache_path`` with ``_MEMMAP_SUFFIX`` (``.dat``) internally,
        so the live ``self._cache_path`` on the reconstructed instance is
        ``<key>.dat`` rather than ``<key>.npy``. The W-5 invariant (a
        non-``None`` ``cache_path`` so ``enable_purge`` can register) holds
        either way.
        """
        npy_path = self._get_npy_path(key)
        json_path = self._get_meta_path(key)
        legacy_pkl = self._get_legacy_pickle_path(key)
        if legacy_pkl.exists() and not (npy_path.exists() and json_path.exists()):
            logger.info(
                "Legacy pre-Phase-2 cache file at %s is not loadable under the new "
                "codec; treating as cache miss. Re-materialise via the upstream factory.",
                legacy_pkl,
            )
            raise KeyError(key)
        if not (npy_path.exists() and json_path.exists()):
            raise KeyError(key)

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict) or meta.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported lazy_disk_cache schema_version "
                f"{meta.get('schema_version') if isinstance(meta, dict) else None}; "
                f"expected {_SCHEMA_VERSION}"
            )
        cls = _resolve_lazy_disk_cache_class(meta["lazy_disk_cache_class"])
        arr = np.load(str(npy_path), allow_pickle=False)
        # Reconstruct the LazyDiskCache subclass. W-5: pass `cache_path=str(npy_path)`
        # so the loaded instance can register its finalizer via enable_purge() (which
        # short-circuits on `if not self._cache_path: return`). LazyDiskCache
        # internally re-suffixes this to `<key>.dat` for memmap usage; the W-5
        # invariant is that `_cache_path` is not None, which holds.
        reconstruct_kwargs: dict[str, Any] = {
            k: meta[k] for k in ("purge_disk_on_gc", "automatic_offloading", "enable_caching") if k in meta
        }
        reconstruct_kwargs["cache_path"] = str(npy_path)
        return cast(T, cls(arr, **reconstruct_kwargs))

    def __getstate__(self) -> dict[str, Any]:
        """Offload everything before pickling and return the resulting ``__dict__`` snapshot.

        The store itself is still pickled here (we serialise our own metadata
        like ``_cache_dir`` / ``_store`` keys); only the per-entry payloads have
        been moved to the codec-pair on disk by the time we get here.
        """
        if self._enable_caching:
            self.offload(pickle_container=True)
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state from a pickled store snapshot and reload any offloaded entries from disk.

        Per D-06 (__setstate__ symmetry), the per-entry load path routes through
        :meth:`_load_entry`, inheriting the legacy-refusal (D-05) and
        cache_path-propagation (W-5) behaviour automatically.
        """
        self.__dict__.update(state)
        if self._enable_caching:
            for key in list(self.keys()):
                if self._store[key] is not None:
                    continue
                try:
                    self._store[key] = self._load_entry(key)
                except KeyError:
                    logger.warning(
                        "File for key %s not found in cache directory %s.",
                        key,
                        self._cache_dir,
                    )
                    continue
