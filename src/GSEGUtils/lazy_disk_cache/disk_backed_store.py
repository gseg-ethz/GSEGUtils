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
"""

import logging
import pickle
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

from numpy.typing import NDArray
from pydantic import ConfigDict, validate_call

# from .disk_backed_ndarray import DiskBackedNDArray
from .lazy_disk_cache import LazyDiskCache, LazyDiskCacheConfig, LazyDiskCacheKw

logger = logging.getLogger(__name__)

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
    """

    _DBNDArrayFileExt = ".pkl"

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

            # Scan for existing files
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
            If no in-memory entry and no on-disk pickle file exist for ``key``.
        """
        obj = self._store.get(key, None)
        if obj is not None:
            return obj

        try:
            with open(self._get_pickle_path(key), "rb") as f:
                loaded_obj = cast(T, pickle.load(f))
        except FileNotFoundError as e:
            raise KeyError(key) from e

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

    def _get_pickle_path(self, feature: str) -> Path:
        return self._cache_dir / f"{feature}{self._DBNDArrayFileExt}"

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
        cache_path = self._cache_dir / f"{key}{self._DBNDArrayFileExt}" if self._cache_dir else None
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
        """Return the directory where offloaded pickles are written."""
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
        the entire container is pickled, the in-memory reference is cleared, and
        the next access reloads it lazily.

        Parameters
        ----------
        keys : str or list[str], optional
            Specific keys to offload. Defaults to every tracked key.
        pickle_container : bool, optional
            When ``True`` pickle the wrapping container; when ``False`` (default)
            delegate to each entry's own :meth:`offload` method.
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
                with open(self._get_pickle_path(key), "wb") as f:
                    pickle.dump(obj, f)
                self._store[key] = None
                logger.debug(
                    "Pickled DiskBackedNDArray for %s to %s and cleared in-memory reference.",
                    key,
                    self._get_pickle_path(key),
                )
                del obj
            else:
                obj.offload()

    def __getstate__(self) -> dict[str, Any]:
        """Offload everything before pickling and return the resulting ``__dict__`` snapshot."""
        if self._enable_caching:
            self.offload(pickle_container=True)
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state from a pickle and reload any offloaded entries that still exist on disk."""
        self.__dict__.update(state)
        if self._enable_caching:
            for key in list(self.keys()):
                if self._store[key] is not None:
                    continue
                try:
                    with open(self._get_pickle_path(key), "rb") as f:
                        loaded = pickle.load(f)
                    self._store[key] = self._check_T(loaded)
                except FileNotFoundError:
                    logger.warning(f"File for key {key} not found in cache directory {self._cache_dir}.")
                    continue
