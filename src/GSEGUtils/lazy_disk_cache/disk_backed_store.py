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
    overload,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, validate_call

from . import paths
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


def register_lazy_disk_cache_class(cls: type[LazyDiskCache]) -> type[LazyDiskCache]:
    """Register a :class:`LazyDiskCache` subclass into the reload allow-list.

    This is the PUBLIC extension point for :data:`_LAZY_DISK_CACHE_CLASS_REGISTRY`,
    the allow-list :func:`_resolve_lazy_disk_cache_class` consults when
    reloading offloaded entries. Downstream packages (e.g. a subclass such as
    ``DiskBackedImageData``) call this once at import time so that a store
    holding instances of ``cls`` can round-trip them through offload → reload.

    The change preserves the D-02 security posture: resolution stays an
    *explicit* allow-list with no ``importlib`` fallback. A hand-crafted
    ``.meta.json`` still cannot coerce the loader into instantiating a class
    that was never registered through this API — it only ADDS a supervised
    registration surface, it does not open dynamic import.

    Parameters
    ----------
    cls : type[LazyDiskCache]
        A concrete ``LazyDiskCache`` subclass. It is registered under its
        ``__name__``, which is the same key :meth:`DiskBackedStore._store_entry`
        writes into the sidecar (``type(entry).__name__``).

    Returns
    -------
    type[LazyDiskCache]
        ``cls`` unchanged, so the function may also be used as a class decorator.

    Raises
    ------
    TypeError
        If ``cls`` is not a subclass of :class:`LazyDiskCache`.
    ValueError
        If a *different* class is already registered under ``cls.__name__``
        (name collision). Re-registering the *same* class object is idempotent
        and does not raise.
    """
    if not (isinstance(cls, type) and issubclass(cls, LazyDiskCache)):
        raise TypeError(f"register_lazy_disk_cache_class expects a LazyDiskCache subclass; got {cls!r}")
    name = cls.__name__
    existing = _LAZY_DISK_CACHE_CLASS_REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Cannot register {cls!r} under {name!r}: a different class "
            f"({existing!r}) is already registered under that name."
        )
    _LAZY_DISK_CACHE_CLASS_REGISTRY[name] = cls
    return cls


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
                # D-09 — WARN AND SKIP. This route reconstructs keys from
                # filenames that are ALREADY on disk, so what it sees is
                # pre-existing data, not a caller mistake. Raising here would
                # turn "open an old cache directory" into a crash, which is
                # precisely the outcome the policy split exists to prevent.
                #
                # This is DELIBERATELY the opposite of __setstate__'s policy
                # (D-10, which raises), and the asymmetry is the substance of
                # the split rather than an inconsistency: a rescan that raised
                # would crash on legitimate legacy data, whereas an unpickle
                # that merely warned would hand back a store silently missing
                # entries — data loss disguised as success, inside a worker,
                # which is the hardest class of failure to attribute.
                #
                # ACCEPTED COST, chosen with eyes open and recorded in D-09:
                # the refused file stays on disk, untracked and unreachable,
                # leaking the same way a nested key leaks today. The migration
                # note's cache-directory scan snippet (which imports
                # `is_valid_store_key`) is what makes that leak visible and
                # actionable rather than silent. Do not "fix" this into a raise.
                if not paths.is_valid_store_key(f.stem):
                    logger.warning(
                        "Skipping cache file %s in cache directory %s: its stem %r is not a legal "
                        "store key, so it cannot be tracked and the entry is unreachable. The file "
                        "is left untouched; scan the directory with "
                        "GSEGUtils.lazy_disk_cache.is_valid_store_key to find every affected entry.",
                        f.name,
                        self._cache_dir,
                        f.stem,
                    )
                    continue
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

        The key is validated lexically **first**, before the in-memory lookup
        and therefore before :meth:`_load_entry` can build or probe any path
        (D-11). D-11 deliberately *extends* SC-1, which is written over the
        routes that write into ``_store``: without a read-side check,
        ``store['../victim']`` is caught only by the resolved containment deep
        in the load path, so the invariant still holds but the caller gets a
        differently-shaped error than the equivalent write would give, at the
        cost of a ``resolve`` rather than a string comparison. It also closes a
        read-side existence probe — :meth:`_load_entry`'s legacy-``.pkl``
        ``exists()`` check would otherwise run on an unvalidated path.

        Raises
        ------
        StoreKeyError
            If ``key`` is not a legal single-segment store key (STORE-01,
            D-11). ``StoreKeyError`` subclasses :class:`ValueError`, **not**
            :class:`KeyError` (D-12), so a caller wrapping a read in
            ``except KeyError`` does not catch it. See :meth:`get`, which is
            overridden precisely because of that.
        KeyError
            If no in-memory entry and no on-disk codec pair exist for ``key``,
            or if a legacy ``.pkl`` is present (refused without invoking
            the legacy pickle reader).
        ValueError
            If the on-disk JSON sidecar has an unsupported ``schema_version``
            or an unknown ``lazy_disk_cache_class``.
        """
        paths.validate_store_key(key, self._cache_dir)

        obj = self._store.get(key, None)
        if obj is not None:
            return obj

        loaded_obj = self._load_entry(key)
        self._store[key] = loaded_obj
        return loaded_obj

    @overload
    def get(self, key: str, /) -> Optional[T]: ...

    @overload
    def get[D](self, key: str, /, default: T | D) -> T | D: ...

    def get[D](self, key: str, /, default: Optional[T | D] = None) -> Optional[T | D]:
        """Return the entry for ``key``, or ``default`` if it is absent or illegal.

        **This override is not redundant with the inherited method — deleting it
        silently re-breaks the accessor.** :class:`DiskBackedStore` subclasses
        :class:`~typing.MutableMapping`, whose :meth:`~typing.Mapping.get` is
        ``try: return self[key] except KeyError: return default``. D-12 makes
        :exc:`StoreKeyError` a :class:`ValueError` — deliberately *not* a
        :class:`KeyError`, because :meth:`add_data_to_store` already raises
        ``KeyError`` for "key exists". So the moment D-11 makes
        :meth:`__getitem__` validate, the inherited ``get`` stops catching the
        refusal and ``store.get('../victim', None)`` **raises** where it
        previously returned ``None``. D-11 enumerated ``__contains__`` and
        ``__delitem__`` explicitly and never reached ``.get()``, which inherits
        the new behaviour without anyone deciding it should.

        The catch tuple ``(KeyError, StoreKeyError)`` is therefore deliberate.
        It preserves D-11's shape rather than weakening it: ``__contains__`` is
        an explicit dict-backed membership test, so ``'../victim' in store`` is
        ``False`` today and stays ``False``; with this override the two
        interrogative read routes agree (membership is ``False``, ``get``
        returns the default) while the subscript still raises. No illegal key
        reaches :meth:`_load_entry` by any route.

        The rejected alternative was widening :meth:`__getitem__` to raise
        :class:`KeyError` instead, so the inherited ``get`` would keep working.
        That contradicts D-12's hard constraint — it would collide with
        :meth:`add_data_to_store`'s existing "key exists" ``KeyError`` and make
        the read route's error shape differ from every write route's, which is
        the inconsistency D-11 exists to remove.

        Parameters
        ----------
        key : str
            The store key to look up.
        default : optional
            Returned when ``key`` is absent from the store *or* refused by the
            lexical rule. Defaults to ``None``, matching ``Mapping.get``.

        Returns
        -------
        T or default
            The entry, or ``default``.
        """
        try:
            return self[key]
        except (KeyError, paths.StoreKeyError):
            return default

    def __setitem__(self, key: str, value: T) -> None:
        """Validate ``key`` lexically and ``value`` structurally, then store in memory.

        The key check is the **first** statement, ahead of ``_check_T`` and the
        store write (STORE-01). SC-1 constrains *how* it is done: the check
        performs no ``stat``, no ``resolve`` and no other filesystem syscall, so
        this route stays pure in-memory and is safe to call inside a ``loky``
        worker. Nothing may be added here that touches the filesystem — the
        resolved-containment layer lands in the path builders instead, which is
        the reason the two layers are separated at all.

        Raises
        ------
        StoreKeyError
            If ``key`` is not a legal single-segment store key (STORE-01). The
            key is not tracked when this raises.
        TypeError
            If ``value`` fails the value-type / validator checks.
        """
        paths.validate_store_key(key, self._cache_dir)
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
        StoreKeyError
            If ``key`` is not a legal single-segment store key (STORE-01).
            Validated **first**, before the "key exists" check and before any
            path is built, and **unconditionally** — the ``if self._cache_dir``
            guard below is dead code (``Path`` defines no ``__bool__`` and
            ``__init__`` assigns a ``Path`` unconditionally), so it must never
            be allowed to gate the validation.
        KeyError
            If ``key`` is already present in the store.
        """
        paths.validate_store_key(key, self._cache_dir)

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

        Raises
        ------
        StoreKeyError
            If the pickled state carries a key the STORE-01 lexical rule
            refuses (D-10).

        Notes
        -----
        D-10 — RAISE. Unpickling is a trust boundary, and this module already
        treats it as one: :func:`_resolve_lazy_disk_cache_class` resolves
        sidecar class names through an explicit allow-list with no
        ``importlib`` fallback for exactly this reason. A *post-fix* pickle
        cannot legitimately carry an illegal key, because :meth:`__getstate__`
        snapshots a ``_store`` whose keys the parent process already validated
        — so an illegal key arriving here means a legacy or a tampered pickle.

        This is DELIBERATELY the opposite of the ``__init__`` rescan's policy
        (D-09, which warns and skips). Warning-and-skipping here would return a
        store missing entries with no error: data loss disguised as success,
        inside a worker. The rescan's input is pre-existing data on disk, so
        crashing on it would be wrong; this route's input is a serialized
        snapshot that should never have been illegal, so failing loudly is the
        only way the caller learns anything.

        The check runs **before** ``__dict__.update`` rather than beside the
        per-entry reload below, because ``__dict__.update`` is itself a write
        into ``_store`` — it installs every pickled key wholesale — and the
        reload loop is gated on ``_enable_caching``. Validating first is the
        only placement that covers all keys on every configuration and leaves
        no partially-updated object behind.
        """
        incoming_store: dict[str, Any] = state.get("_store", {})
        incoming_cache_dir: Optional[Path] = state.get("_cache_dir")
        for incoming_key in incoming_store:
            paths.validate_store_key(incoming_key, incoming_cache_dir)

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
