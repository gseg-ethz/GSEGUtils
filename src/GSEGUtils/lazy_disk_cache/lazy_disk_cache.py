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

"""Abstract :class:`LazyDiskCache` base + configuration TypedDict / dataclass.

Defines the offload-to-memmap / load-back-into-memory protocol that
:class:`DiskBackedNDArray` and any future cache subclasses implement.
"""

import logging
import os
import tempfile
import threading
import weakref
from abc import ABC, abstractmethod
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import (
    Final,
    Literal,
    Optional,
    Self,
    TypedDict,
    Unpack,
)

import numpy as np
from numpy.typing import DTypeLike, NDArray
from pydantic import ConfigDict, validate_call
from pydantic.dataclasses import dataclass

from . import paths

logger = logging.getLogger(__name__)

# Phase 4 PERF-04 D-04: optional psutil for RAM-aware chunk sizing; ImportError fallback.
try:
    import psutil  # noqa: F401
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

_MEMMAP_FALLBACK_CHUNK_BYTES = 64 * 1024**2  # 64 MB (D-04 default)

#: The :meth:`LazyDiskCache.load` modes that open the ``.dat`` for writing, and
#: therefore the modes Plan 14-20 verifies containment on (D-31, STORE-02).
#:
#: Named as a set rather than spelled inline at the call site so the mode scope
#: is one editable fact rather than a condition a reader has to reverse-engineer.
#: Membership was decided by measurement, not by reading ``numpy``'s docs -- the
#: four-row table is recorded at the ``load()`` call site and in
#: :func:`GSEGUtils.lazy_disk_cache.paths._assert_write_contained`. ``"r"`` and
#: ``"c"`` are absent because neither reaches the file: ``"r"`` opens read-only
#: and ``"c"`` is copy-on-write, whose writes never leave RAM.
_MUTATING_MEMMAP_MODES: Final[frozenset[str]] = frozenset({"r+", "w+"})


def _purge_cache_pair(cache_path: Path) -> None:
    """Unlink ``cache_path`` and its paired ``.meta.json`` sidecar (FRAG-03 / W-1).

    The Plan-02-01 codec writes each entry as a ``<key>.npy + <key>.meta.json``
    pair. The finalizer must unlink both files; otherwise the sidecar leaks
    across the GC of the owning :class:`LazyDiskCache` instance.

    Safe to call with a path whose suffix is not ``.npy`` — only the primary
    ``cache_path`` is unlinked in that case (backwards-compatible with the
    canonical ``_MEMMAP_SUFFIX`` (``.dat``) memmap path produced by
    :meth:`LazyDiskCache._init_from_config`).
    """
    cache_path.unlink(missing_ok=True)
    if cache_path.suffix == ".npy":
        cache_path.with_suffix(".meta.json").unlink(missing_ok=True)


class LazyDiskCacheKw(TypedDict, total=False):
    """Keyword-argument TypedDict for :class:`LazyDiskCache` constructors.

    Attributes
    ----------
    enable_caching : bool
        When ``True`` the live buffer is backed by a ``numpy.memmap``; when
        ``False`` it stays in plain RAM.
    cache_path : pathlib.Path, optional
        Destination path for the memmap file. When ``None`` a temporary file is
        created.
    purge_disk_on_gc : bool
        When ``True`` the memmap file is deleted via :func:`weakref.finalize`
        once the cache object is collected.
    automatic_offloading : bool
        When ``True`` the cache offloads after every load.
    """

    enable_caching: bool
    cache_path: Optional[Path]
    purge_disk_on_gc: bool
    automatic_offloading: bool


@dataclass(config=ConfigDict(arbitrary_types_allowed=True), frozen=True)
class LazyDiskCacheConfig:
    """Frozen pydantic dataclass mirroring :class:`LazyDiskCacheKw`.

    Useful as a single argument that can be threaded through factory chains and
    re-derived via :meth:`updated` / :meth:`extend_cache_path`.

    Attributes
    ----------
    enable_caching : bool
        See :class:`LazyDiskCacheKw`.
    cache_path : pathlib.Path, optional
        See :class:`LazyDiskCacheKw`.
    purge_disk_on_gc : bool
        See :class:`LazyDiskCacheKw`.
    automatic_offloading : bool
        See :class:`LazyDiskCacheKw`.
    """

    enable_caching: bool = False
    cache_path: Optional[Path] = None
    purge_disk_on_gc: bool = True
    automatic_offloading: bool = False

    def as_kwargs(self) -> LazyDiskCacheKw:
        """Return the configuration as a :class:`LazyDiskCacheKw` mapping."""
        lazy_disk_cache_kw = LazyDiskCacheKw(
            enable_caching=self.enable_caching,
            cache_path=self.cache_path,
            purge_disk_on_gc=self.purge_disk_on_gc,
            automatic_offloading=self.automatic_offloading,
        )
        return lazy_disk_cache_kw

    @classmethod
    def from_kwargs(cls, settings: LazyDiskCacheKw) -> Self:
        """Construct a :class:`LazyDiskCacheConfig` from a TypedDict-shaped mapping."""
        return cls(**settings)

    def updated(self, **overrides: LazyDiskCacheKw) -> Self:
        """Return a copy of this configuration with the given fields overridden."""
        return replace(self, **overrides)

    @validate_call()
    def extend_cache_path(self, new_folder: str) -> Self:
        """Return a copy with ``cache_path`` extended by ``new_folder``.

        Parameters
        ----------
        new_folder : str
            Sub-directory name to append to the existing ``cache_path``.

        Returns
        -------
        LazyDiskCacheConfig
            A new configuration. If ``self.cache_path`` is ``None`` the new
            configuration also carries ``cache_path=None`` (and an informational
            log entry is emitted).

        Raises
        ------
        GSEGUtils.lazy_disk_cache.StoreKeyError
            If ``new_folder`` is not a single path segment, under the same
            lexical rule that guards store keys (D-02). This route looks like it
            sits outside the containment story — it is a configuration helper,
            and no store exists yet when it runs — but it joins a
            caller-supplied string straight into the cache root, and the callers
            are real: pc2img calls it **directly**, passing a per-tile id and a
            computed settings hash, and iof3D calls it **directly** with a path
            extension derived from a filename stem and also reaches it
            **indirectly** through pc2img's plural ``extend_cache_paths``
            wrapper. STORE-02's guarantee that no path the store builds lands
            outside the cache directory is hollow if the root those paths hang
            off can itself be walked upward, so the rule applies here too.

            The individual call sites are enumerated in ``MIGRATION-v1.0.md``
            under BC-GSEG-006 delta (4), and **deliberately not repeated here**:
            they are line numbers in another repository, they drift, and a
            second copy of them is precisely how the previous version of this
            sentence came to name a count that neither repository matched
            (§ WR-06). One place, not two.

            The check runs **before** the join and **regardless** of whether
            ``self.cache_path`` is ``None``: a bad folder name is a caller
            mistake either way, and gating the guard on configuration state is
            the same shape as the dead ``if self._cache_dir`` conditional this
            phase removes from the store.

        Notes
        -----
        The ``@validate_call()`` decorator above is a **type** check and never
        was a value check — a hostile folder name is a perfectly well-typed
        :class:`str`, so the decorator would accept ``'../..'`` without
        complaint. It is left exactly as it was; the value rule is the separate
        shared validator called below.
        """
        paths.validate_store_key(new_folder, self.cache_path)
        new_path = self.cache_path / new_folder if self.cache_path else None
        if new_path is None:
            logger.info("Cache path is None; cannot extend.")
        else:
            # STORE-02 / EB-02. The lexical rule above refuses a traversing *name*
            # ('../x'), but it cannot see the filesystem, so it cannot refuse a
            # perfectly legal name that is already a symlink pointing out of the
            # cache directory. That case is not an ordinary containment miss: this
            # component BECOMES the cache root of the store built on the returned
            # config, so every later `_assert_contained` measures against the
            # relocated base and stays self-consistent and silent while artefacts
            # land outside the configured root.
            #
            # The *write*-flavoured guard is the correct one here even though no
            # write happens on this line, and that is the whole point: it is the
            # only one that resolves the FINAL component. `_assert_contained`
            # deliberately leaves it unresolved (D-17) so a symlinked adopted entry
            # still loads — correct for reading an entry, and unable to fire for a
            # directory that is about to become a containment base. Verified: this
            # refuses a planted directory symlink while still accepting a cache root
            # that is itself a symlink (STORE-03) and a plain subfolder.
            paths._assert_write_contained(self.cache_path, new_path)
        return replace(self, cache_path=new_path)


class LazyDiskCache(ABC):
    """Abstract base for cache objects that transparently offload to a NumPy memmap.

    Subclasses provide the live buffer via :meth:`_describe_buffer` /
    :meth:`_set_buffer` and the shape/dtype metadata via
    :meth:`_describe_shape_dtype`; the base class handles the offload/load
    state machine, the finalizer that cleans up the memmap on GC, and the
    pickle protocol.
    """

    _MEMMAP_SUFFIX = ".dat"

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        **settings: Unpack[LazyDiskCacheKw],
        # **overrides: Unpack[CacheDefaults],
    ) -> None:
        config = LazyDiskCacheConfig(**settings)
        self._init_from_config(config)

    def _init_from_config(self, config: LazyDiskCacheConfig) -> None:
        # defaults = get_defaults()
        # automatic_offloading = overrides.get("preset_automatic_offloading", defaults["preset_automatic_offloading"])
        self._enable_caching = config.enable_caching
        if config.cache_path is None:
            fd, cache_path = tempfile.mkstemp(
                suffix=self._MEMMAP_SUFFIX
            )  # Todo: Think about a case where the provided path is a dir
            os.close(fd)
            self._cache_path = Path(cache_path)
        else:
            self._cache_path = config.cache_path.with_suffix(self._MEMMAP_SUFFIX)
        # elif cache_path.is_dir():

        # self._cache_path = cache_path.with_suffix(self._MEMMAP_SUFFIX) if cache_path else None
        self._automatic_offloading = config.automatic_offloading  # and (cache_path is not None)
        self._purge_disk_on_gc = config.purge_disk_on_gc

        self._lock = threading.RLock()
        # track in-memory state
        self._in_memory = True
        self._mmap: Optional[np.memmap] = None
        if self._enable_caching:
            self._convert_to_memmap()
        else:
            self._convert_to_ndarray()

        if config.automatic_offloading:
            self.offload()
        if self._cache_path and self._purge_disk_on_gc:
            self._finalizer = weakref.finalize(self, _purge_cache_pair, self._cache_path)

    def _convert_to_memmap(self) -> None:
        """Allocate (or reopen) the persistent memmap and adopt it as the live buffer.

        Notes
        -----
        Phase 4 PERF-04 D-04 / D-06: small arrays use the fast in-RAM copy path
        (today's behaviour); arrays exceeding the per-host chunk budget
        (~10% of ``psutil.virtual_memory().available``, or
        ``_MEMMAP_FALLBACK_CHUNK_BYTES`` when psutil is unavailable) are
        streamed in row-chunks. The new code never materialises a full-RAM
        copy of the source array on the streaming path.
        """
        with self._lock:
            # D-31 / STORE-02: verify containment before ANY write on this path.
            #
            # WHAT THIS GUARANTEES, stated as what it is rather than as a
            # universal. The memmap path is derived from the path the store
            # built, by re-suffixing it, so the honest property is that **the
            # memmap artefact lands beside the artefact its path was derived
            # from** -- not "inside the cache directory", which is true when a
            # store constructed the entry and false when a caller constructed
            # one directly with an arbitrary `cache_path=`.
            #
            # SCOPE, and the limit of it. This call and the one in `load()`
            # cover the three mutable opens the Plan 14-20 enumeration found:
            # the allocate-or-reopen `np.memmap` under `if self._mmap is None`,
            # the reopen `np.memmap` under the `elif`, and `load()`'s open on
            # its mutating modes. Completeness rests on **re-running that
            # enumeration whenever this module changes** -- no test would notice
            # a fourth mutable open added later; it would simply be unguarded
            # and the suite would stay green. A structural pin (a test asserting
            # the line-set of `np.memmap(` opens here) was considered and
            # rejected by the maintainer as a brittle line-number pin over a
            # module this phase is not otherwise restructuring.
            #
            # PLACEMENT, twice over. Ahead of the parent `mkdir` below, because
            # creating a directory is itself a write. And ahead of the mode
            # selection, so ONE call covers both `np.memmap` opens rather than
            # being duplicated into two branches a later edit could fix one of
            # and miss the other -- which is the shape of the defect being
            # closed here.
            #
            # THE UNCONFIGURED BRANCH, named rather than skipped. When no cache
            # path is configured, `_init_from_config` creates this `.dat` with
            # `tempfile.mkstemp` and `self._cache_path.parent` is the **system
            # temp directory**. The check still runs and still has a well-defined
            # base there, but that base is not a cache directory: **that branch
            # has no cache-directory root at all**, so the check catches a
            # symlink swapped in after `mkstemp` created the file and nothing
            # else. That is the limit of what it is worth there -- not a
            # statement that the case is safe. The file that branch leaks is
            # deferred item D-14-01.
            #
            # BOUNDARY -- STORE-08 owns the other half. This adds containment and
            # **not** torn-write safety: the temp-file / fsync / atomic-replace
            # machinery `DiskBackedStore._store_entry` already applies to the
            # `.npy` + `.meta.json` codec pair does not apply to this artefact,
            # so an interrupted write can still leave a partial `.dat`. That
            # half, and the purge-family interaction, is STORE-08 (Phase 15),
            # paired with D-14-01. A reader who finds a containment check beside
            # an unguarded truncating write needs to know the gap is filed.
            paths._assert_write_contained(self._cache_path.parent, self._cache_path)

            shape, dtype, array = self._describe_buffer()

            # D-04 chunk budget: RAM-fraction if psutil is available, else fixed-bytes.
            item_size_per_row = array.itemsize * (int(np.prod(array.shape[1:])) if array.ndim > 1 else 1)
            chunk_bytes = (
                int(psutil.virtual_memory().available * 0.10) if psutil is not None else _MEMMAP_FALLBACK_CHUNK_BYTES
            )
            chunk_rows = max(1, chunk_bytes // max(1, item_size_per_row))

            # 1) allocate or reopen the mmap file (unchanged from pre-rewrite)
            if self._mmap is None:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                mode: Literal["w+", "r+"] = "w+" if not self._cache_path.exists() else "r+"
                self._mmap = np.memmap(self._cache_path, dtype=dtype, mode=mode, shape=shape)
            elif self._mmap.mode != "r+":
                self._mmap = np.memmap(self._cache_path, dtype=dtype, mode="r+", shape=shape)

            # D-06 fast path: small arrays use today's one-shot copy semantics.
            if array.nbytes < chunk_bytes:
                array_copy = np.array(array, dtype=dtype, copy=True)
                self._mmap[:] = array_copy
            else:
                # D-06 streaming path: chunked slice-write, no full-RAM copy.
                for start in range(0, shape[0], chunk_rows):
                    self._mmap[start : start + chunk_rows] = array[start : start + chunk_rows]

            # 2) hand the memmap off to your subclass as the live buffer
            self._set_buffer(self._mmap)
            self._in_memory = True
            logger.debug(f"Switched buffer to memmap @ {self._cache_path}")

    def _convert_to_ndarray(self) -> None:
        """Pull the memmap entirely into RAM as a plain ndarray."""
        with self._lock:
            # if there's no mmap on disk yet, nothing to do
            if self._mmap is None or not self._cache_path.exists():
                return

            # load (or reload) the mmap so we can read it
            shape, dtype = self._describe_shape_dtype()
            mem = np.memmap(self._cache_path, dtype=dtype, mode="r", shape=shape)

            # copy it into a true ndarray
            array = np.array(mem, dtype=dtype, copy=True)

            # hand it back to the subclass
            self._set_buffer(array)
            self._in_memory = True

            # close & drop the mmap handle
            try:
                # there's no explicit close(), but deleting the object frees it
                del self._mmap
            finally:
                self._mmap = None
                logger.debug(f"Converted memmap @ {self._cache_path} to ndarray")

    @staticmethod
    def ensure_loaded(func):
        """Decorate ``func`` so the cache is materialised before the call and offloaded afterwards.

        If ``self.automatic_offloading`` is ``True`` and the cache was offloaded
        on entry, it is re-offloaded once the wrapped function returns.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            was_offloaded = self.offloaded
            if was_offloaded:
                self.load()
            result = func(self, *args, **kwargs)
            if self.automatic_offloading and was_offloaded:
                self.offload()
            return result

        return wrapper

    @property
    def offloaded(self) -> bool:
        """Return ``True`` when the live buffer currently lives on disk."""
        return not self._in_memory

    @property
    def automatic_offloading(self) -> bool:
        """Return whether the cache automatically offloads after each load."""
        return self._automatic_offloading

    @automatic_offloading.setter
    def automatic_offloading(self, value: bool):
        """Toggle automatic offloading."""
        assert isinstance(value, bool)
        self._automatic_offloading = value

    @property
    def cache_enabled(self) -> bool:
        """Return ``True`` when this instance is backed by a memmap."""
        return self._enable_caching

    def enable_caching(self) -> None:
        """Turn on memmap-backed caching, converting the live buffer if needed."""
        if self._enable_caching:
            return
        self._enable_caching = True
        self._convert_to_memmap()
        logger.debug(f"Enabled caching for {self._cache_path}")

        if self._automatic_offloading:
            self.offload()

    def disable_caching(self) -> None:
        """Turn off memmap-backed caching, copying any memmap content back into RAM."""
        if not self._enable_caching:
            return
        self.load()
        self._enable_caching = False
        self._convert_to_ndarray()
        logger.debug(f"Disabled caching for {self._cache_path}")

    @property
    def purge_disk_on_gc(self) -> bool:
        """Return whether the memmap file is deleted when this object is collected."""
        return self._purge_disk_on_gc

    def disable_purge(self):
        """Disable automatic deletion of the cache file on garbage collection."""
        with self._lock:
            if hasattr(self, "_finalizer"):
                self._finalizer.detach()
            self._purge_disk_on_gc = False
            logger.debug(f"Disabled purge for {self._cache_path}")

    def enable_purge(self):
        """Enable automatic deletion of the cache file on garbage collection.

        Registers a fresh :func:`weakref.finalize` callback if one is not
        currently alive.
        """
        with self._lock:
            if not self._cache_path:
                return
            if hasattr(self, "_finalizer") and self._finalizer.alive:
                # already enabled
                self._purge_disk_on_gc = True
                return
            # register a new finalizer (FRAG-03 / W-1: unlinks both <key>.npy + sidecar
            # when the path uses the .npy convention; falls through to single-file
            # unlink for the canonical ``.dat`` memmap path).
            self._finalizer = weakref.finalize(self, _purge_cache_pair, self._cache_path)
            self._purge_disk_on_gc = True
            logger.debug(f"Enabled purge for {self._cache_path}")

    @property
    def cache_path(self) -> Optional[Path]:
        """Return the memmap file path associated with this cache."""
        return self._cache_path

    @cache_path.setter
    def cache_path(self, value: Path) -> None:
        """Refuse reassignment: an entry's cache path is fixed at construction (D-01).

        Reading :attr:`cache_path` is unchanged; only assignment is sealed.

        What this closes
        ----------------
        The setter used to be public *and* it ``mkdir``-ed the new parent, so it
        could repoint a live entry anywhere on the filesystem — including outside
        the cache directory — **after** that entry's store key had already been
        validated, creating the destination directory on the way. That is the
        reproduced escape (R6): STORE-02's guarantee that no path the store
        builds lands outside the cache directory would otherwise read "contained,
        provided nobody assigns ``entry.cache_path``", which is a filter rather
        than an invariant.

        Note that the previous ``assert isinstance(value, Path)`` was never a
        real check — ``python -O`` strips assertions — so this raise is not
        replacing an enforcement, it is the first one on this route.

        Why there is no deprecation cycle here
        --------------------------------------
        Deliberately asymmetric with the ``DiskBackedStore._get_*_path`` builder
        methods (D-15), which do get a full ``DeprecationWarning`` cycle. Same
        principle, opposite outcome, and the principle is **evidence of use**:
        this setter has *zero* measured assigners across GSEGUtils, pchandler,
        pc2img and iof3D — every reference in all four repos is a read, and this
        package's own tests only assert equality — while the builder methods have
        measured live callers in pc2img. A deprecation cycle here would buy no
        migration room from anyone and would leave the reproduced escape live for
        another release while a downstream consumer is blocked on it. Recorded
        because the pair reads as arbitrary otherwise.

        Parameters
        ----------
        value : pathlib.Path
            The attempted path. Used only to make the refusal message
            actionable; it is never assigned.

        Raises
        ------
        AttributeError
            Always. ``AttributeError`` rather than
            :exc:`~GSEGUtils.lazy_disk_cache.StoreKeyError` because this is not a
            bad *key*, it is a forbidden *operation* — ``AttributeError`` is what
            Python raises for an unsettable attribute and what a caller's
            ``hasattr`` / ``setattr`` reasoning already expects.
        """
        current = getattr(self, "_cache_path", None)
        entry = f"{type(self).__name__} entry at {str(current)!r}" if current is not None else type(self).__name__
        raise AttributeError(
            f"Cannot reassign cache_path on the {entry}: the attempted path was {value!r}; "
            "an entry's cache path is fixed at construction — create the entry through "
            "DiskBackedStore (which derives the path from the validated store key), or pass "
            "cache_path= to the constructor."
        )

    def offload(self) -> None:
        """Flush the current buffer to disk, drop the in-RAM array, and mark offloaded."""
        with self._lock:
            if not self._enable_caching:
                logger.info(f"Caching disabled ==> {self.__class__}.`offload()` ignored for {id(self)}.")
                return
            if self.offloaded:
                logger.info(f"{self.__class__}: {id(self)} already offloaded to {self.cache_path}.")
                return

            # make sure we have a memmap buffer
            if self._mmap is None:
                self._convert_to_memmap()

            # 1) copy any non-memmap array into the memmap
            shape, dtype, array = self._describe_buffer()
            if not isinstance(self._mmap, np.memmap):
                # should not really happen, but just in case:
                self._mmap[:] = np.array(array, dtype=dtype, copy=True)  # type: ignore
            else:
                # if subclass buffer is plain ndarray, copy it in
                if not isinstance(array, np.memmap):
                    self._mmap[:] = array

            # 2) flush to disk
            self._mmap.flush()  # type: ignore

            # 3) drop the Python buffer and memmap handle
            self._drop_buffer()
            del self._mmap
            self._mmap = None
            self._in_memory = False

            try:
                self.on_offload()
            except Exception:
                logger.exception("Error in on_offload hook")
            logger.debug(f"Flushed buffer to from {self._cache_path}.")

    def load(self, mode: Literal["r", "r+", "w+", "c"] = "r+") -> None:
        """Reload the buffer from disk into memory (i.e. make self._mmap your active buffer)."""
        with self._lock:
            if not self.offloaded or self._cache_path is None:
                return

            # D-31 / STORE-02 / RV-01: this is a SECOND mutable open of the same
            # path, and it was unguarded until Plan 14-20. `mode` defaults to
            # the mutable "r+" and the truncating "w+" is accepted from the
            # caller, so a reload of an already-offloaded entry reached
            # `np.memmap(self._cache_path, ..., mode=mode)` below with no
            # containment check at all.
            #
            # The condition names the modes rather than testing something
            # incidental, and the set it names was decided by MEASUREMENT --
            # a 256-byte sentinel outside the cache directory, a `.dat` symlink
            # planted at it, one fresh entry and one fresh sentinel per mode:
            #
            #   'r'   no exception, sentinel untouched, handle NOT writable
            #   'r+'  no exception, sentinel untouched BY the call -- but the
            #         handle it installs is mutable, so the write arrives one
            #         statement later, through the buffer the caller is handed
            #   'w+'  no exception, sentinel TRUNCATED 256 -> 64 bytes
            #   'c'   no exception, sentinel untouched (copy-on-write never
            #         reaches disk)
            #
            # So the two mutating modes are guarded and "r"/"c" are not. Leaving
            # them out is a decision with a reason, not an omission: forcing the
            # write-flavoured final-component resolution onto a read is exactly
            # what D-17 and STORE-03 refuse, because a final-component symlink
            # is the legitimate adopted entry. The accepted residual is that
            # `load(mode="r")` can still READ through a planted symlink -- that
            # is disclosure, not corruption, and out of scope for a
            # containment-on-write decision. See `paths._assert_write_contained`
            # for the table and for the STORE-08 boundary.
            if mode in _MUTATING_MEMMAP_MODES:
                paths._assert_write_contained(self._cache_path.parent, self._cache_path)

            # (re)open the mmap if needed
            shape, dtype = self._describe_shape_dtype()
            if (
                self._mmap is None
                or self._mmap.shape != shape
                or self._mmap.dtype != np.dtype(dtype)
                or self._mmap.mode != mode
            ):
                self._mmap = np.memmap(self._cache_path, dtype=dtype, mode=mode, shape=shape)

            # hand that mmap to your subclass as its “buffer”
            self._set_buffer(self._mmap)
            self._in_memory = True
            try:
                self.on_load()
            except Exception:
                logger.exception("Error in on_load hook")
            logger.debug(
                f"Loaded buffer from {self._cache_path}.\r\n"
                # f"Min value: {np.nanmin(self._mmap)}; Max value: {np.nanmax(self._mmap)}"
            )

    # def __reduce__(self):
    #     self.disable_purge()
    #
    #     if self.cache_enabled:
    #         self.offload()
    #
    #     init_kwargs = {
    #             "enable_caching": self.enable_caching,
    #             "cache_path": Optional[Path]
    #             "purge_disk_on_gc": bool
    #             "automatic_offloading": bool
    #     }

    def __getstate__(self):
        """Snapshot purge intent, detach the finalizer, offload, and return a pickle-safe ``__dict__``.

        :class:`weakref.finalize` objects are not safely pickle-able (RESEARCH
        Pitfall 5: unpickled finalizers are technically present but
        ``alive=False`` — the callback never fires). We therefore detach the
        finalizer via :meth:`disable_purge` (which also flips
        ``_purge_disk_on_gc`` to ``False``), then *override* the dumped flag
        with the user's original intent so :meth:`__setstate__` can re-register
        via :meth:`enable_purge` (D-19 single source of truth).
        """
        # FRAG-03 / D-18: snapshot user's original intent BEFORE disable_purge()
        # mutates self._purge_disk_on_gc to False.
        original_purge_intent = self._purge_disk_on_gc
        self.disable_purge()

        if self.cache_enabled:
            self.offload()
        state = self.__dict__.copy()
        state.pop("_lock", None)
        # Restore the user's original purge intent into the dumped state so
        # __setstate__ can route through enable_purge() (D-19).
        state["_purge_disk_on_gc"] = original_purge_intent
        # The pickled `_finalizer` (if present) is dead-on-arrival per Pitfall 5;
        # drop it from the state so __setstate__ doesn't carry over a corpse.
        state.pop("_finalizer", None)
        return state

    def __setstate__(self, state):
        """Restore state, re-create the lock, and re-register the finalizer (FRAG-03 / D-18 + D-19).

        After ``__dict__.update(state)`` the loaded ``_purge_disk_on_gc`` flag
        reflects the user's *original* intent (preserved by
        :meth:`__getstate__`'s snapshot). If ``True``, route re-registration
        through :meth:`enable_purge` so the canonical
        :func:`weakref.finalize` registration path is the single source of
        truth (D-19).
        """
        self.__dict__.update(state)
        self._lock = threading.RLock()
        if self._cache_path and self._purge_disk_on_gc:
            # Flip _purge_disk_on_gc to False first so enable_purge()'s
            # alive-check passes through to the registration branch — the
            # pickled state may not contain a live _finalizer (Pitfall 5).
            # enable_purge() will then re-register and flip the flag back.
            self._purge_disk_on_gc = False
            self.enable_purge()

    @abstractmethod
    def _describe_buffer(self) -> tuple[tuple[int, ...], DTypeLike, NDArray]:
        """Return ``(shape, dtype, in_memory_array)`` describing the current live buffer."""
        ...

    @abstractmethod
    def _drop_buffer(self) -> None:
        """Drop the in-memory array reference (e.g. set the subclass slot to ``None``)."""
        ...

    @abstractmethod
    def _describe_shape_dtype(self) -> tuple[tuple[int, ...], DTypeLike]:
        """Return ``(shape, dtype)`` without materialising the full array."""
        ...

    @abstractmethod
    def _set_buffer(self, buf: NDArray) -> None:
        """Adopt ``buf`` as the new live buffer (typically a memmap returned from disk)."""
        ...

    # Optional hooks subclasses can override
    def on_offload(self) -> None:  # noqa: B027  # Optional override hook — intentionally no-op default; not abstract.
        """Run after offloading to disk; subclasses may override to prune extra resources."""
        pass

    def on_load(self) -> None:  # noqa: B027  # Optional override hook — intentionally no-op default; not abstract.
        """Run after loading from disk; subclasses may override to reinitialise extra state."""
        pass
