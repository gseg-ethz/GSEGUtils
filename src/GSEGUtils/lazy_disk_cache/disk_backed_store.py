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
import weakref
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Final,
    Iterator,
    Mapping,
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
# Phase-15 purge refusals (D-07 + the D-10 aggregate)
# ---------------------------------------------------------------------------


class StorePurgeRefusedError(RuntimeError):
    """Raised when :meth:`DiskBackedStore.purge` is called from a foreign process.

    It deliberately does **not** join ``StoreKeyError``'s :class:`ValueError`
    hierarchy, and the difference is the whole point of a second type: nothing
    is wrong with the key. :class:`RuntimeError` says *you called this in the
    wrong place*, which is exactly what happened, and the broad
    ``except RuntimeError`` that worker code already writes still catches it
    (D-07). Re-parenting it under a ``StoreError`` root buys taxonomic tidiness
    at the cost of a published-type change, which Phase 14's D-12 rated one-way.
    """


class StorePurgeIncompleteError(OSError):
    """Raised when some of a key's artefacts survived :meth:`DiskBackedStore.purge`.

    The D-10 aggregate. POSIX gives no atomicity across N unlinks, so the
    contract is stated rather than implied: every artefact is attempted, the
    failures are collected, and one exception names the survivors.

    :class:`OSError` rather than a bare :class:`ExceptionGroup`, deliberately.
    A caller of a deleting operation already writes ``except OSError``; an
    ``ExceptionGroup`` is not caught by it, so the migrating downstream's
    existing handler would silently stop working at exactly the moment it was
    needed. It is published for the same reason its sibling is — a name a
    documented migration tells you to catch, but does not export, is reachable
    only if you insist.
    """


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


# ---------------------------------------------------------------------------
# Phase-14 pop sentinel (D-25)
# ---------------------------------------------------------------------------

#: Distinguishes "no default was supplied" from ``default=None`` in
#: :meth:`DiskBackedStore.pop`.
#:
#: ``None`` cannot serve as that marker, and this is a property of *this* store
#: rather than a stylistic preference: the entry mapping is typed
#: ``dict[str, Optional[T]]`` — ``None`` is what an *offloaded* entry looks like
#: — so a caller writing ``store.pop(key, None)`` is supplying a perfectly
#: legitimate default and must stay distinguishable from a caller writing
#: ``store.pop(key)``. Encoding "missing" as ``None`` would silently convert the
#: no-default form into the defaulting one and stop ``pop(key)`` raising, which
#: is the half of D-25 that is deliberately *not* changing.
#:
#: Annotated ``Any`` so it can be the default of a parameter annotated
#: ``T | D``. That is sound rather than a hole: the object is never returned and
#: never compared as a value — it is only ever tested with ``is``.
_POP_DEFAULT_MISSING: Final[Any] = object()


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

    # D-27 — THE ARTEFACT-SUFFIX ATTRIBUTES ARE GONE, AND THEIR ABSENCE IS THE
    # MECHANISM. This class used to publish three suffix aliases here and invite
    # subclasses to repoint them. Every codec suffix now comes from `paths`, so
    # a subclass cannot introduce a second vocabulary and there is nothing left
    # for a later round to re-diverge. The rationale in full — including what the
    # class gives up and what the deletion does *not* buy — is in the rescan
    # comment in `__init__`, beside the code the property protects.

    _store: dict[str, Optional[T]]
    _cache_dir: Path
    _enable_caching: bool
    _automatic_offloading: bool
    _purge_disk_on_gc: bool
    #: The pid of the process that ran ``__init__`` (D-05). Read only by
    #: :meth:`purge`; see the assignment for the four-route measurement.
    _owner_pid: int
    #: Weak references to the live entries each key has had (D-15-G1). Read
    #: only through :meth:`_registered_entries`; see the assignment in
    #: :meth:`__init__` for what it is for and what it must never become.
    _entry_registry: dict[str, list[weakref.ref[T]]]

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
        # D-05 — the worker guard's whole mechanism, captured once here and
        # compared per `purge()` call. Measured before it was chosen, on all
        # four routes a store can travel:
        #
        #   pickle.loads(pickle.dumps(store))  the attribute rides
        #                                      `__getstate__`'s `__dict__.copy()`
        #                                      and `__setstate__`'s
        #                                      `__dict__.update`, so a worker
        #                                      copy sees the PARENT's pid and
        #                                      differs from its own -> refuses
        #   os.fork()                          no pickle at all; the child
        #                                      inherits `__dict__` wholesale and
        #                                      its own pid differs -> refuses
        #   copy.copy(store) same process      same pid -> allowed, correctly
        #   a store CONSTRUCTED in a worker    owns its own files, gets its own
        #                                      pid here -> allowed, correctly
        #
        # No new pickle plumbing is needed for any of that, which is why the
        # attribute is a plain assignment and not a `__getstate__` special case.
        #
        # REJECTED, and recorded so it is not re-proposed: a `__setstate__`-
        # stamped "reconstructed copy" flag. Measured to false-positive on
        # `copy.copy` (refusing a legitimate same-process copy, since that route
        # also travels `__setstate__`) and to miss `fork` entirely, since no
        # pickle is involved there. Carrying both signals is the "symmetric
        # hooks that must stay in sync forever" shape Phase 14's D-03 removed by
        # re-deriving rather than storing.
        self._owner_pid = os.getpid()
        # D-15-G1 — THE WEAK ENTRY REGISTRY, in three sentences.
        #
        # WHAT IT IS FOR: it answers the one question `purge` could not
        # otherwise ask — *which live entries belong to this key when the
        # mapping no longer holds them?* — because `del`, `pop` and
        # `offload(pickle_container=True)` all leave `self._store.get(key)`
        # returning nothing while a live entry, and its armed
        # `weakref.finalize`, is still in the caller's hands.
        # WHAT IT MUST NEVER BECOME: a strong reference, a resurrection
        # mechanism, or state that outlives a pickle — it holds `weakref.ref`
        # objects only, `__getstate__` drops it (a `weakref.ref` is not
        # picklable) and `__setstate__` rebuilds it empty and re-registers what
        # the restored mapping can see.
        # WHICH ROUTES POPULATE IT: the four that install a LIVE entry into
        # `_store` — `__setitem__`, `add_data_to_store`, `__getitem__`'s
        # load-and-install, and `__setstate__`. The reopen rescan below installs
        # `None`, not an entry, so it registers nothing and that is a fact
        # rather than an omission.
        #
        # No drop route clears it. `__delitem__`, `pop`, `clear` and `offload`
        # deliberately leave it alone, and THAT IS THE MECHANISM: it is exactly
        # the entries those routes make unreachable through the mapping that
        # `purge` still has to disarm.
        self._entry_registry = {}
        self._automatic_offloading = config.automatic_offloading and config.cache_path is not None
        self._purge_disk_on_gc = config.purge_disk_on_gc

        self._factory = factory
        self._value_type = value_type
        self._validator = validator

        # Unconditional: the former `if self._cache_dir is not None` guard was
        # dead for exactly the reason recorded while its twin was deleted from
        # `add_data_to_store` — `__init__` assigns a `Path` on *both* branches a
        # few lines up, so no route can present a cache directory the
        # interpreter reads as empty. Leaving one instance of the pattern
        # standing weakened the argument the phase made when it removed the
        # other (WR-09), so both are gone and the claim now holds in both places
        # it was made.
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Scan for existing files. We track any key that has a Phase-2 codec
        # pair (.npy + .meta.json); legacy .pkl files are intentionally NOT
        # registered here so __getitem__ surfaces them as a cache miss with
        # the D-05 INFO log via _load_entry.
        available_files = [f for f in self._cache_dir.glob(f"*{paths.NPY_SUFFIX}") if f.is_file()]
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
            #
            # D-22 (Plan 14-10) extends that accepted cost to ONE MORE CLASS of
            # file rather than changing the policy. Two halves, and both are
            # needed:
            #
            #   1. The key is derived by stripping the artefact suffix off the
            #      file NAME, not with `Path.stem`. `Path('.npy').stem` is
            #      '.npy' — a leading dot reads as a name, not an extension —
            #      which is a legal-looking key the builders can never rebuild,
            #      so the store used to advertise through `keys()` an entry it
            #      could never load. The published scan snippet meanwhile
            #      reported '' for that same byte on disk. One file, two keys,
            #      neither of them the other: the doc/code drift the snippet's
            #      design exists to prevent, moved from the RULE to the
            #      DERIVATION.
            #
            #   2. The derived key must rebuild the file it came from, checked
            #      through the shared builder. Half one alone makes the property
            #      true by COINCIDENCE of the current suffix vocabulary; the
            #      rebuild is what makes it an invariant. That is the same
            #      filter-versus-invariant distinction this phase draws about
            #      paths, applied to keys.
            #
            #      D-27 (Plan 14-16, CR-01) FINISHES that half BY DELETION, and
            #      it supersedes D-26's answer to the same question. The
            #      superseded position is stated here rather than quietly
            #      dropped, because it was held on purpose and it reads as the
            #      obvious fix to anyone who arrives at this code cold.
            #
            #      WHAT D-26 DID, AND WHY IT WAS NOT ENOUGH. The glob, the
            #      derivation and (as of D-26) the rebuild all read the withdrawn
            #      artefact-suffix instance attribute — the three names are
            #      enumerated in `MIGRATION-v1.0.md` under BC-GSEG-006 and in the
            #      regression test's `WITHDRAWN_SUFFIX_ATTRS`, and deliberately
            #      NOT here, so that grepping this module for them returns
            #      nothing. NOTHING ELSE IN THE CLASS READ THEM:
            #      `_load_entry`, `_store_entry` and `offload` call
            #      `paths.get_npy_path` / `paths.get_meta_path`, which read the
            #      module constants. So the attribute governed DISCOVERY and
            #      nothing governed RETRIEVAL. For the base class the two are the
            #      same object, so nothing showed. For the subclass the change was
            #      written for, they disagreed for every file: measured, such a
            #      store adopted `['alpha']`, answered `'alpha' in s` with True,
            #      emitted NO warning, and raised `KeyError` on every read. Round 3
            #      traded "empty store plus a per-file warning" for "non-empty
            #      store, no warning, every advertised key unreadable" — which is
            #      the worse of the two by this module's own standard.
            #
            #      WHY DELETION RATHER THAN UNIFICATION. Unification — moving
            #      retrieval and writing onto the instance vocabulary — is the
            #      larger diff, and it leaves the divergence POSSIBLE: a fifth
            #      seam added later reads whichever vocabulary its author reaches
            #      for. Deleting the attribute makes the divergence
            #      UNREPRESENTABLE. That is the same filter-versus-invariant
            #      distinction this phase draws everywhere else, applied to the
            #      vocabulary itself.
            #
            #      WHAT THE CLASS GIVES UP. A subclass can no longer choose its
            #      artefact extension. That capability never worked: measured, a
            #      suffix-repointing subclass WROTE THE BASE SUFFIX
            #      (`beta.npy`, not `beta.sub.npy`), because the write path
            #      ignored the attribute — so it could not produce an artefact its
            #      own glob would find, and reopening its own directory adopted
            #      nothing. The documentation paragraph calling the repoint a
            #      supported configuration was therefore false in both directions,
            #      and it is gone with the attribute.
            #
            #      IN-02 GOES WITH IT. `key = f.name[: -len(suffix)]` degenerates
            #      to the empty key for EVERY file when the suffix is empty
            #      (`-len('')` is `0`, so the slice is `f.name[:0]`), and the empty
            #      suffix was reachable only through the published override point.
            #      The slice length is now decided at import time, so the
            #      degenerate case is unreachable and needs no separate guard.
            #
            #      WHAT THE DELETION DOES NOT BUY — READ THIS CLAUSE BEFORE
            #      REPEATING THE CLAIM ANYWHERE. This leaves `DiskBackedStore` with
            #      ONE **CODEC** ARTEFACT VOCABULARY: `.npy` / `.meta.json` /
            #      `.pkl`. It does NOT leave the class, and certainly not the
            #      package, with one artefact vocabulary full stop. The `.dat`
            #      memmap suffix keeps its own, duplicated between
            #      `paths.MEMMAP_SUFFIX` and `LazyDiskCache._MEMMAP_SUFFIX`, and
            #      unifying those is STORE-08 / Phase 15 work (filed beside
            #      D-14-01) that this round does not do. The unqualified claim
            #      would be false at the moment it was written, which is the same
            #      defect class as the documentation sentence deleted above.
            #
            # ORDERING, which is what keeps D-09's prohibition intact: check the
            # cheap predicate FIRST — it cannot raise, and it is what makes the
            # warning fire for the shapes D-09 already covered. Only then
            # rebuild, and compute the rebuild inside a handler, because the
            # shared builder RAISES for a refused key and can raise a
            # containment error on an odd directory. A round-trip check that
            # propagated would violate the very policy it is being added under.
            key = f.name[: -len(paths.NPY_SUFFIX)]
            skip_reason: Optional[str] = None
            if not paths.is_valid_store_key(key):
                skip_reason = "the key derived from its name is not a legal store key"
            else:
                try:
                    # The PUBLISHED builder, not `paths._build`. D-26 reached for
                    # the private seam only so it could pass a per-instance
                    # suffix; with the attribute gone there is no second suffix to
                    # pass, so the private seam buys nothing and this is the same
                    # builder `_load_entry` and `_store_entry` already call.
                    # Discovery and retrieval now cannot disagree.
                    rebuilt = paths.get_npy_path(self._cache_dir, key)
                except paths.StoreKeyError as exc:
                    skip_reason = f"the shared path builder refuses the key derived from its name ({exc})"
                else:
                    if rebuilt != f:
                        skip_reason = "the key derived from its name does not rebuild this file"

            if skip_reason is not None:
                # One call site, not two, with the reason interpolated: a
                # downstream grepping its logs has one message shape to match.
                #
                # D-13 (Plan 14-12, CR-01) — EVERY UNTRUSTED COMPONENT OF THAT
                # SHAPE IS QUOTED, and this is the message where the rule
                # actually bites. It is the ONLY one in this subsystem whose
                # input is genuinely untrusted: it renders names read off the
                # cache directory, written by whoever can write into it, and a
                # newline is a legal POSIX filename character. The derived key
                # beside it was already `%r`; the filename was not, which is
                # what makes the omission an oversight rather than a policy.
                #
                # The failure mode is not a garbled line but a WELL-FORMED
                # FORGED RECORD: the reproduction planted a filename that made
                # one `logger.warning` call emit three lines, the second reading
                # "WARNING:all clear, 0 files skipped" while a file was in fact
                # being skipped. That is worse here than anywhere else in the
                # module precisely because D-09 chose warn-and-skip — the log is
                # the ONLY signal this policy leaves the reader, and BC-GSEG-006
                # tells downstreams to grep it.
                #
                # `str(self._cache_dir)` rather than the `Path`, so `%r` renders
                # the path TEXT quoted rather than the `PosixPath(...)`
                # constructor repr — the shape `_refuse` already established with
                # `repr(str(cache_dir))`, so the two messages read alike.
                #
                # `skip_reason` is checked rather than assumed: two of its three
                # values are module-owned literals, and the third interpolates a
                # `StoreKeyError` built by `_refuse`, which already applies
                # `repr` to both the key and the cache directory — so it is safe
                # transitively rather than by construction. The regression test
                # asserts the WHOLE emitted record is single-line, which covers
                # that branch whatever produced it.
                logger.warning(
                    "Skipping cache file %r in cache directory %r: %s (derived key %r), so it "
                    "cannot be tracked and the entry is unreachable. The file is left untouched; "
                    "scan the directory with GSEGUtils.lazy_disk_cache.is_valid_store_key to find "
                    "every affected entry.",
                    f.name,
                    str(self._cache_dir),
                    skip_reason,
                    key,
                )
                continue
            self._store[key] = None

    def _check_T(self, value: object) -> T:
        if not isinstance(value, LazyDiskCache):
            raise TypeError(f"value must be LazyDiskCache; got {type(value)}")

        if self._value_type is not None and not isinstance(value, self._value_type):
            raise TypeError(f"value must be {self._value_type}; got {type(value)}")

        if self._validator is not None and not self._validator(value):
            raise TypeError(f"value rejected by validator; got {type(value)}")

        return cast(T, value)

    def _register_entry(self, key: str, entry: T) -> None:
        """Record a weak reference to ``entry`` under ``key`` (D-15-G1).

        Called from every route that installs a **live** entry into ``_store``,
        so that :meth:`purge` can reach the entry's finalizer after ``del``,
        ``pop`` or ``offload`` have taken it back out again.

        **This helper performs no filesystem syscall, and that is a constraint
        rather than a coincidence.** :meth:`__setitem__` documents of itself
        that its check "performs no ``stat``, no ``resolve`` and no other
        filesystem syscall, so this route stays pure in-memory and is safe to
        call inside a ``loky`` worker" (STORE-01 / SC-1, D-15-G2) — and it calls
        this helper. A later edit that added a path computation here would
        therefore break a documented guarantee of a *different* method, from a
        function whose own docstring said nothing about it. The reconciliation
        that genuinely needs to resolve paths lives in :meth:`purge`, where
        resolving is already permitted and where D-15-G2 puts it.

        **Presence is tested with** ``is``**, never** ``==``. Entries are
        array-like, and :func:`register_lazy_disk_cache_class` is a published
        extension point through which a downstream may register a subclass that
        defines ``__eq__``; an equality test here would then be a broadcast
        comparison and would raise on the truth value of an array. The same
        reasoning is why the structure is a list of weak references rather than
        a :class:`weakref.WeakSet`, which would additionally require its members
        to be hashable — a subclass defining ``__eq__`` has ``__hash__`` set to
        ``None``, so the failure would arrive at a consumer's insertion site
        with a :exc:`TypeError` naming neither this file nor the cause.
        """
        live = [ref for ref in self._entry_registry.get(key, ()) if ref() is not None]
        if not any(ref() is entry for ref in live):
            live.append(weakref.ref(entry))
        self._entry_registry[key] = live

    def _registered_entries(self, key: str) -> list[T]:
        """Return ``key``'s still-live registered entries, pruning dead references.

        The **only** read path into :attr:`_entry_registry`; nothing else may
        reach into it directly. Dead references are dropped on the way through
        and a key whose list empties is removed outright, so the mapping does
        not grow without bound on a long-lived store (STORE-05, axis 3).

        The references stay **weak** throughout: dereferencing one here binds
        the entry for the duration of the caller's loop and nothing longer, so
        an entry the caller has dropped is still collected and its finalizer
        still fires under the default ``purge_disk_on_gc=True``.
        """
        entries: list[T] = []
        live: list[weakref.ref[T]] = []
        for ref in self._entry_registry.get(key, ()):
            entry = ref()
            if entry is None:
                continue
            live.append(ref)
            entries.append(entry)
        if live:
            self._entry_registry[key] = live
        else:
            self._entry_registry.pop(key, None)
        return entries

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
        ``False`` today and stays ``False``; with this override the **three**
        interrogative read routes agree (membership is ``False``, ``get``
        returns the default, and :meth:`pop` returns the default when one was
        supplied) while the subscript still raises. No illegal key reaches
        :meth:`_load_entry` by any route.

        ↻ **CORRECTED by Plan 14-14 (D-25, § WR-01).** That sentence used to say
        *two*, and the miscount was not a typo — it was the defect. ``pop``
        reaches the subscript by exactly this route and stopped agreeing the
        moment D-11 and D-12 landed, one accessor over from the change this
        docstring argues for, with nobody deciding it. The surrounding argument
        is unchanged because it was correct; only its enumeration was short. See
        :meth:`pop`.

        The rejected alternative was widening :meth:`__getitem__` to raise
        :class:`KeyError` instead, so the inherited ``get`` would keep working.
        That contradicts D-12's hard constraint — it would collide with
        :meth:`add_data_to_store`'s existing "key exists" ``KeyError`` and make
        the read route's error shape differ from every write route's, which is
        the inconsistency D-11 exists to remove.

        **The containment carve-out (WR-03), which is why the handler is
        ordered rather than a single tuple.** D-12 makes
        :exc:`~GSEGUtils.lazy_disk_cache.StoreContainmentError` a *subclass* of
        :exc:`~GSEGUtils.lazy_disk_cache.StoreKeyError`, so the tuple above
        caught it too — and swallowing it is exactly what the published
        contract page tells consumers not to do:

            *"A per-item handler that skips one bad key should not silently
            swallow the second kind, so catch the base type only where you mean
            'this key was bad'."*

        This method **is** such a handler, and it is the library's own. A
        refused key is evidence about *the caller's key*; a containment
        violation is evidence about *the environment* — something planted a
        symlink in the cache directory — and degrading that into a ``None``
        turns an attack signal into an ordinary cache miss inside the caller's
        loop. So the subclass is caught and re-raised **before** the broader
        tuple. The order is the entire mechanism: a subclass caught after its
        base is never reached, and inverting these two clauses silently
        restores the swallow.

        *Reachability, stated honestly rather than implied.* For a plain
        :class:`str` key this carve-out cannot fire from here: once the lexical
        layer has accepted the key it carries no separator, so the built path's
        parent is lexically the cache directory and
        :func:`~GSEGUtils.lazy_disk_cache.paths._assert_contained` is
        unconditionally satisfied. It fires for a :class:`str` **subclass**
        whose ``__str__`` disagrees with its characters, and it would fire on
        any future lexical-layer regression — which is precisely when the
        signal is most wanted, and precisely when a swallowed one would be
        least recoverable.

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

        Raises
        ------
        StoreContainmentError
            If the path built for ``key`` would resolve outside the cache
            directory. Deliberately **not** converted into ``default`` (WR-03).
        """
        try:
            return self[key]
        except paths.StoreContainmentError:
            # Environment evidence, not a bad key — never swallowed by a read.
            # Must stay ABOVE the tuple below: it is a subclass of one of its
            # members, so a handler ordered the other way never reaches here.
            raise
        except (KeyError, paths.StoreKeyError):
            return default

    @overload
    def pop(self, key: str, /) -> T: ...

    @overload
    def pop[D](self, key: str, /, default: T | D) -> T | D: ...

    def pop[D](self, key: str, /, default: T | D = _POP_DEFAULT_MISSING) -> T | D:
        """Remove ``key`` and return its entry, or ``default`` if it is absent or illegal.

        **This override is not redundant with the inherited method, for the
        identical reason :meth:`get` is not — and it is here because the phase
        broke this accessor without noticing.**
        :class:`~typing.MutableMapping`'s ``pop`` is
        ``try: value = self[key] except KeyError: return default``, so the moment
        D-11 made :meth:`__getitem__` validate and D-12 made the refusal a
        :class:`ValueError` rather than a :class:`KeyError`, ``pop`` stopped
        catching it: ``store.pop('../victim', None)`` began **raising** where it
        previously returned ``None``. That is the same mechanism, one accessor
        over from the one it was diagnosed on. D-25 restores it, so every
        *defaulting or interrogative read* route agrees again — membership
        answers ``False``, :meth:`get` returns its default, and this returns its
        default.

        **Only the defaulting form is restored.** ``pop(key)`` with no default
        raises on a miss in every mapping, and for an illegal key it raises
        :exc:`~GSEGUtils.lazy_disk_cache.StoreKeyError`, consistent with the
        subscript (D-11/D-12) and with :meth:`~typing.MutableMapping.setdefault`.
        Widening the fix to the bare form would make ``pop`` the one mapping
        route that answers a miss with ``None``.

        **Why** :meth:`~typing.MutableMapping.setdefault` **is deliberately left
        raising**, stated here rather than left to be inferred from which method
        happens to be overridden. ``setdefault`` travels the same subscript and
        would move the same way — and it is a **write** route: it *inserts*, and
        refusing an illegal key at a write route is the whole point of this
        phase. A ``setdefault('../victim', entry)`` that returned its default
        would answer as though the key were fine and then be expected to have
        stored something under it. So the two accessors diverge **on purpose**;
        a reader who finds one overridden and the other not is looking at a
        decision, not an oversight. It is pinned by
        ``test_route_setdefault_still_raises_because_it_is_a_write_route``.

        **The containment carve-out, and why the handler is ordered.** Exactly as
        in :meth:`get`:
        :exc:`~GSEGUtils.lazy_disk_cache.StoreContainmentError` is a *subclass*
        of :exc:`~GSEGUtils.lazy_disk_cache.StoreKeyError`, so the broader catch
        below would swallow it too — the library's own per-item handler doing
        what the published contract page tells consumers not to do. A refused key
        is evidence about *the caller's key*; a containment violation is evidence
        about *the environment*, and degrading it into a default turns an attack
        signal into an ordinary cache miss inside the caller's loop. The subclass
        is therefore re-raised **before** the broader clause, and the order is
        the entire mechanism: a subclass caught after its base is never reached,
        so inverting these two clauses silently restores the swallow in a second
        place.

        **The delete is preserved, and it is what keeps this a** ``pop``. On
        success the key is removed through :meth:`__delitem__` before the value
        is returned. Losing that step would turn a write route into a read and
        leave this method a second :meth:`get` under another name; it is asserted
        directly rather than assumed.

        ↻ **CORRECTED by Plan 14-17 (D-29, § WR-02).** The paragraph above was
        true only of the *success* path, and the two things it left unsaid were
        both wrong in shipped code.

        **First, the delete now happens on the refusal path too, when there is
        something to delete.** The handler below could not tell *no such key*
        from *key exists, payload unreadable* — both arrive as ``KeyError`` —
        so ``pop`` answered a present-but-unloadable key with the default and
        **left it tracked**. ``dict.pop(k, d)`` removes ``k`` when ``k`` is
        present; this now does too, on **both** forms, so they differ only in
        what they return. A caller draining a store with ``pop(k, None)``
        previously completed a removal loop having removed nothing and having
        been told nothing.

        **Second, and it is not changed here: the delete is in-memory only.**
        It drops tracking and **leaves the on-disk artefacts** — the ``.npy``,
        the ``.meta.json`` and the ``.dat`` — exactly where they are. So a
        popped *offloaded* key is re-adopted by the very next subscript, since
        :meth:`__getitem__` falls back to :meth:`_load_entry` for any untracked
        key, and again by the reopen rescan. Between the ``pop`` and that
        subscript, ``key not in store`` and ``store[key]`` both succeed, which
        is a :class:`~typing.Mapping` contract violation and is a **known
        limitation rather than a designed guarantee**. It is not fixed here on
        purpose: **STORE-05** requires that *where the data lives*, *whether the
        key is tracked* and *whether the file outlives the object* stay three
        separate axes and be characterization-tested, and **STORE-04** owns the
        atomic drop-key-and-delete-files primitive, with its no-partial-effect
        and no-stale-finalizer requirements. Adding an unlink here would
        collapse the first and half-implement the second. Pinned by
        ``test_route_pop_of_an_offloaded_entry_leaves_its_artefacts_and_the_key_is_re_adopted``,
        which says in its own docstring that it characterizes a limitation.

        Parameters
        ----------
        key : str
            The store key to remove. **Positional-only**, matching
            ``dict.pop``; ``pop(key=..., default=...)`` raises
            :class:`TypeError`. Recorded in ``BC-GSEG-006`` delta (3) (§ IN-03)
            because it is a silent narrowing of ``MutableMapping``'s signature
            on the route that entry otherwise documents as a *restoration*.
        default : optional
            Returned when ``key`` is absent from the store, refused by the
            lexical rule, *or* present with a payload that cannot be loaded —
            and in that last case the key is **removed** first. The word
            *absent* used to stand alone here, and it is what made the no-op on
            a present key read as correct (§ WR-02a). When **no** default is
            supplied the refusal or the lookup error propagates instead — but
            the removal still happens — and the two cases are told apart by
            :data:`_POP_DEFAULT_MISSING`, because ``None`` is a legitimate
            caller-supplied default for an ``Optional``-valued store.

        Returns
        -------
        T or default
            The removed entry, or ``default``.

        Raises
        ------
        StoreKeyError
            If ``key`` is not a legal single-segment store key and no default
            was supplied (STORE-01, D-11/D-12).
        KeyError
            If ``key`` is legal but absent and no default was supplied.
        StoreContainmentError
            If the path built for ``key`` would resolve outside the cache
            directory. Deliberately **not** converted into ``default``, however
            the call was made (WR-03). **Nothing is removed when this
            propagates** — it is evidence about the environment, so the store is
            left intact for the caller to inspect.
        """
        try:
            value = self[key]
        except paths.StoreContainmentError:
            # Environment evidence, not a bad key — never swallowed by a read.
            # Must stay ABOVE the tuple below: it is a subclass of one of its
            # members, so a handler ordered the other way never reaches here.
            #
            # D-29 restructured the clause below and deliberately did NOT touch
            # this one, in either of the two ways it could have gone wrong:
            # it stays first, and it removes nothing. A containment violation
            # says something planted a symlink in the cache directory; dropping
            # the key on the way out would destroy the evidence.
            raise
        except (KeyError, paths.StoreKeyError):
            # D-29 / § WR-02a. This is the only place that can tell *no such
            # key* from *key exists, payload unreadable* — both arrive here as
            # a `KeyError`, and collapsing them is what made `pop(k, default)`
            # answer for a key it then left tracked. The membership test is
            # the distinction, and it must come BEFORE the `raise`, so that the
            # bare and defaulting forms differ only in what they return.
            if key in self:
                del self[key]
            if default is _POP_DEFAULT_MISSING:
                raise
            return default
        del self[key]
        return value

    def clear(self) -> None:
        r"""Remove every tracked key, whether or not its payload can be loaded.

        **Overridden because the inherited implementation returned successfully
        with the store still populated** (D-29, § WR-04).
        :class:`~typing.MutableMapping`'s ``clear`` is
        ``while True: self.popitem()`` under ``except KeyError: pass``, and
        ``popitem`` reaches the validating subscript — so the first tracked key
        whose payload cannot be loaded raises :class:`KeyError`, the handler
        reads that as *the mapping is empty*, and the loop stops. Measured over
        four tracked keys with one unloadable:
        ``['a', 'ghost', 'b', 'c']`` → ``['ghost', 'b', 'c']``, **raising
        nothing**. One key of four dropped, and the caller told it succeeded.

        **This removes tracking only** — the same in-memory scope :meth:`pop`
        and :meth:`__delitem__` have. No artefact is unlinked, so every cleared
        key that was offloaded is re-adopted by the reopen rescan. STORE-04 owns
        the combined drop-and-delete; see :meth:`__delitem__`.

        **Why the loop is over a snapshot of the keys and goes through**
        :meth:`__delitem__`, rather than one bulk ``self._store.clear()``. The
        inherited version's fault is not that it loops but that it *reads a
        payload* on the way; :meth:`__delitem__` reads nothing, builds no path
        and cannot raise for a key that is tracked, so looping through it has
        the defect designed out rather than guarded against. Going through the
        documented removal route also keeps the hook a subclass may already
        rely on: a subclass whose ``__delitem__`` unlinks would silently stop
        unlinking under a bulk clear, and would leak an artefact per key.

        The trailing check is the prohibition made structural rather than
        inferred: a route that cannot honour a key must raise, not drop part of
        the work and report success.

        **Why** :meth:`~typing.MutableMapping.popitem` **is deliberately left
        inherited**, stated here rather than left to be inferred from which
        method happens to be overridden — the same courtesy :meth:`pop`'s
        docstring extends to ``setdefault``. Its two ``KeyError``\\ s are
        **distinguishable**: on a non-empty store the payload error propagates
        as ``KeyError(key)`` (measured ``args == ('ghost',)``), while on an
        empty one ``next(iter(self))`` raises a bare ``KeyError`` (measured
        ``args == ()``). The information this method needed was therefore always
        present, and the round-3 defect was a handler that discarded it — **the
        fault was in the handler, not in the signal**, and with ``clear`` no
        longer routing through ``popitem`` nothing depends on it. Three
        override shapes were considered and each is worse than leaving it:

        * *Return a value anyway.* It would make ``popitem`` the one route that
          answers with a value for a key whose payload cannot be produced.
        * *Raise and remove, mirroring the bare* :meth:`pop`. For ``pop`` the
          caller named the key, so removing it is what they asked for; for
          ``popitem`` the store chose the key, and the caller cannot recover
          which one from the exception — so it would destroy tracking for a key
          they never identified.
        * *Raise a different type.* It changes nothing about the ambiguity and
          breaks the ``Mapping`` contract's documented ``KeyError`` on empty.

        Pinned by
        ``test_route_popitem_is_left_inherited_and_its_key_error_carries_the_key``,
        so a reader who finds ``clear`` overridden and ``popitem`` not can tell
        a decision from an oversight by grepping the suite.

        Raises
        ------
        RuntimeError
            If the store is still non-empty after every tracked key has been
            passed to :meth:`__delitem__` — reachable only through a subclass
            whose ``__delitem__`` does not remove. Raising is the point: the one
            thing this method must never do is return short.
        """
        for key in list(self._store):
            del self[key]
        if self._store:
            raise RuntimeError(
                f"clear() ran to completion with {sorted(self._store)!r} still tracked; a "
                "__delitem__ override that does not remove turns clear() back into the silently "
                "short route D-29 fixed (WR-04)"
            )

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
        """Remove ``key`` from the in-memory store, **leaving its files on disk**.

        This is the whole removal mechanism — :meth:`pop` and :meth:`clear` both
        route through it — and the second half of that first sentence is the
        half a reader actually needs. It drops tracking. It unlinks nothing: the
        ``.npy``, the ``.meta.json`` and the ``.dat`` stay exactly where they
        are.

        **The consequence, stated rather than left to be discovered.** For an
        *offloaded* entry the removal does not survive the next read:
        :meth:`__getitem__` falls back to :meth:`_load_entry` for any untracked
        key, so the very next ``store[key]`` re-adopts it, and so does the
        reopen rescan. Between the two, ``key not in store`` and ``store[key]``
        both succeed, which is a :class:`~typing.Mapping` contract violation.
        Treat it as a **known limitation, not a designed guarantee** — the
        earlier one-line docstring said only *"removes from the in-memory
        store"*, which is true and answers about half of what a caller reading
        it is trying to find out (D-29, § WR-02b).

        **The durable counterpart is** :meth:`purge`, **and the pair is the
        whole answer.** ``del store[key]`` drops tracking and unlinks nothing;
        :meth:`purge` drops tracking **and** removes every artefact whose name
        derives from the key, and it is the only removal that sticks — it
        sticks *because* it unlinks, since with nothing on disk there is
        nothing for the next read or the reopen rescan to re-adopt. Reach for
        this method when you mean *stop tracking this key in this process*, and
        for :meth:`purge` when you mean *this key and its data are gone*.

        **The re-adoption above is measured, not described**, and the
        measurement is a standing assertion rather than a transcript:
        ``tests/test_store_lifecycle_axes.py::test_delitem_is_undone_by_the_very_next_read``
        pins the read-back half and
        ``tests/test_store_lifecycle_axes.py::test_a_fresh_store_re_adopts_a_deleted_key``
        pins the fresh-store half. If either goes red, this docstring is what
        became wrong.

        **Do not "finish" this method** by adding an ``unlink`` here. **STORE-05**
        is the reason: it requires that *where the data lives* (offload),
        *whether the key is tracked* (this method) and *whether the file
        outlives the object* (``purge_disk_on_gc``) remain three separate axes
        and be characterization-tested. An unlink on this route would collapse
        the first two into one — and the operation you would be reaching for
        already exists, one method away, with the validate-before-mutate
        ordering and the process guard that make it safe.

        No key validation runs here, and that is D-11's decision rather than an
        omission: removal builds no path, so there is nothing to contain, and a
        key already tracked has been through a validating write route.
        """
        del self._store[key]

    def purge(self, key: str) -> None:
        """Drop ``key`` from tracking **and** unlink every artefact derived from it.

        The durable counterpart to :meth:`__delitem__`, and the half that method
        deliberately does not do. Where ``del store[key]`` is *untrack
        temporarily* — undone by the very next read, and by any fresh store over
        the same directory — this is the removal that sticks, and it sticks
        *because* it unlinks: with nothing on disk there is nothing to re-adopt.

        **What it removes, by rule rather than by list.** Every artefact whose
        name derives from ``key``: ``<key>.dat``, ``<key>.dat.tmp``,
        ``<key>.npy``, ``<key>.npy.tmp``, ``<key>.meta.json`` and
        ``<key>.meta.json.tmp`` (D-14). The ``.tmp`` names are in the set and
        not an oversight — each persists from creation until its rename, and a
        crash in between leaves one indefinitely, so a purge that skipped them
        would leak the very files the atomicity work creates. The legacy
        ``<key>.pkl`` is **not** removed (D-09); see the *Notes*.

        **The consequence, stated rather than left to be discovered.** An
        explicit purge wins over ``purge_disk_on_gc=False`` (D-03). That flag
        governs *implicit, GC-time* deletion; it is not a write-protect bit, and
        reading it as one would make this method unusable in precisely the
        configuration that accumulates the most artefacts. Every purge that
        exercises the override logs an INFO record naming the key, so the
        override is transparent rather than merely permitted.

        **What a future reader must not "fix".** Three things:

        1. The step below detaches the live entry's finalizer **directly**, and
           must not be routed through :meth:`LazyDiskCache.disable_purge`, which
           looks like the tidy call and is the wrong one — it also flips
           ``_purge_disk_on_gc``, which the ``__getstate__``/``__setstate__``
           loky dance snapshots and replays.
        2. The key is dropped **before** the first unlink and stays dropped even
           when an unlink fails. Re-tracking it on failure would point a live
           entry at a half-deleted artefact set and would make this method
           non-idempotent (D-10).
        3. Untracked-but-on-disk counts as present. Do not tighten this to
           "tracked only": that is exactly the state ``del store[key]`` leaves
           behind, and exactly the state a caller most needs to clean up, so the
           tighter reading would make the orphan case unpurgeable through the
           one verb built to purge it (D-02).

        Parameters
        ----------
        key : str
            The store key to purge. Validated lexically before anything else
            runs, and re-validated by every path builder.

        Raises
        ------
        StoreKeyError
            If ``key`` is not a legal single-segment store key (STORE-01).
            Raised before any path is built and before anything is touched.
        StoreContainmentError
            If a path built for ``key`` would resolve outside the cache
            directory (STORE-02). A subclass of ``StoreKeyError``.
        StorePurgeRefusedError
            If the calling process is not the one that constructed this store
            (D-05). Raised before any mutation, so a refused purge is a no-op.
        KeyError
            If ``key`` is present neither in tracking nor on disk (D-02).
        StorePurgeIncompleteError
            If one or more artefacts could not be unlinked. The key is still
            dropped, and the message names each survivor.

        Notes
        -----
        **The atomicity boundary, stated as what it is.** "Atomic" here — in
        STORE-04's text and in the ordering below — means *atomic with respect
        to store-owned ordering and refusal*: validation precedes every
        mutation, a refused purge touches nothing, and the key is dropped before
        the first unlink so the tracking state never describes a half-deleted
        artefact set. It does **not** mean globally atomic, and the two
        negatives are worth saying rather than leaving to be inferred.
        ``DiskBackedStore`` holds no store-level lock, so this method is
        **not safe against concurrent mutation of the same key** — a
        ``__setitem__`` or :meth:`add_data_to_store` racing this call may write
        an artefact between the existence check and the unlink, or have its
        freshly-written artefact unlinked underneath it. It is likewise
        **not globally atomic** across threads, or across processes sharing a
        cache directory, where POSIX offers nothing that would make it so.

        Do not read the one lock that *is* taken as more than it is: the live
        entry's ``RLock`` is held for the finalizer detach alone, never across
        the unlinks. Holding it across N unlinks would be a guarantee that
        exists only when the key happens to be tracked — the D-02 orphan case
        has no entry and therefore no lock — and a guarantee that sometimes
        exists is a filter, not an invariant. Single-threaded per-store use is
        the supported model, matching the project's threading constraint that a
        ``PointCloudData`` is not multi-thread-mutable either.

        **The process guard is on this method and nothing else** (D-08), and the
        asymmetry is deliberate rather than an omission. Workers legitimately
        *write*: ``__getstate__`` force-calls ``offload(pickle_container=True)``
        before pickling, so a guard on the write routes would break the joblib
        path outright. Deletion is the only operation where "wrong process" means
        "destroying someone else's data", so it is the only one guarded. Do not
        generalise the check to ``__setitem__`` or :meth:`add_data_to_store`.

        **The legacy pickle is left alone** (D-09), with a consequence recorded
        as deferred rather than hidden: a ``<key>.pkl`` is unreadable by design
        and now unremovable by the only removal verb, so a pre-0.5 cache
        directory keeps it and a listing after a "complete" purge still shows
        the key's name.
        """
        # 1. LEXICAL REFUSAL FIRST. Nothing below may run for a key the STORE-01
        #    rule rejects. SC-2 requires a refused purge to be a bit-for-bit
        #    no-op, and the only way to guarantee that is to refuse before the
        #    first filesystem call rather than after it — the reverse order is
        #    what once made a missing-key error irreversibly destructive
        #    downstream.
        paths.validate_store_key(key, self._cache_dir)

        # 2. EVERY PATH THROUGH THE BUILDERS (D-14) — never string concatenation,
        #    never `with_suffix` on another builder's result, never a glob. Each
        #    builder re-validates the key and verifies containment, so these six
        #    names are the only six paths this method can ever unlink, and an
        #    escaping key has already raised by the time the tuple exists.
        #    `get_legacy_pickle_path` is deliberately NOT built (D-09).
        sidecars_and_tmp = (
            paths.get_meta_path(self._cache_dir, key),
            paths.get_meta_tmp_path(self._cache_dir, key),
            paths.get_npy_tmp_path(self._cache_dir, key),
            paths.get_memmap_tmp_path(self._cache_dir, key),
        )
        payload = (
            paths.get_npy_path(self._cache_dir, key),
            paths.get_memmap_path(self._cache_dir, key),
        )
        artefacts = (*sidecars_and_tmp, *payload)

        # 3. THE WORKER GUARD (D-05/D-06/D-08). Above the existence check and
        #    above every mutation, because a purge issued from a forked or
        #    unpickled copy would be deleting the PARENT process's data — so the
        #    refusal has to happen before anything is touched, not after. It
        #    raises rather than warn-and-no-op (D-06): in a tile worker this
        #    fails the tile, which is the correct outcome for a call that would
        #    otherwise have destroyed data belonging to another process.
        #    THE ORDERING IS THE POINT: the comparison lives here, above the
        #    existence check and above every mutation. Moving it below the unlink
        #    loop still raises and still looks like a guard, and is what the
        #    ordering mutation proof breaks on purpose — measured, with the
        #    forked child exiting `_CHILD_PURGED` rather than `_CHILD_REFUSED`.
        current_pid = os.getpid()
        if current_pid != self._owner_pid:
            raise StorePurgeRefusedError(
                f"Refusing to purge {key!r}: this store was constructed by process "
                f"{self._owner_pid} and purge was called from process {current_pid}. "
                "A store's cache files belong to the process that constructed it, so a purge "
                "issued here would delete another process's data — which is why deletion is "
                "the one route guarded on process identity while the write routes are not "
                "(D-08: workers legitimately write, and pickling a store force-offloads it, "
                "so guarding writes would break the joblib path outright). Construct a store "
                "in this process for files this process owns, or return the key to the "
                "constructing process for cleanup."
            )

        # 4. EXISTENCE (D-02). The key counts as present when it is tracked OR
        #    when any of the six derived artefacts is on disk. Untracked-but-on-
        #    disk is exactly what `del store[key]` leaves behind, so the looser
        #    reading is the one that makes the orphan case reachable. No `bool`
        #    return: a silent no-op on a typo'd key would also make the removal
        #    verbs disagree with `__delitem__` and `pop` about missing keys.
        tracked = key in self._store
        if not tracked and not any(p.exists() for p in artefacts):
            raise KeyError(key)

        # 5. ONLY NOW MUTATE — and the finalizer goes first. Without the detach,
        #    a stale `weakref.finalize` survives this call and deletes whatever
        #    occupies its recorded path when the entry is eventually collected,
        #    eating a LATER entry created under the same key (the ABA hazard).
        #    `disable_purge()` is the wrong call here even though it performs the
        #    same detach: it also flips `_purge_disk_on_gc`, which the
        #    `__getstate__`/`__setstate__` loky dance snapshots and replays, so
        #    routing through it would silently rewrite an entry's durability
        #    intent as a side effect of deleting a different key's files. The
        #    entry lock is taken for the detach alone, because `enable_purge` and
        #    `disable_purge` both take it and the detach must not race them.
        #
        #    D-15-G1 — WHY THE ENTRIES COME FROM THE REGISTRY AND NOT FROM THE
        #    MAPPING, recorded because the line this replaces looked correct.
        #    The detach used to be gated on `self._store.get(key)` returning a
        #    live object, and that expression is the wrong question: it is
        #    `None` for an OFFLOADED entry — the mapping is typed
        #    `dict[str, Optional[T]]` — and absent for the untracked orphan
        #    `del` and `pop` leave behind, which is precisely the state D-02
        #    widened the EXISTENCE check above to reach. The existence check was
        #    widened and the detach was not, so the two disagreed about what
        #    "this key" means. Measured at 0956838, before this change: the
        #    finalizer stayed ARMED on THREE of the four routes out of the
        #    mapping (`offload`, `del`, `pop`), and a forced collection then ate
        #    a LATER entry's `<key>.dat` — a deletion attributable to nothing in
        #    the caller's code, arriving at an arbitrary GC (G-1 / SC-3). The
        #    weak registry is what makes the two agree: no drop route clears it,
        #    so this loop reaches every live entry the key has ever had, tracked
        #    or not.
        registered = self._registered_entries(key)
        for entry in registered:
            if hasattr(entry, "_finalizer"):
                with entry._lock:
                    entry._finalizer.detach()

        # 6. Drop the key. Before the unlinks, so the tracking state never
        #    describes a half-deleted artefact set.
        if tracked:
            del self._store[key]
        # The detached finalizers have nothing left to say, so the key's
        # registry list goes with the key. A later entry inserted under the same
        # key registers fresh, which is what keeps the detach scoped to the
        # purged lifetime rather than applied to the key's registration
        # generally.
        self._entry_registry.pop(key, None)

        # D-03 transparency. The flag is never read as a permission bit; it is
        # read here only to record that the override happened. The key is passed
        # as a lazy `%s` argument and never f-string-interpolated (CWE-117, the
        # same fix Phase 14's CR-01 applied to the rescan WARNING).
        if not self._purge_disk_on_gc:
            logger.info(
                "Explicit purge of key %s overrides purge_disk_on_gc=False: that flag governs "
                "implicit GC-time deletion, not explicit removal, so this key's artefacts are "
                "being unlinked (D-03).",
                key,
            )

        # 7. UNLINK — sidecars and `.tmp` names first, payloads last. The order
        #    is the D-10 contract: POSIX gives no atomicity across N unlinks, so
        #    a partial failure must leave a state the reader ALREADY treats as a
        #    cache miss (`_load_entry` requires both halves of the codec pair)
        #    rather than a pair whose sidecar disagrees with its array. Failures
        #    are collected rather than aborted on, so one unreadable artefact
        #    does not strand the other five.
        failures: list[OSError] = []
        survivors: list[Path] = []
        for artefact in artefacts:
            try:
                artefact.unlink(missing_ok=True)
            except OSError as exc:
                failures.append(exc)
                survivors.append(artefact)
        if failures:
            surviving = ", ".join(repr(str(p)) for p in survivors)
            raise StorePurgeIncompleteError(
                f"Purge of {key!r} was incomplete: {len(survivors)} of {len(artefacts)} artefacts "
                f"could not be unlinked and survive on disk — {surviving}. The key has been "
                "dropped from tracking and stays dropped: re-tracking it would point a live "
                "entry at a half-deleted artefact set and would make purge non-idempotent "
                "(D-10). Remove the listed files, or fix the permissions on the cache "
                "directory, and call purge again — it is safe to repeat."
            ) from failures[0]

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
            path is built, and **unconditionally**. The former
            ``if self._cache_dir`` guard on the cache-path construction below
            was dead code (``Path`` defines no ``__bool__`` and ``__init__``
            assigns a ``Path`` unconditionally) and has been removed, so no
            route can present a cache directory the interpreter reads as empty
            and thereby skip the builder.
        StoreContainmentError
            If the ``.npy`` path built for ``key`` would resolve outside the
            cache directory (STORE-02). A subclass of ``StoreKeyError``.
        KeyError
            If ``key`` is already present in the store.
        """
        paths.validate_store_key(key, self._cache_dir)

        if key in self:
            raise KeyError(f"Key {key} already exists in store.")

        enable_caching = enable_caching_override if enable_caching_override is not None else self._enable_caching
        # Unconditional: the former `if self._cache_dir else None` guard was
        # dead (`Path` defines no `__bool__`, and `__init__` always assigns a
        # `Path`), so it could only ever have looked like a route that skips
        # the builder — and with the builder now carrying the containment
        # check, a route that skips it is a route that skips validation.
        cache_path = paths.get_npy_path(self._cache_dir, key)
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
        # D-15-G1 — one of the four registration routes. Immediately after the
        # install, so a `del` / `pop` / `offload` between here and the next
        # `purge` cannot make the entry's armed finalizer unreachable.
        self._register_entry(key, new_container)

    @property
    def store(self) -> Mapping[str, Optional[T]]:
        """Return a **read-only view** of the mapping of keys to in-memory entries.

        The value is ``None`` where an entry is currently offloaded to disk.

        Mutating the returned mapping raises :exc:`TypeError` — assignment,
        ``del``, ``update``, ``pop``, ``clear`` and ``setdefault`` all refuse.
        Reads are unaffected: membership, iteration, ``len`` and subscript-read
        behave exactly as they did. The accessor is *narrowed*, not removed.

        Notes
        -----
        D-19 — **the route is closed structurally rather than validated.**
        Before this, the property handed out ``self._store`` itself, so
        ``store.store['../victim'] = entry`` installed an illegal key with no
        validation and ``keys()`` duly reported it. STORE-01's claim is written
        over *every* route that writes into the internal mapping — "including
        the routes that bypass the public insertion API" — so adding a fifth
        validation site here would have left that claim an **enumeration** of
        the routes someone remembered, rather than an **invariant**. A
        read-only view removes the route instead of guarding it: there is no
        write to validate, so there is no site to forget.

        ↻ **CORRECTED by Plan 14-12 (CR-02).** The paragraph above is kept
        exactly as it was written, and the *argument* in it is the thing worth
        keeping — but the claim as made was **premature**. When it was written,
        ``__getstate__`` was still a second route into the same mapping
        through a public protocol method: its ``self.__dict__.copy()`` is
        shallow, so the state it returned carried ``_store`` **itself**, and
        ``copy.copy(store)`` handed back a store sharing the original's entry
        mapping. What closed it is the detached snapshot in
        ``__getstate__`` (Plan 14-12); the claim holds now, and did not
        before. That is the argument proving itself: a claim quantified over
        *every route* but maintained by **enumerating** routes degrades
        silently at the next route somebody adds — which is precisely how this
        one was missed for a round. Do not restore the unqualified wording.

        ↻ **CORRECTED A THIRD TIME by Plan 14-18 (D-32, § WR-03).** Both
        paragraphs above stay exactly as written — the D-19 argument because it
        is right, and Plan 14-12's correction because it is the second of three
        instances of one defect and the sequence is the point. What is withdrawn
        is a single sentence, by annotation rather than by deletion: **"the claim
        holds now" was false when it was written.** Round 3 detached the
        snapshot on the way *out*; :meth:`__setstate__` still installed the
        caller's mapping on the way *in*, because ``self.__dict__.update(state)``
        binds the state's ``_store`` **object** onto the instance. Measured
        against the round-3 tree: ``store._store is state["_store"]`` was
        ``True``, and writing ``state["_store"]["../victim"] = None`` after the
        call put ``'../victim'`` in ``keys()``.

        **What this note claims, and what it deliberately does not.** The route
        closed in round 4 is :meth:`__setstate__`'s installation of the incoming
        mapping; the mechanism is a rebind to a fresh ``dict``, placed after the
        instance-dict update and before the reload loop; the plan is 14-18. It
        does **not** claim that every route is now covered. That claim has been
        made three times and falsified three times, each time by a route the
        asserting round had itself edited, and nothing in this module enforces
        it — the quantifier is maintained by whoever last enumerated the routes,
        which is the failure the paragraph above names one sentence before
        committing it. Three routes, three mechanisms, three plans is what is
        known: the ``store`` property (read-only view, 14-08), ``__getstate__``
        (detached snapshot, 14-12), ``__setstate__`` (detaching rebind, 14-18).

        *What would enforce the withdrawn claim, written down so the withdrawal
        does not read as a shrug —* **and explicitly not built here, not
        promised, and not part of this round.** *One mechanism rather than a
        longer list: make* ``_store`` *unreachable as a plain attribute — a
        private mapping type whose own* ``__setitem__`` *validates — so every
        write route that exists, or is ever added, passes through one site by
        construction and no route can be forgotten because none can bypass it.
        Until something of that shape exists, a per-route enumeration is a
        snapshot of one author's memory and should be written as one.*

        *Measured basis, not assumed.* Across ``30_GSEGUtils``,
        ``41_pchandler``, ``pc2img`` and ``iof3D`` there are **zero
        write-through sites** for this property; every measured usage is a
        read.

        This is deliberately the **same evidence principle as D-01's sealed
        ``cache_path`` setter**, and the pair should be read as one policy
        rather than two unrelated removals: in both cases a public mutation
        route was withdrawn outright, with no deprecation cycle, *because the
        measurement said nobody was using it that way*. Where the evidence says
        the opposite it is honoured the other way — D-15 gives the promoted
        path builders a full deprecation cycle precisely because they have
        measured live callers.

        *The one known downstream cost*, named here so the next reader meets it
        beside the change: pc2img's legacy alias property ``image_data``
        (``disk_backed_image_store.py``) simply returns ``self.store`` but is
        annotated ``-> dict[str, DiskBackedImageData | None]``. That annotation
        becomes wrong under a :class:`~types.MappingProxyType` return and turns
        pc2img's strict mypy red until it widens to
        :class:`~collections.abc.Mapping`. It is a one-line fix downstream, and
        it is a breaking change to a published API — carried in the migration
        note as **BC-GSEG-007**.

        Returns
        -------
        Mapping[str, Optional[T]]
            A read-only proxy over the internal mapping. It is a *view*, not a
            copy: entries offloaded or loaded after this call are visible
            through it.
        """
        return MappingProxyType(self._store)

    @property
    def cache_dir(self) -> Path:
        """Return the directory where offloaded codec pairs are written.

        The ``-> Path`` annotation is **enforced**, including across the one
        route that could falsify it. :meth:`__init__` assigns a ``Path`` on both
        of its branches, and since Plan 14-14 (§ WR-03) :meth:`__setstate__`
        refuses any pickled containment base that is not one — absent and
        ``None`` included — so this accessor cannot return a value the
        annotation forbids. Before that it could, and a type checker had no way
        to see it happen.
        """
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
                # D-15-G1 — the entry registry is deliberately left alone here.
                # This line is the one a reader will suspect, because it is what
                # makes `self._store.get(key)` return `None` for a still-live
                # entry; leaving the weak reference in place is exactly what
                # lets `purge` still find and disarm that entry's finalizer.
                self._store[key] = None
                logger.debug(
                    "Wrote codec pair for %s under %s and cleared in-memory reference.",
                    key,
                    paths.get_npy_path(self._cache_dir, key),
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
        npy_final = paths.get_npy_path(self._cache_dir, key)
        json_final = paths.get_meta_path(self._cache_dir, key)
        npy_tmp = paths.get_npy_tmp_path(self._cache_dir, key)
        json_tmp = paths.get_meta_tmp_path(self._cache_dir, key)
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
        npy_path = paths.get_npy_path(self._cache_dir, key)
        json_path = paths.get_meta_path(self._cache_dir, key)
        legacy_pkl = paths.get_legacy_pickle_path(self._cache_dir, key)
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

        The returned state is a **snapshot, detached at the mapping level**: its
        ``_store`` is a fresh :class:`dict`, not this instance's own mapping, so
        writing into it cannot reach the live store. It is **not** a deep copy —
        the entry *values* are the same objects — and nothing more should be
        read into the detachment than that.
        """
        if self._enable_caching:
            self.offload(pickle_container=True)
        state = self.__dict__.copy()
        # D-19 completed (Plan 14-12, CR-02). `__dict__.copy()` is SHALLOW, so
        # without this line `state["_store"] is self._store` and there are two
        # consequences — a reader who sees only the pickle one will eventually
        # "simplify" this away:
        #
        #   (a) the returned state is a LIVE WRITE ROUTE into `_store` through a
        #       public protocol method, with no key validation on it. That is
        #       the same shape as the `store` property CR-02 closed in round 1,
        #       one accessor over, and it is what made D-19's structural-closure
        #       claim false as written.
        #   (b) `copy.copy(store)` — which travels `__reduce_ex__` →
        #       `__getstate__` → `__setstate__` — handed back a store SHARING
        #       the original's entry mapping. Insert into the copy and the
        #       original grows the key; `del` from it and the original loses
        #       one. That half has nothing to do with keys: it is a plain
        #       data-integrity defect on a class explicitly designed to travel
        #       through joblib/loky.
        #
        # It must stay AFTER the conditional offload above, or the copied values
        # would be the pre-offload ones.
        state["_store"] = dict(self._store)
        # D-15-G1, in the same register as the D-19 note above: the weak entry
        # registry does not cross this boundary, and it is REMOVED here rather
        # than tolerated. A `weakref.ref` is not picklable, so leaving it in the
        # state makes `pickle.dumps(store)` raise
        # `TypeError: cannot pickle 'weakref.ReferenceType' object` — and the
        # joblib path force-offloads and pickles on EVERY dispatch, so that
        # would not be a rare failure but the common one.
        #
        # It rides `self.__dict__.copy()` unless this line exists, which is the
        # same shallow-copy footgun the line above it corrects, one attribute
        # over. `__setstate__` rebuilds the registry empty and re-registers
        # every live entry the restored mapping can see; what it deliberately
        # does NOT reconstruct is a registration for an entry the caller holds
        # but the mapping does not, because such an entry is simply not visible
        # across a pickle and claiming otherwise would be a lie in state.
        state.pop("_entry_registry", None)
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
            refuses (D-10), **or** if its ``_cache_dir`` is not a
            :class:`~pathlib.Path` — absent and ``None`` included, since
            :meth:`__init__` can produce neither (D-21 as tightened by § WR-03).

        Notes
        -----
        D-10 — RAISE. Unpickling is a trust boundary, and this module already
        treats it as one: :func:`_resolve_lazy_disk_cache_class` resolves
        sidecar class names through an explicit allow-list with no
        ``importlib`` fallback for exactly this reason. A *post-fix* pickle
        cannot legitimately carry an illegal key — so an illegal key arriving
        here means a legacy or a tampered pickle.

        **Where that guarantee actually comes from, stated precisely because an
        earlier version of this note got it wrong.** It does *not* come from
        :meth:`__getstate__`: that method offloads when caching is enabled and
        then copies the instance dictionary verbatim
        (``state = self.__dict__.copy()``), validating **no keys of any kind**.
        Reading it as a validating snapshot was a false premise, and
        verification falsified it. The guarantee comes instead from the write
        routes plus the read-only view: every route that installs a key into
        ``_store`` validates it — :meth:`__setitem__`, :meth:`add_data_to_store`
        and this method — the ``__init__`` rescan refuses to track one (D-09),
        and after D-19 the mapping is no longer handed out mutably through the
        public :attr:`store` accessor. So an illegal key cannot enter ``_store``
        after construction, and therefore cannot be in a snapshot this codebase
        produced. The conclusion below is unchanged; only its stated reason was
        wrong.

        ↻ **CORRECTED AGAIN by Plan 14-12 (CR-02).** The correction above is
        Plan 14-08's, and it stays: **a note that has been wrong twice, with
        both corrections visible, is a better warning to the next reader than a
        note that reads as though it had always been right.** Do not collapse
        these two into one tidy paragraph.

        What 14-08's replacement text got wrong is its *conclusion*. Having
        correctly quoted ``state = self.__dict__.copy()`` as validating no keys,
        it concluded that an illegal key "cannot be in a snapshot this codebase
        produced" — which did **not** follow from the quoted line, because that
        line **aliased** the mapping rather than copying it. The snapshot was
        the live ``_store``, so anyone holding a state object could write an
        illegal key straight into the store the snapshot came from, and the
        enumeration of write routes above was therefore incomplete at the moment
        it was written. The conclusion follows **now**: Plan 14-12 makes
        :meth:`__getstate__` return a detached ``dict``, so the snapshot is a
        copy, and every route that installs a key into ``_store`` validates it.

        ↻ **CORRECTED A THIRD TIME by Plan 14-18 (D-32, § WR-03).** Both
        corrections above stay, per the instruction two paragraphs up — and
        **this makes three, which is now the most useful thing this block
        says.** The sentence withdrawn here is *"the conclusion follows now"*:
        it did not follow, and the counterexample was again one line away, this
        time inside this very method. ``self.__dict__.update(state)`` installs
        the caller's ``_store`` **object**, so the restored store shared its
        entry mapping with whoever held the state, and a write into that state
        *after* the call installed a key into a store that had validated every
        key it was shown. Measured on the round-3 tree: identity ``True``, and
        ``keys()`` gaining ``'../victim'`` from a post-call write.

        *Reachability, stated rather than implied, because a deserialization
        defect reads as worse than it is unless its shape is named.*
        :func:`pickle.loads` and :func:`copy.copy` both discard the state
        mapping once restoration is done, so reaching this needs an explicit
        ``__setstate__`` call by a caller who retains the state — the shape this
        module's own tests use. That makes it narrower than the ``store``
        property's route, and it was still a live, unvalidated write route into
        ``_store`` on a trust boundary: precisely the thing the two paragraphs
        above assert does not exist.

        **The progression, which is the part worth carrying forward.** This is
        one defect on three surfaces of one protocol pair. The ``store``
        property handed out the live mapping (round 1, closed by D-19's
        read-only view); ``__getstate__`` handed out the same mapping through a
        shallow snapshot (round 2, closed by Plan 14-12's detached ``dict``);
        ``__setstate__`` installed the caller's mapping on the way in (round 3,
        closed by Plan 14-18's rebind below). Each was found by the round after
        the one that declared the set complete.

        **So this paragraph names a route and a mechanism and stops there.** The
        route is this method's installation of the incoming mapping; the
        mechanism is the rebind to a fresh ``dict``, placed after the
        instance-dict update and ahead of the reload loop; the plan is 14-18. No
        every-route quantifier is restored, because nothing enforces one — see
        the matching note on the :attr:`store` property for what would, and for
        why it is not built here.

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

        **D-21 — the containment base is state too, and it was the one thing
        crossing this boundary unchecked (WR-06).** Every incoming key used to be
        validated against ``_cache_dir``, which was then installed *verbatim from
        the same untrusted state*. ``_cache_dir`` is the authorization boundary
        every builder resolves against, so validating keys against a base the
        state itself chose does not constrain where bytes land: every key passes,
        and every path lands wherever the base points. The type guard below
        precedes the per-key loop for a concrete reason beyond tidiness — the
        refusal message for a bad key interpolates the incoming base, so a
        malformed base would otherwise be rendered into a message before anything
        noticed it was malformed.

        ↻ **TIGHTENED by Plan 14-14 (§ WR-03).** As first shipped the guard
        carved out an absent or explicitly-``None`` base, to protect the
        no-cache-path configuration. That configuration produces a
        ``mkdtemp`` :class:`~pathlib.Path`, never ``None``, so the carve-out
        protected nothing while admitting a state :meth:`__init__` cannot
        produce — after which the store accepted keys and died on the first path
        build with a bare :exc:`TypeError`. The test is now a single positive
        one: the base must **be** a :class:`~pathlib.Path`.

        **What that guard buys, and what it does not.** It catches *malformed or
        legacy state* — which is the case D-10 actually names. It does **not**
        defend against a hostile pickle: unpickling untrusted data is
        arbitrary-code execution regardless of any validation this method
        performs, so no check here is a security boundary against an attacker who
        already controls the byte stream. Stated plainly rather than left implied,
        because a guard on a deserialization route reads as a security boundary
        unless it says otherwise, and the trust-boundary framing above would
        otherwise be overclaimed.

        **No ``expected_cache_dir`` parameter was added.** The review proposed an
        opt-in check against a caller-supplied expected directory; it was
        considered and declined, because the user did not want the extra public
        surface — and a published parameter in a package on production PyPI
        cannot be withdrawn without a break. The refusal is part of D-21 rather
        than an omission; do not re-propose it as one.
        """
        incoming_store: dict[str, Any] = state.get("_store", {})
        incoming_cache_dir: Optional[Path] = state.get("_cache_dir")
        # D-21. An absent base and an explicitly-absent base both stay legal: a
        # store with no configured cache path is a supported configuration and
        # this guard must not break it. Only a *present, non-Path* base is
        # refused.
        #
        # ↻ CORRECTED by Plan 14-14 (§ WR-03). The sentence above is kept rather
        # than deleted, because half of it is true and the half that is not is
        # the instructive part. A store with no configured cache path IS a
        # supported configuration — and it does NOT produce either of the shapes
        # the carve-out admitted: `__init__` assigns a `Path` on BOTH of its
        # branches, falling back to `Path(tempfile.mkdtemp())` when
        # `config.cache_path` is `None`. So the carve-out protected no real
        # configuration while admitting a state the constructor cannot produce.
        #
        # What that bought, measured rather than argued: the store accepted keys,
        # tracked them, and then died at the first path build with
        # `TypeError: unsupported operand type(s) for /: 'NoneType' and 'str'` —
        # the untyped-crash class this phase replaced with typed refusals
        # everywhere else — while `cache_dir` returned `None` under a `-> Path`
        # annotation no type checker can see through a pickle.
        #
        # The test is therefore a single POSITIVE type test: absent, `None` and
        # every other non-`Path` value are refused alike, because `__init__` can
        # produce none of them. The configuration that genuinely needs protecting
        # is the no-cache-path store, which carries a `Path` like any other and
        # is covered by
        # `test_route_setstate_ordinary_round_trips_still_work_under_every_configuration`
        # and by the closing assertion of
        # `test_route_setstate_refuses_a_cache_dir_shape_init_cannot_produce`.
        #
        # The guard stays AHEAD of the per-key loop, and that placement is a
        # requirement rather than tidiness: the refusal message for a bad key
        # interpolates the incoming base, so a guard below the loop would render
        # a malformed base into a message before anything noticed it was
        # malformed. Pinned by `test_route_setstate_reports_the_base_before_the_keys`.
        if not isinstance(incoming_cache_dir, Path):
            raise paths.StoreKeyError(
                f"Invalid pickled cache directory {incoming_cache_dir!r}: the restored containment "
                f"base must be a pathlib.Path, but got {type(incoming_cache_dir).__name__}. "
                "__init__ assigns a Path on both of its branches — a configured cache_path, or a "
                "tempfile.mkdtemp() fallback — so no store this codebase constructs can present "
                "this shape. That makes it malformed or legacy pickled state; nothing has been "
                "restored."
            )
        for incoming_key in incoming_store:
            paths.validate_store_key(incoming_key, incoming_cache_dir)

        self.__dict__.update(state)
        # D-32 (Plan 14-18, § WR-03). Detach on the way IN as well as out — the
        # mirror of the Plan 14-12 fix in `__getstate__`, and the same defect on
        # the other side of the same protocol pair. `__dict__.update` installs
        # the caller's mapping OBJECT, so without this rebind the restored store
        # SHARES `_store` with whoever holds `state`: a write into the retained
        # state after this call lands in a store that validated every key it was
        # shown. Measured before the fix: `self._store is state["_store"]` was
        # `True`, and `state["_store"]["../victim"] = None` afterwards put
        # `'../victim'` in `keys()`.
        #
        # It must stay AHEAD of the reload loop below, which writes into the
        # mapping and must write into the detached one.
        #
        # Do NOT add a validation pass here. The per-key loop above already ran
        # against this exact mapping, and a second validation site is precisely
        # the enumeration approach D-19 argues against — the rebind closes the
        # route STRUCTURALLY, which is the whole point of doing it this way
        # rather than re-checking. Pinned by
        # `test_route_setstate_detaches_the_incoming_entry_mapping`.
        #
        # The source is `self._store` — which `__dict__.update` has just bound to
        # the incoming mapping — and NOT `incoming_store`, though the two are the
        # same object on every well-formed state. They differ on one input: a
        # state carrying no `_store` key at all, where `incoming_store` is the
        # `{}` default and rebinding from it would silently EMPTY a store this
        # method was never asked to change. This line does one thing, detach, and
        # `dict(self._store)` is not the no-op it looks like: it is the copy that
        # breaks the alias `__dict__.update` just created.
        self._store = dict(self._store)
        # D-15-G1 — rebuild the weak registry EMPTY. `__getstate__` removed it
        # (a `weakref.ref` is not picklable), so `__dict__.update` above has not
        # restored it and the attribute would otherwise be absent on every
        # unpickled or copied store. Assigned unconditionally rather than
        # defensively, so a hand-built state carrying one cannot install it.
        self._entry_registry = {}
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

        # D-15-G1 — RE-REGISTER, and this is load-bearing rather than
        # housekeeping. `copy.copy(store)` travels
        # `__reduce_ex__` -> `__getstate__` -> `__setstate__`, and
        # `__getstate__` hands over the SAME entry objects. A copy that came
        # back with an empty registry would have a `purge` that no longer
        # detaches even on the `tracked` route — the one route that worked
        # before this change — so the omission would be a silent regression of
        # existing behaviour rather than a missing new feature. Pinned by
        # `test_a_copied_store_still_detaches_on_purge`.
        #
        # One consolidated pass over the restored mapping, after both the state
        # restore and the reload loop, so there is a single place where the
        # question "which entries does this store now hold?" is answered.
        for restored_key, restored_entry in self._store.items():
            if restored_entry is not None:
                self._register_entry(restored_key, restored_entry)
