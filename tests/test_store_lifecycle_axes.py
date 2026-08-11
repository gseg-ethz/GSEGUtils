"""STORE-05 three-axis characterization suite for :class:`DiskBackedStore` (Plan 15-01).

This file pins the three lifecycle axes downstream consumers depend on **before**
Phase 15's ``purge`` primitive exists:

1. **Where the data lives** — ``offload(pickle_container=True)`` moves a payload from
   memory to the ``<key>.npy + <key>.meta.json`` codec pair and clears the in-memory slot.
2. **Whether the key is tracked** — ``__delitem__`` / ``pop`` / ``clear`` drop tracking and
   unlink nothing.
3. **Whether the file outlives the object** — ``purge_disk_on_gc`` governs the entry's
   ``weakref.finalize``, which owns the ``<key>.dat`` memmap and nothing else.

Every test here is written to **pass against the pre-change code** and to keep passing
after ``purge`` lands. That ordering is the ROADMAP's within-phase constraint: *a
primitive added before the axes are pinned is a primitive whose blast radius is
unmeasured.* If one of these goes red after a later Phase-15 plan, the primitive changed
an axis it was required to leave alone — the test is the finding, not the bug.

Two of the tests below are discussion-session **measurements promoted to standing
assertions**: the D-01 re-adoption repro (``del store[key]`` is not durable) and the D-12
repro (the codec pair must outlive an entry's GC). Both are recorded in
``15-CONTEXT.md``; they live here so they gate rather than merely document.

The suite uses the shared ``make_store`` / ``tmp_cache_dir`` fixtures from
``conftest.py`` rather than shadowing them — ``test_lazy_disk_cache.py``'s file-local
copies are the deliberate exception documented at ``conftest.py`` lines 7-10.
``pytest-randomly`` shuffles order, so nothing here carries cross-test state.
"""

import gc
from pathlib import Path
from typing import Callable

import numpy as np

from GSEGUtils.lazy_disk_cache.disk_backed_ndarray import DiskBackedNDArray
from GSEGUtils.lazy_disk_cache.disk_backed_store import DiskBackedStore
from GSEGUtils.lazy_disk_cache.lazy_disk_cache import LazyDiskCacheConfig

#: A store factory as injected by ``conftest.make_store``. Annotated structurally
#: rather than by importing the conftest ``MakeStore`` protocol, matching
#: ``test_store_containment.py:119`` — ``tests/`` is not a package, so a relative
#: import of ``conftest`` would not resolve.
MakeStore = Callable[..., DiskBackedStore[DiskBackedNDArray]]

_PAYLOAD = np.arange(4, dtype=np.float32)


def _artefact_paths(cache_dir: Path, key: str) -> tuple[Path, Path, Path]:
    """Return the ``(.npy, .meta.json, .dat)`` triple for ``key`` under ``cache_dir``."""
    return (
        cache_dir / f"{key}.npy",
        cache_dir / f"{key}.meta.json",
        cache_dir / f"{key}.dat",
    )


def _offloaded_store(
    make_store: MakeStore,
    cache_dir: Path,
    keys: tuple[str, ...],
    *,
    purge_disk_on_gc: bool = False,
) -> DiskBackedStore[DiskBackedNDArray]:
    """Build a store holding ``keys``, each offloaded to its codec pair on disk."""
    store = make_store(cache_dir, enable_caching=True, purge_disk_on_gc=purge_disk_on_gc)
    for key in keys:
        store.add_data_to_store(key, _PAYLOAD.copy())
    store.offload(pickle_container=True)
    return store


# ---------------------------------------------------------------------------
# Group A — axis 2: tracking is in-memory only (SC-4).
#
# `__delitem__`, `pop` and `clear` are one mechanism: `pop` and `clear` both
# route through `__delitem__`, which drops tracking and unlinks nothing. The
# ROADMAP's *Out of Scope* row ("Making `__delitem__` purge") is what these
# three tests defend: a convenience call that silently discarded a downstream
# consumer's resumable session cache is the failure Phase 15 exists to make
# impossible.
# ---------------------------------------------------------------------------


def test_delitem_drops_tracking_and_unlinks_nothing(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-01 / STORE-05 axis 2 / SC-4: ``del store[key]`` leaves all three artefacts on disk."""
    key = "feat"
    store = _offloaded_store(make_store, tmp_cache_dir, (key,))
    npy, meta, dat = _artefact_paths(tmp_cache_dir, key)
    assert npy.exists() and meta.exists() and dat.exists(), (
        "precondition: offload(pickle_container=True) must materialise the codec pair beside the .dat memmap"
    )

    del store[key]

    assert key not in store, "STORE-05 axis 2: __delitem__ must drop tracking"
    assert list(store.keys()) == [], "STORE-05 axis 2: the key must be gone from keys()"
    assert npy.exists(), (
        "SC-4 violated: __delitem__ unlinked <key>.npy — tracking and deletion must stay separate verbs"
    )
    assert meta.exists(), (
        "SC-4 violated: __delitem__ unlinked <key>.meta.json (ROADMAP Out of Scope: making __delitem__ purge)"
    )
    assert dat.exists(), (
        "SC-4 violated: __delitem__ unlinked <key>.dat (ROADMAP Out of Scope: making __delitem__ purge)"
    )


def test_pop_drops_tracking_and_unlinks_nothing(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-01 / STORE-05 axis 2 / SC-4: ``store.pop(key)`` leaves every artefact on disk.

    ``pop`` routes through ``__delitem__`` (it reads the payload first, which lazily
    reloads an offloaded key, then deletes the tracking entry). The returned entry is
    dropped and collected here so the assertion covers the finalizer too.
    """
    key = "k"
    store = _offloaded_store(make_store, tmp_cache_dir, (key,))
    npy, meta, dat = _artefact_paths(tmp_cache_dir, key)

    popped = store.pop(key)
    assert popped is not None, "precondition: pop must return the reloaded entry"
    del popped
    gc.collect()

    assert key not in store, "STORE-05 axis 2: pop must drop tracking"
    assert npy.exists(), "SC-4 violated: pop unlinked <key>.npy — a session-resume cache must survive pop()"
    assert meta.exists(), "SC-4 violated: pop unlinked <key>.meta.json — a session-resume cache must survive pop()"
    assert dat.exists(), "SC-4 violated: pop unlinked <key>.dat — a session-resume cache must survive pop()"


def test_clear_empties_the_mapping_and_unlinks_nothing(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-01 / STORE-05 axis 2 / SC-4: ``store.clear()`` is not cache-destroying.

    Two keys, six artefacts. ``clear()`` empties the mapping and every one of the six
    files stays. This is iof3D's session-resume cache surviving a convenience call.
    """
    store = _offloaded_store(make_store, tmp_cache_dir, ("a", "b"))
    expected = [p for key in ("a", "b") for p in _artefact_paths(tmp_cache_dir, key)]
    assert all(p.exists() for p in expected), "precondition: both keys must have all three artefacts on disk"

    store.clear()

    assert len(store) == 0, "STORE-05 axis 2: clear() must empty the mapping"
    assert list(store.keys()) == [], "STORE-05 axis 2: clear() must leave no tracked key"
    survivors = sorted(p.name for p in expected if p.exists())
    assert survivors == sorted(p.name for p in expected), (
        "SC-4 violated: store.clear() destroyed on-disk cache artefacts "
        f"(survivors={survivors}); ROADMAP Out of Scope forbids making clear()/__delitem__ purge"
    )


# ---------------------------------------------------------------------------
# Group B — axis 3: file lifetime.
#
# `purge_disk_on_gc` governs the entry's `weakref.finalize`, whose legitimate
# job is the `.dat` memmap. Both sides of the threshold are checked, one step
# either side, with an explicit `del entry; gc.collect()` rather than a scope
# exit — CPython's refcounting would otherwise make the timing incidental.
#
# These construct `DiskBackedNDArray` instances directly, so the assertion is
# about the flag and not about store bookkeeping. They are deliberately NOT the
# pickle-round-trip finalizer tests in `test_lazy_disk_cache.py` (which assert
# on FRAG-03 re-registration); no pickle is involved here.
# ---------------------------------------------------------------------------


def test_purge_disk_on_gc_defaults_to_true() -> None:
    """Plan 15-01 / STORE-05 axis 3: the default is ``True``, by construction not by round-trip."""
    assert LazyDiskCacheConfig().purge_disk_on_gc is True, (
        "STORE-05 axis 3: the purge_disk_on_gc default must stay True — Phase 15 is additive and MUST NOT change it"
    )


def test_purge_disk_on_gc_true_unlinks_the_memmap_at_collection(tmp_cache_dir: Path) -> None:
    """Plan 15-01 / STORE-05 axis 3 (upper side): a ``True`` entry's ``.dat`` is unlinked at GC."""
    entry = DiskBackedNDArray(
        _PAYLOAD.copy(),
        enable_caching=True,
        cache_path=tmp_cache_dir / "t",
        purge_disk_on_gc=True,
    )
    dat = Path(entry.cache_path) if entry.cache_path is not None else None
    assert dat is not None and dat.name == "t.dat", "precondition: _init_from_config re-suffixes cache_path to .dat"
    entry.offload()
    assert dat.exists(), "precondition: the memmap must be on disk before the collection"

    del entry
    gc.collect()

    assert not dat.exists(), (
        "STORE-05 axis 3 regressed: purge_disk_on_gc=True must still unlink <key>.dat when the entry is collected"
    )


def test_purge_disk_on_gc_false_preserves_the_memmap_at_collection(tmp_cache_dir: Path) -> None:
    """Plan 15-01 / STORE-05 axis 3 (lower side): a ``False`` entry's ``.dat`` survives the same GC.

    ``purge_disk_on_gc=False`` is iof3D's configured mode (``base.yaml``); the file
    surviving collection is the whole point of that configuration.
    """
    entry = DiskBackedNDArray(
        _PAYLOAD.copy(),
        enable_caching=True,
        cache_path=tmp_cache_dir / "f",
        purge_disk_on_gc=False,
    )
    dat = Path(entry.cache_path) if entry.cache_path is not None else None
    assert dat is not None and dat.name == "f.dat", "precondition: _init_from_config re-suffixes cache_path to .dat"
    entry.offload()
    assert dat.exists(), "precondition: the memmap must be on disk before the collection"

    del entry
    gc.collect()

    assert dat.exists(), (
        "STORE-05 axis 3 regressed: purge_disk_on_gc=False must leave <key>.dat on disk across the entry's "
        "collection — this is the durable mode iof3D's session resume depends on"
    )
