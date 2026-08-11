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
import pickle
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


# ---------------------------------------------------------------------------
# Group C — the purge-intent pickle round-trip (STORE-05's third clause).
#
# This is the loky-facing axis: a store travels to a joblib worker by pickle,
# and the property that matters is that collecting the worker's copy does not
# delete the parent process's cache. The intent is carried in the `.meta.json`
# sidecar (`_store_entry` writes `purge_disk_on_gc`; `_load_entry` replays it
# into the reconstructed entry), so a round-trip that dropped it would silently
# flip a `False` entry to the `True` default and start deleting on GC.
# ---------------------------------------------------------------------------


def test_pickle_round_trip_restores_purge_intent_per_entry(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-01 / STORE-05 / D-05: an in-process pickle round-trip preserves each entry's purge intent.

    Two entries with opposite intents, so the assertion cannot pass by the round-trip
    defaulting everything to ``True`` (the config default) or everything to ``False``.
    """
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False)
    store.add_data_to_store("keep", _PAYLOAD.copy(), purge_disk_on_gc_override=False)
    store.add_data_to_store("gone", _PAYLOAD.copy(), purge_disk_on_gc_override=True)
    assert store["keep"].purge_disk_on_gc is False, "precondition: the 'keep' entry must be constructed False"
    assert store["gone"].purge_disk_on_gc is True, "precondition: the 'gone' entry must be constructed True"

    restored = pickle.loads(pickle.dumps(store))

    assert restored["keep"].purge_disk_on_gc is False, (
        "STORE-05 purge-intent round-trip regressed: a False entry came back True — a worker would start "
        "deleting files the parent configured to survive"
    )
    assert restored["gone"].purge_disk_on_gc is True, (
        "STORE-05 purge-intent round-trip regressed: a True entry came back False — the memmap would leak"
    )
    assert restored._cache_dir == store._cache_dir, (
        "STORE-05 purge-intent round-trip regressed: the restored store must point at the same cache directory"
    )


def test_collecting_the_restored_store_leaves_the_originals_files_on_disk(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-01 / STORE-05 / T-15-02: collecting an unpickled copy must not destroy the parent's cache.

    This is the destructive half of the axis, and the reason the assertion is on **file
    existence** rather than on a flag: the failure it guards is a joblib worker's copy
    being collected and taking the parent process's cache with it.

    **The artefact set is snapshotted after ``pickle.dumps``, deliberately.**
    ``DiskBackedStore.__getstate__`` force-calls ``offload(pickle_container=True)``,
    which writes each codec pair, clears the in-memory slot and drops the entry — so the
    ``purge_disk_on_gc=True`` entry is collected *during pickling* and its ``.dat`` is
    unlinked right there (measured, Plan 15-01). The parent's own artefact set at the
    moment the copy exists is therefore both codec pairs plus the ``False`` entry's
    ``.dat``, and that is exactly the set this test requires to survive. A ``<key>.dat``
    that the *restored* store recreates on reload belongs to the copy, not to the parent.
    """
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False)
    store.add_data_to_store("keep", _PAYLOAD.copy(), purge_disk_on_gc_override=False)
    store.add_data_to_store("gone", _PAYLOAD.copy(), purge_disk_on_gc_override=True)

    blob = pickle.dumps(store)
    owned_by_parent = sorted(p.name for p in tmp_cache_dir.iterdir())
    assert owned_by_parent == ["gone.meta.json", "gone.npy", "keep.dat", "keep.meta.json", "keep.npy"], (
        "precondition drifted: the parent's post-pickle artefact set is not what Plan 15-01 measured "
        f"(got {owned_by_parent})"
    )

    restored = pickle.loads(blob)
    del restored
    gc.collect()

    survivors = sorted(p.name for p in tmp_cache_dir.iterdir())
    missing = [name for name in owned_by_parent if name not in survivors]
    assert missing == [], (
        "T-15-02 violated: collecting the unpickled copy destroyed files belonging to the original store "
        f"({missing}); this is a joblib worker deleting the parent process's cache"
    )
    assert np.array_equal(store["keep"], _PAYLOAD), "the parent must still be able to reload 'keep' after the copy died"
    assert np.array_equal(store["gone"], _PAYLOAD), "the parent must still be able to reload 'gone' after the copy died"


def test_getstate_detaches_the_entry_mapping(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-01 / Phase-14 D-19 / CR-02: ``__getstate__`` hands back a detached ``_store``.

    One line, and it is the invariant the Group E re-adoption test would otherwise be
    able to corrupt: if the snapshot shared the live mapping, a write into retained state
    would land in a store that validated every key it was shown.
    """
    store = _offloaded_store(make_store, tmp_cache_dir, ("feat",))

    state = store.__getstate__()

    assert "_cache_dir" in state, "__getstate__ must carry _cache_dir (it is the restored containment base)"
    assert "_store" in state, "__getstate__ must carry _store"
    assert state["_store"] is not store._store, (
        "Phase-14 D-19/CR-02 regressed: __getstate__ returned the LIVE entry mapping, making the snapshot a "
        "write route into the store with no key validation on it"
    )


# ---------------------------------------------------------------------------
# Group D — axis 1: where the data lives.
#
# The offload axis, observed at the store's own bookkeeping (`_store[key]`)
# rather than inferred from the files alone. Reaching into `_store` is
# established practice in this suite; it is what distinguishes "offloaded"
# (slot cleared, payload on disk) from "never inserted".
# ---------------------------------------------------------------------------


def test_offload_moves_the_payload_to_the_codec_pair_and_reload_restores_it(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-01 / STORE-05 axis 1: insert → offload → lazy reload, each step observable.

    The three states of the axis, in order: in memory (slot populated, no ``.npy``); on
    disk (codec pair written, slot cleared); and back in memory after a subscript read.
    ``purge`` must leave every step of this sequence exactly as it is.
    """
    key = "x"
    arr = np.arange(6, dtype=np.float32)
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False)
    store.add_data_to_store(key, arr.copy())
    npy, meta, _dat = _artefact_paths(tmp_cache_dir, key)

    assert store._store[key] is not None, "STORE-05 axis 1: a freshly inserted entry lives in memory"
    assert not npy.exists(), "STORE-05 axis 1: no codec pair may be written before offload(pickle_container=True)"

    store.offload(pickle_container=True)

    assert npy.exists(), "STORE-05 axis 1: offload(pickle_container=True) must write <key>.npy"
    assert meta.exists(), "STORE-05 axis 1: offload(pickle_container=True) must write <key>.meta.json"
    assert store._store[key] is None, "STORE-05 axis 1: offload must clear the in-memory slot"

    assert np.array_equal(store[key], arr), "STORE-05 axis 1: the lazy reload must return the array that was written"
    assert store._store[key] is not None, "STORE-05 axis 1: the lazy reload must re-populate the in-memory slot"


# ---------------------------------------------------------------------------
# Group E — the D-01 measurement, promoted from transcript to assertion.
#
# `del store[key]` is *untrack-temporarily*. `__getitem__` does
# `self._store.get(key, None)` and falls straight through to `_load_entry`
# (disk_backed_store.py:451) — it never asks whether the key was DELIBERATELY
# dropped — and the `__init__` rescan globs `*.npy` and re-adopts anything with
# a codec pair. Both routes undo the removal.
# ---------------------------------------------------------------------------


def test_delitem_is_undone_by_the_very_next_read(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-01 / STORE-05 / D-01: ``del store[key]`` is not durable — the next read re-adopts it.

    The three assertions are in one function on purpose, in this order, so the property
    cannot regress half-way: ``key not in store`` → ``store[key]`` **succeeds** →
    ``key in store`` again. Between the first two, membership answers ``False`` while the
    subscript answers with data, which is a :class:`~typing.Mapping` contract violation.

    **This is recorded as a known limitation, not a designed guarantee** —
    ``__delitem__``'s own docstring says so — and this test is here so nobody "fixes"
    ``__delitem__`` into a purge. It is also the whole reason ``purge`` is a distinct
    verb (D-01) rather than a flag: ``purge`` is the only removal that sticks, and it
    sticks *because* it unlinks, leaving nothing on disk to re-adopt.
    """
    key = "feat"
    store = _offloaded_store(make_store, tmp_cache_dir, (key,))

    del store[key]

    assert key not in store, "D-01 step 1: __delitem__ must drop tracking"
    reloaded = store[key]
    assert np.array_equal(reloaded, _PAYLOAD), (
        "D-01 step 2 regressed: store[key] no longer re-adopts a deleted-but-still-on-disk key. If this is "
        "now a KeyError, __delitem__ has been given unlink behaviour — that is forbidden (SC-4); the durable "
        "removal is purge()"
    )
    assert key in store, "D-01 step 3: the read must have re-tracked the key"


def test_a_fresh_store_re_adopts_a_deleted_key(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-01 / STORE-05 / D-01: the reopen rescan re-adopts a key that ``__delitem__`` dropped.

    The second half of the D-01 transcript, and the one a downstream consumer actually
    meets: reopening a cache directory in a later session re-adopts every key with a
    codec pair on disk, including ones a previous session deleted from tracking.
    """
    key = "feat"
    store = _offloaded_store(make_store, tmp_cache_dir, (key,))
    del store[key]
    assert key not in store, "precondition: the key must be untracked in the original store"

    fresh = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False)

    assert key in fresh, (
        "D-01 regressed: a freshly constructed store over the same cache directory no longer re-adopts the "
        "key. The rescan globs *.npy; if the codec pair is gone, something unlinked it"
    )
    assert list(fresh.keys()) == [key], (
        f"D-01: the rescan must track exactly the key on disk (got {list(fresh.keys())})"
    )


# ---------------------------------------------------------------------------
# Group F — the D-12 measurement, as the guard against making the dead branch
# live.
#
# `_purge_cache_pair`'s `.npy` branch (lazy_disk_cache.py:68-82) is
# unconditionally dead: `_load_entry` passes `cache_path=str(npy_path)` but
# `_init_from_config` re-suffixes to `.dat`, so no route in the package produces
# a `LazyDiskCache` whose `cache_path.suffix == ".npy"`.
#
# Phase 14's D-14 established the branch was DEAD. D-12 established it is
# HARMFUL, by measurement: make it live and the codec pair is unlinked the
# moment the entry is collected, so the first lazy reload after GC fails with
# `KeyError`. The `.npy`/`.meta.json` pair surviving an entry's GC is therefore
# REQUIRED, not leaked — the pair is store-owned (written by `_store_entry`,
# read by `_load_entry`) while the finalizer's legitimate job is the `.dat`
# memmap and nothing else. The FRAG-03/W-1 intent quoted in that helper's own
# docstring is obsolete.
# ---------------------------------------------------------------------------


def test_codec_pair_survives_the_entrys_collection_and_reload_still_works(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-01 / STORE-06 / D-12: an entry's GC must leave ``<key>.npy`` + ``<key>.meta.json`` on disk.

    ``purge_disk_on_gc=True`` here — the config default, and the only configuration in
    which the finalizer runs at all, so this is the case where a live ``.npy`` branch
    would fire. The finalizer takes the ``.dat`` memmap, which is its job. It must take
    nothing else.

    **Anyone who "completes" ``_purge_cache_pair``'s** ``.npy`` **branch ships silent
    cache destruction at an arbitrary GC, and this is the test that will go red.**
    Measured (Plan 15-01 mutation proof): dropping the ``if cache_path.suffix == ".npy"``
    guard so the sidecar is unlinked for a ``.dat`` path leaves ``['feat.npy']`` in the
    directory and the reload below raises ``KeyError: 'feat'`` — the exact D-12 shape.
    """
    key = "feat"
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)
    store.add_data_to_store(key, _PAYLOAD.copy())
    entry = store._store[key]
    assert entry is not None and entry.purge_disk_on_gc is True, (
        "precondition: the entry must carry the default True purge intent, or the finalizer never runs"
    )
    store.offload(pickle_container=True)
    npy, meta, dat = _artefact_paths(tmp_cache_dir, key)

    del entry
    gc.collect()

    assert npy.exists(), (
        "D-12 violated: <key>.npy was unlinked when the entry was collected. The codec pair is STORE-owned; "
        "the finalizer owns the .dat memmap and nothing else — this is silent cache destruction"
    )
    assert meta.exists(), (
        "D-12 violated: <key>.meta.json was unlinked when the entry was collected — this is exactly what "
        "making _purge_cache_pair's .npy branch live does, and the reload below cannot survive it"
    )
    assert not dat.exists(), "axis 3: the finalizer's legitimate job — unlinking <key>.dat — must still happen"
    assert np.array_equal(store[key], _PAYLOAD), (
        "D-12 violated: the first lazy reload after the entry's GC failed. The codec pair surviving the "
        "collection is REQUIRED, not leaked"
    )
