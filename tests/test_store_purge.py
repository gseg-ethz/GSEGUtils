"""STORE-04 end-to-end proof for :meth:`DiskBackedStore.purge` (Plan 15-02).

The primitive's whole contract, proven **by listing the directory** rather than by
reading the implementation: one ``purge(key)`` call drops the key from tracking and
removes every artefact whose name derives from that key, built through the ``paths``
builder seam so an illegal or escaping key is refused before anything is touched.

Two instrument choices in this file are deliberate and should not be "simplified":

1. **Artefact assertions are per exact builder-produced path, never by name prefix.**
   A ``startswith("k0.")`` filter is the wrong instrument: ``"k0."`` is also a prefix
   of ``"k0.bar.npy"``, so in a directory holding an adjacent key it silently asserts
   something stronger than SC-1 says, and in a single-key directory it asserts nothing
   the six exact checks do not already cover. The whole-directory *equality* check that
   accompanies them is the secondary instrument, kept because an exact-name check cannot
   catch a seventh artefact nobody thought of and an equality can.
2. **The six paths are built through the same ``paths`` builders the implementation
   uses.** Restating them as string literals would let the test and the implementation
   drift into two vocabularies, which is the split D-14 exists to have closed.

``pytest-randomly`` shuffles test order, so nothing here carries cross-test state.
"""

import copy
import errno
import gc
import logging
import os
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

# ``tests/`` is not a package, but pytest's prepend import mode puts it on
# ``sys.path``, so a sibling test module is importable by its plain name. The
# refusal matrix is imported rather than restated so this file and Phase 14's
# rule cannot drift into two disagreeing definitions of "illegal key".
from test_store_key_rules import ACCEPTED_KEYS, REFUSED_KEYS

from GSEGUtils.lazy_disk_cache import StoreKeyError, StorePurgeIncompleteError, StorePurgeRefusedError
from GSEGUtils.lazy_disk_cache import paths as paths_mod
from GSEGUtils.lazy_disk_cache.disk_backed_ndarray import DiskBackedNDArray
from GSEGUtils.lazy_disk_cache.disk_backed_store import DiskBackedStore

#: A store factory as injected by ``conftest.make_store``. Annotated structurally
#: rather than by importing the conftest ``MakeStore`` protocol — ``tests/`` is not a
#: package, so a relative import of ``conftest`` would not resolve
#: (``test_store_containment.py:119``).
MakeStore = Callable[..., DiskBackedStore[DiskBackedNDArray]]

_PAYLOAD = np.arange(4, dtype=np.float32)

#: The logger ``purge``'s D-03 override record is emitted on.
_STORE_LOGGER = "GSEGUtils.lazy_disk_cache.disk_backed_store"


def _six_artefact_paths(cache_dir: Path, key: str) -> dict[str, Path]:
    """Return the six builder-produced artefact paths for ``key``, by artefact name.

    Built through the same seam the implementation builds through (D-14), so the
    test cannot assert against a second, drifting vocabulary. The mapping keys are
    the artefact names, used verbatim in failure messages so a red assertion names
    *which* artefact survived.
    """
    return {
        "<key>.meta.json": paths_mod.get_meta_path(cache_dir, key),
        "<key>.meta.json.tmp": paths_mod.get_meta_tmp_path(cache_dir, key),
        "<key>.npy.tmp": paths_mod.get_npy_tmp_path(cache_dir, key),
        "<key>.dat.tmp": paths_mod.get_memmap_tmp_path(cache_dir, key),
        "<key>.npy": paths_mod.get_npy_path(cache_dir, key),
        "<key>.dat": paths_mod.get_memmap_path(cache_dir, key),
    }


def _offloaded_store(
    make_store: MakeStore,
    cache_dir: Path,
    key: str,
    *,
    purge_disk_on_gc: bool = False,
) -> DiskBackedStore[DiskBackedNDArray]:
    """Build a store holding ``key``, offloaded so all three artefact families exist.

    ``purge_disk_on_gc`` defaults to ``False`` here because that is the only
    configuration in which the ``.dat`` memmap survives ``offload(pickle_container=True)``
    — with ``True`` the entry is collected during the offload and its finalizer takes
    the ``.dat`` right there (measured in 15-01, Finding 1). A test that wants all six
    families on disk at once needs the durable mode.
    """
    store = make_store(cache_dir, enable_caching=True, purge_disk_on_gc=purge_disk_on_gc)
    store.add_data_to_store(key, _PAYLOAD.copy())
    store.offload(pickle_container=True)
    return store


def _assert_all_six_gone(cache_dir: Path, key: str) -> None:
    """Assert each of the six exact builder-produced paths for ``key`` is absent."""
    for artefact, path in _six_artefact_paths(cache_dir, key).items():
        assert not path.exists(), (
            f"SC-1 violated: {artefact} survived purge({key!r}) at {str(path)!r}. "
            "purge must remove every artefact whose name derives from the key (D-14)."
        )


# ---------------------------------------------------------------------------
# The headline proof — one key, the whole derived set, by directory listing.
# ---------------------------------------------------------------------------


def test_purge_removes_every_derived_artefact_and_drops_the_key(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-02 / STORE-04 / SC-1 as amended by D-14.

    The end-to-end path through every layer the phase touches: path vocabulary →
    store method → filesystem → directory listing. All three artefact families exist
    before the call; nothing derived from the key exists after it, and the key is gone
    from tracking.
    """
    key = "k0"
    store = _offloaded_store(make_store, tmp_cache_dir, key)
    six = _six_artefact_paths(tmp_cache_dir, key)
    assert six["<key>.npy"].exists() and six["<key>.meta.json"].exists(), (
        "precondition: offload(pickle_container=True) must have written the codec pair"
    )
    assert six["<key>.dat"].exists(), "precondition: the .dat memmap must survive offload under purge_disk_on_gc=False"

    store.purge(key)

    assert key not in store, "STORE-04: purge must drop the key from tracking"
    assert list(store.keys()) == [], "STORE-04: the key must be gone from keys()"
    _assert_all_six_gone(tmp_cache_dir, key)
    # Secondary, and not redundant: an exact-name check cannot catch a seventh
    # artefact nobody thought of; an equality against the whole listing can.
    assert sorted(p.name for p in tmp_cache_dir.iterdir()) == [], (
        "SC-1 violated: the cache directory still holds something after purging its only key"
    )


def test_purge_leaves_no_trace_of_the_key_after_a_fresh_store_reopens_the_directory(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-02 / STORE-04 / D-01.

    The durability half of D-01, stated as the difference from ``__delitem__``: a
    purged key does not come back. The reopen rescan globs ``*.npy`` and re-adopts any
    key with a codec pair, which is exactly why ``del store[key]`` is undone by a fresh
    store — and exactly why a purge is not. With nothing on disk there is nothing to
    re-adopt.
    """
    key = "feat"
    store = _offloaded_store(make_store, tmp_cache_dir, key)
    store.purge(key)

    reopened = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False)

    assert key not in reopened, "D-01: a purged key must not be re-adopted by a fresh store over the same directory"
    with pytest.raises(KeyError):
        _ = reopened[key]


# ---------------------------------------------------------------------------
# SC-2 — a refused purge is a bit-for-bit no-op.
# ---------------------------------------------------------------------------


def test_purge_refuses_an_escaping_key_before_touching_anything(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-02 / STORE-04 / SC-2 / T-15-04.

    ``'../victim'`` violates the lexical rule, so it raises the base ``StoreKeyError``
    rather than the ``StoreContainmentError`` subclass — the layer-order property
    Phase 14 pinned. The point of the test is what comes *after* the raise: the cache
    directory's listing and every file's bytes are unchanged, and the file the key was
    aiming at is untouched. Ordering is what makes that true; a validate-after-unlink
    implementation would pass a "raises" assertion and fail this one.
    """
    key = "k0"
    store = _offloaded_store(make_store, tmp_cache_dir, key)
    victim = tmp_cache_dir.parent / "victim"
    victim.write_bytes(b"belongs to someone else")

    before_listing = sorted(p.name for p in tmp_cache_dir.iterdir())
    before_bytes = {p.name: p.read_bytes() for p in tmp_cache_dir.iterdir() if p.is_file()}

    with pytest.raises(StoreKeyError):
        store.purge("../victim")

    assert sorted(p.name for p in tmp_cache_dir.iterdir()) == before_listing, (
        "SC-2 violated: a refused purge changed the cache directory listing"
    )
    assert {p.name: p.read_bytes() for p in tmp_cache_dir.iterdir() if p.is_file()} == before_bytes, (
        "SC-2 violated: a refused purge changed a file's bytes"
    )
    assert victim.read_bytes() == b"belongs to someone else", (
        "T-15-04 violated: a refused purge reached a path outside the cache directory"
    )
    assert key in store, "SC-2 violated: a refused purge dropped an unrelated key from tracking"


# ---------------------------------------------------------------------------
# D-02 — the missing-key contract.
# ---------------------------------------------------------------------------


def test_purge_removes_an_untracked_but_on_disk_key(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-02 / STORE-04 / D-02.

    Untracked-but-on-disk counts as present. That is exactly the state
    ``del store[key]`` leaves behind and exactly the state a caller most needs to clean
    up, so a "tracked only" reading would make the orphan case unpurgeable through the
    one verb built to purge it.
    """
    key = "orphan"
    store = _offloaded_store(make_store, tmp_cache_dir, key)
    del store[key]
    assert key not in store, "precondition: __delitem__ drops tracking"
    assert _six_artefact_paths(tmp_cache_dir, key)["<key>.npy"].exists(), (
        "precondition: __delitem__ leaves the artefacts on disk (SC-4)"
    )

    store.purge(key)

    _assert_all_six_gone(tmp_cache_dir, key)
    assert sorted(p.name for p in tmp_cache_dir.iterdir()) == []


def test_purge_raises_key_error_when_absent_from_tracking_and_from_disk(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-02 / STORE-04 / D-02.

    ``KeyError`` only when the key exists in *neither* place. Not a ``bool`` return: a
    silent no-op on a typo'd key would make the removal verbs disagree with
    ``__delitem__`` and ``pop`` about missing keys.
    """
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False)

    with pytest.raises(KeyError):
        store.purge("never_existed")

    assert sorted(p.name for p in tmp_cache_dir.iterdir()) == [], (
        "a refused-as-missing purge must not have created or removed anything"
    )


# ---------------------------------------------------------------------------
# D-03 — an explicit purge wins over purge_disk_on_gc=False, transparently.
# ---------------------------------------------------------------------------


def test_purge_overrides_purge_disk_on_gc_false_and_says_so(
    make_store: MakeStore, tmp_cache_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Plan 15-02 / STORE-04 / D-03 + the transparency prohibition.

    ``purge_disk_on_gc`` governs *implicit, GC-time* deletion; it is not a
    write-protect bit, and reading it as one would make the primitive unusable in
    precisely the configuration that accumulates the most artefacts. The override is
    permitted **and recorded**: an INFO record naming the key is what stops a
    destructive override of a configured durability intent from being silent (T-15-10).
    """
    key = "durable"
    store = _offloaded_store(make_store, tmp_cache_dir, key, purge_disk_on_gc=False)

    with caplog.at_level(logging.INFO, logger=_STORE_LOGGER):
        store.purge(key)

    _assert_all_six_gone(tmp_cache_dir, key)
    override_records = [r for r in caplog.records if r.levelno == logging.INFO and key in r.getMessage()]
    assert override_records, (
        "D-03 transparency violated: purging a purge_disk_on_gc=False store emitted no INFO record naming the key. "
        f"Records seen: {[r.getMessage() for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# D-09 — the legacy pickle is left alone.
# ---------------------------------------------------------------------------


def test_purge_leaves_the_legacy_pickle_beside_the_codec_pair(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-02 / STORE-04 / D-09.

    ``get_legacy_pickle_path`` is deliberately not among the six paths ``purge``
    builds. The consequence is recorded as deferred rather than hidden: a ``.pkl`` is
    unreadable by design and now unremovable by the only removal verb, so a listing
    after a "complete" purge still shows the key's name.
    """
    key = "legacy"
    store = _offloaded_store(make_store, tmp_cache_dir, key)
    pkl = paths_mod.get_legacy_pickle_path(tmp_cache_dir, key)
    pkl.write_bytes(b"pre-0.5 pickle")

    store.purge(key)

    _assert_all_six_gone(tmp_cache_dir, key)
    assert pkl.exists(), "D-09 violated: purge removed the legacy <key>.pkl"
    assert pkl.read_bytes() == b"pre-0.5 pickle", "D-09 violated: purge rewrote the legacy <key>.pkl"
    assert sorted(p.name for p in tmp_cache_dir.iterdir()) == [f"{key}.pkl"], (
        "the .pkl must be the only thing a complete purge leaves behind"
    )


# ---------------------------------------------------------------------------
# T-15-07 — the ABA seam: detach the finalizer, do not flip the flag.
#
# The four-route coverage lives in `tests/test_store_purge_identity.py`
# (`test_the_aba_hazard_cannot_fire_on_any_route_out_of_the_mapping`), added by
# 15-08 and turned green by 15-09. This test and the two ABA tests below it stay:
# WR-05 was about their REACH — all three take the `tracked` route, the one route
# where the pre-15-09 lookup happened to be correct — and not about their
# correctness, which is undiminished. They remain valid narrow assertions on that
# route, and they assert things the parameterised body does not (the purge-intent
# flag, the replacement entry's own finalizer).
# ---------------------------------------------------------------------------


def test_purge_detaches_the_live_entrys_finalizer_without_flipping_its_purge_intent(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-02 / STORE-04 / T-15-07 + the sanctioned-detach criterion.

    Detaching without flipping is the whole difference between the sanctioned
    ``self._finalizer.detach()`` and the forbidden ``disable_purge()``, which performs
    the same detach *and* sets ``_purge_disk_on_gc = False`` — a flag the
    ``__getstate__``/``__setstate__`` loky dance snapshots and replays. Only a
    behavioural assertion can tell the two apart, so this asserts on both halves: the
    finalizer is dead, and the entry still reports its configured purge intent.
    """
    key = "aba"
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)
    store.add_data_to_store(key, _PAYLOAD.copy())
    entry = store[key]
    assert entry.purge_disk_on_gc is True, "precondition: the entry was constructed with purge_disk_on_gc=True"
    assert entry._finalizer.alive, "precondition: a purge_disk_on_gc=True entry registers a finalizer"

    store.purge(key)

    assert not entry._finalizer.alive, (
        "T-15-07 violated: purge left the entry's weakref.finalize live, so collecting the entry later would "
        "unlink whatever occupies its recorded path — including a later entry created under the same key"
    )
    assert entry.purge_disk_on_gc is True, (
        "purge flipped the entry's purge intent. That is what disable_purge() does and why it is the forbidden "
        "call here: the flag is snapshotted and replayed by the __getstate__/__setstate__ loky dance"
    )


def test_a_purged_keys_stale_finalizer_cannot_eat_a_later_entry_under_the_same_key(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-02 / STORE-04 / T-15-07 — the ABA hazard, end to end.

    The detach is not decoration. Purge a key, re-add it (which allocates a fresh
    ``<key>.dat``), then collect the *original* entry. Without the detach its stale
    ``weakref.finalize`` fires against its recorded path and deletes the new entry's
    file — a deletion attributable to nothing in the caller's code, at an arbitrary GC.
    """
    key = "reused"
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)
    store.add_data_to_store(key, _PAYLOAD.copy())
    old_entry = store[key]

    store.purge(key)
    store.add_data_to_store(key, _PAYLOAD.copy())
    dat = paths_mod.get_memmap_path(tmp_cache_dir, key)
    assert dat.exists(), "precondition: re-adding the key allocates a fresh <key>.dat"

    del old_entry
    gc.collect()

    assert dat.exists(), (
        "T-15-07 violated (ABA): collecting the purged entry deleted the LATER entry's <key>.dat. "
        "purge must detach the finalizer before unlinking"
    )
    assert np.array_equal(np.asarray(store[key]), _PAYLOAD), "the later entry's payload must still be readable"


# ---------------------------------------------------------------------------
# T-15-06 — the worker guard (D-05 / D-06 / D-08).
#
# Raw `os.fork` rather than a loky worker, deliberately: `joblib` is not a
# GSEGUtils dependency, `os.fork` needs none, and D-05's own measurement used it.
# The child always leaves via `os._exit`, never `sys.exit` — the latter would run
# the parent's pytest atexit handlers inside the child and corrupt the report.
# ---------------------------------------------------------------------------

#: Child exit statuses. Distinct per outcome so a red assertion names *which*
#: wrong thing the child did rather than merely that it did not refuse.
_CHILD_REFUSED = 17
_CHILD_PURGED = 18
_CHILD_WRONG_EXCEPTION = 19
_CHILD_UNREACHED = 20

_FORK_ONLY = pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork is POSIX-only")


def _status_from_fork(child: Callable[[], int]) -> int:
    """Run ``child`` in a forked process and return its exit status.

    The child never returns through this frame: it leaves via ``os._exit`` in the
    ``finally``, so no pytest teardown, no ``atexit`` handler and no buffered stream
    of the parent's is ever flushed twice.
    """
    pid = os.fork()
    if pid == 0:  # pragma: no cover - executes only in the forked child
        status = _CHILD_UNREACHED
        try:
            status = child()
        finally:
            os._exit(status)
    _, wait_status = os.waitpid(pid, 0)
    assert os.WIFEXITED(wait_status), f"forked child did not exit normally: {wait_status}"
    return os.WEXITSTATUS(wait_status)


def _purge_outcome(store: DiskBackedStore[DiskBackedNDArray], key: str) -> int:
    """Attempt ``store.purge(key)`` and report the outcome as an exit status."""
    try:
        store.purge(key)
    except StorePurgeRefusedError:
        return _CHILD_REFUSED
    except BaseException:
        return _CHILD_WRONG_EXCEPTION
    return _CHILD_PURGED


@_FORK_ONLY
def test_a_forked_childs_purge_refuses_and_the_parents_files_survive(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-02 / STORE-04 / D-05 + D-06 / T-15-06 / SC-3.

    The whole point of the guard, and the reason D-03 is safe: a downstream
    consumer runs ``purge_disk_on_gc=False`` for session resume, so a stray purge in
    a tile worker would delete the parent process's session data. It refuses, and it
    refuses *before touching anything* — which is what the second half of this test
    measures. A guard placed after the unlink loop would still raise and would still
    fail this.
    """
    key = "parentdata"
    store = _offloaded_store(make_store, tmp_cache_dir, key)
    six = _six_artefact_paths(tmp_cache_dir, key)

    status = _status_from_fork(lambda: _purge_outcome(store, key))

    assert status == _CHILD_REFUSED, (
        f"SC-3 violated: a purge from a forked child exited {status} rather than {_CHILD_REFUSED}. "
        f"{_CHILD_PURGED} means it destroyed the parent's data; {_CHILD_WRONG_EXCEPTION} means it "
        "raised something other than StorePurgeRefusedError"
    )
    for artefact in ("<key>.npy", "<key>.meta.json", "<key>.dat"):
        assert six[artefact].exists(), (
            f"SC-3 violated: the forked child unlinked the parent's {artefact}. The refusal must "
            "precede every mutation, not follow it"
        )


@_FORK_ONLY
def test_a_forked_child_that_unpickles_the_store_also_refuses(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-02 / STORE-04 / D-05 — the pickle route, end to end.

    The route a real loky worker takes. The pid rides ``__getstate__``'s
    ``__dict__.copy()`` and ``__setstate__``'s ``__dict__.update`` with no pickle
    plumbing of its own, so the restored copy carries the *parent's* pid and compares
    unequal to the child's own.
    """
    key = "shipped"
    store = _offloaded_store(make_store, tmp_cache_dir, key)
    blob = pickle.dumps(store)
    six = _six_artefact_paths(tmp_cache_dir, key)

    def child() -> int:
        return _purge_outcome(pickle.loads(blob), key)

    assert _status_from_fork(child) == _CHILD_REFUSED, "SC-3 violated: an unpickled store purged from a foreign process"
    for artefact in ("<key>.npy", "<key>.meta.json", "<key>.dat"):
        assert six[artefact].exists(), f"SC-3 violated: the unpickled child unlinked the parent's {artefact}"


@_FORK_ONLY
def test_a_forked_child_may_still_offload_the_guard_is_on_purge_only(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-02 / STORE-04 / D-08 — the positive control.

    Workers legitimately **write**: ``__getstate__`` force-calls
    ``offload(pickle_container=True)`` before pickling, so a guard on the write routes
    would break the joblib path outright. Deletion is the only operation where "wrong
    process" means "destroying someone else's data". Without this control the guard
    could spread to every disk-mutating route and every other test here would still
    pass.
    """
    key = "written_by_worker"
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False)
    store.add_data_to_store(key, _PAYLOAD.copy())
    six = _six_artefact_paths(tmp_cache_dir, key)
    assert not six["<key>.npy"].exists(), "precondition: the codec pair is not written until offload"

    def child() -> int:
        store.offload(pickle_container=True)
        return 0

    assert _status_from_fork(child) == 0, (
        "D-08 violated: the process guard spread to a write route — a forked child could not offload"
    )
    assert six["<key>.npy"].exists() and six["<key>.meta.json"].exists(), (
        "D-08 violated: the forked child's offload produced no codec pair, so the control proves nothing"
    )


def test_copy_copy_in_the_constructing_process_purges_normally(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-02 / STORE-04 / D-05 — the case the rejected alternative got wrong.

    ``copy.copy`` travels ``__reduce_ex__`` → ``__getstate__`` → ``__setstate__``, so a
    "reconstructed copy" flag stamped in ``__setstate__`` would refuse here — a
    false positive on a legitimate same-process copy, and the measured reason that
    alternative was rejected. The pid comparison gets it right because the pid *is*
    the same. Asserted on the file listing, not merely on the absence of an exception.
    """
    key = "sameproc"
    store = _offloaded_store(make_store, tmp_cache_dir, key)
    twin = copy.copy(store)

    twin.purge(key)

    _assert_all_six_gone(tmp_cache_dir, key)
    assert sorted(p.name for p in tmp_cache_dir.iterdir()) == [], (
        "a same-process copy must purge exactly as the original would"
    )


def test_the_owner_pid_survives_the_pickle_protocol(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-02 / STORE-04 / D-05 — the mechanism, pinned independently of the fork harness.

    If this ever goes red while the fork tests stay green, the guard has started
    working by accident rather than by the attribute it is documented to use.
    """
    store = _offloaded_store(make_store, tmp_cache_dir, "roundtrip")

    restored = pickle.loads(pickle.dumps(store))

    assert restored._owner_pid == os.getpid(), (
        "D-05 violated: _owner_pid did not ride __getstate__/__setstate__, so a worker copy would "
        "compare its own pid against a value this process never set"
    )
    assert store._owner_pid == os.getpid()


def test_the_refusal_message_names_the_key_and_both_pids(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-02 / STORE-04 / D-06 + Phase-14 D-13's message register.

    The foreign process is simulated in-process by moving the *owner* pid rather than
    the caller's, so the message can be matched deterministically without a fork. A
    refusal that does not say which key, which owner and which caller leaves the
    reader of a worker traceback with nothing to act on.
    """
    key = "named"
    store = _offloaded_store(make_store, tmp_cache_dir, key)
    foreign_owner = os.getpid() + 1
    store._owner_pid = foreign_owner

    with pytest.raises(StorePurgeRefusedError, match=rf"'{key}'.*{foreign_owner}.*{os.getpid()}"):
        store.purge(key)

    _assert_at_least_the_offloaded_three_survive(tmp_cache_dir, key)


def _assert_at_least_the_offloaded_three_survive(cache_dir: Path, key: str) -> None:
    """Assert the three artefacts an offloaded key owns survived a refused purge."""
    six = _six_artefact_paths(cache_dir, key)
    for artefact in ("<key>.npy", "<key>.meta.json", "<key>.dat"):
        assert six[artefact].exists(), (
            f"SC-2 violated: a refused purge unlinked {artefact} — the refusal must precede every mutation"
        )


# ===========================================================================
# Plan 15-03 — the STORE-04 contract battery.
#
# THE D-10 / SC-2 BOUNDARY, AND THIS PLAN HONOURS IT.
# ---------------------------------------------------
# SC-2's bit-for-bit no-op covers a **refused** purge: a refusal raised before
# any mutation — an illegal key, an escaping key, or a call from a process that
# did not construct the store. It does **not** cover a *partially failed* one.
# D-10 makes a partial directory state the stated contract rather than a
# discovered surprise: POSIX gives no atomicity across N unlinks, so purge
# attempts all six, collects the failures and raises once, and the key stays
# dropped. No test in this file may assert a no-op — of listing, of bytes or of
# tracking — after a partial unlink failure. Such a test would contradict D-10,
# and would pass only for as long as nobody made the failure real.
# ===========================================================================


def _snapshot(cache_dir: Path) -> dict[str, bytes]:
    """Return a byte-level snapshot of ``cache_dir``: file name → file contents.

    **Contents, not just names.** SC-2's wording is that the directory listing
    *and* every file's contents are byte-identical across a refused purge, and
    the two halves fail differently: a listing comparison stays green while a
    file has been truncated to zero bytes in place, which is exactly the shape
    a "unlink, then discover the key was illegal" implementation would leave if
    it opened before it unlinked. One mapping carries both halves — a missing or
    extra entry is a listing change, a differing value is a content change — so
    a single ``==`` covers the whole property and a failure prints which file.

    Non-regular entries are recorded by name with a sentinel value rather than
    dropped, so a directory appearing (or a file becoming one) is still a
    difference rather than an invisible no-op.
    """
    return {
        path.name: (path.read_bytes() if path.is_file() else _NOT_A_REGULAR_FILE)
        for path in sorted(cache_dir.iterdir())
    }


#: Snapshot placeholder for a directory entry that is not a regular file. Its
#: only job is to be stable and to be distinguishable from any real payload.
_NOT_A_REGULAR_FILE = b"\x00<not a regular file>"

#: The two keys :func:`_two_key_store` populates. Named rather than inlined so a
#: failure message can say which of the two the assertion was about.
_KEY_A = "alpha"
_KEY_B = "beta"


def _two_key_store(make_store: MakeStore, cache_dir: Path) -> DiskBackedStore[DiskBackedNDArray]:
    """Return a fully-populated store: two offloaded keys plus a planted legacy pickle.

    "Fully populated" is load-bearing for the SC-2 group rather than decorative.
    A refusal that touched nothing is trivially provable over an empty
    directory; the property worth pinning is that a refusal over a directory
    with something to lose still loses none of it. The legacy ``alpha.pkl``
    is planted because D-09 excludes it from the six derived paths, so it is the
    one file whose survival a *successful* purge also guarantees — having it in
    the snapshot keeps the two guarantees from being confused for each other.

    ``purge_disk_on_gc=False`` is the durable mode: it is the only configuration
    in which the ``.dat`` memmap survives ``offload(pickle_container=True)``
    (measured in 15-01, Finding 1), so it is the only one where all three
    artefact families are on disk at once.
    """
    store = make_store(cache_dir, enable_caching=True, purge_disk_on_gc=False)
    store.add_data_to_store(_KEY_A, _PAYLOAD.copy())
    store.add_data_to_store(_KEY_B, _PAYLOAD.copy() * 2)
    store.offload(pickle_container=True)
    paths_mod.get_legacy_pickle_path(cache_dir, _KEY_A).write_bytes(b"pre-0.5 pickle")
    for key in (_KEY_A, _KEY_B):
        six = _six_artefact_paths(cache_dir, key)
        for artefact in ("<key>.npy", "<key>.meta.json", "<key>.dat"):
            assert six[artefact].exists(), f"precondition: {key}'s {artefact} must be on disk before a refusal test"
    return store


# ---------------------------------------------------------------------------
# SC-2, refusal kind 1 — an illegal key.
# ---------------------------------------------------------------------------

#: The four illegal-key shapes this group drives — a nested key, a
#: backslash-bearing key, an absolute key and the empty key — **selected from**
#: :data:`test_store_key_rules.REFUSED_KEYS` rather than restated inline. The
#: selection is a filter over that one matrix, so a key Phase 14 ever moves out
#: of the refused set drops out here too and :func:`test_the_illegal_key_selection_still_resolves`
#: goes red, instead of this file quietly asserting a refusal the rule no longer
#: makes.
_ILLEGAL_PURGE_KEYS: tuple[str, ...] = tuple(
    key for key, _clause in REFUSED_KEYS if key in {"a/b", "..\\..\\x", "/etc/passwd", ""}
)


def test_the_illegal_key_selection_still_resolves() -> None:
    """Plan 15-03 / SC-2 — the guard on the selection above.

    ``@pytest.mark.parametrize`` over an empty sequence generates **zero** tests
    and reports nothing wrong, so a selection that silently stopped matching
    would take the whole refusal group with it and the suite would still be
    green. This is the assertion that makes that impossible.
    """
    assert len(_ILLEGAL_PURGE_KEYS) == 4, (
        "the illegal-key selection no longer resolves against REFUSED_KEYS; the four shapes it needs are a "
        f"nested key, a backslash-bearing key, an absolute key and the empty key. Resolved: {_ILLEGAL_PURGE_KEYS!r}"
    )


@pytest.mark.parametrize("illegal_key", _ILLEGAL_PURGE_KEYS)
def test_purge_refusing_an_illegal_key_is_byte_identical(
    make_store: MakeStore, tmp_cache_dir: Path, illegal_key: str
) -> None:
    """Plan 15-03 / STORE-04 / SC-2.

    The lexical refusal, proven as a *no-op* rather than merely as a raise.
    ``pytest.raises`` alone cannot tell a purge that refused first from one that
    unlinked first and refused afterwards — both raise, and only the second
    destroys data. The snapshot is what separates them, and it compares file
    **bytes**, not only names, so a truncate-then-refuse implementation is
    caught as well as an unlink-then-refuse one.
    """
    store = _two_key_store(make_store, tmp_cache_dir)
    before = _snapshot(tmp_cache_dir)
    tracked_before = sorted(store.keys())

    with pytest.raises(StoreKeyError):
        store.purge(illegal_key)

    assert _snapshot(tmp_cache_dir) == before, (
        f"SC-2 violated: refusing the illegal key {illegal_key!r} changed the cache directory — "
        "either its listing or some file's bytes"
    )
    assert sorted(store.keys()) == tracked_before, (
        f"SC-2 violated: refusing the illegal key {illegal_key!r} changed which keys the store tracks"
    )


# ---------------------------------------------------------------------------
# SC-2, refusal kind 2 — a foreign process.
# ---------------------------------------------------------------------------


def test_purge_refusing_a_foreign_process_is_byte_identical(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-03 / STORE-04 / SC-2 / D-05.

    The second refusal kind, held to the same standard as the first. The foreign
    process is simulated by moving the store's **own** ``_owner_pid`` rather than
    by forking — a direct, honest simulation that exercises the same comparison
    the fork tests in 15-02 drive from a real child, and the one that makes the
    *whole directory* assertable, which a forked child cannot do from inside its
    own address space.

    Reaching into ``_owner_pid`` is established practice in this file
    (``test_the_refusal_message_names_the_key_and_both_pids``); what is new here
    is asserting the byte-level no-op rather than the message.
    """
    store = _two_key_store(make_store, tmp_cache_dir)
    before = _snapshot(tmp_cache_dir)
    tracked_before = sorted(store.keys())
    store._owner_pid = os.getpid() + 1

    with pytest.raises(StorePurgeRefusedError):
        store.purge(_KEY_A)

    assert _snapshot(tmp_cache_dir) == before, (
        "SC-2 violated: a purge refused on process identity changed the cache directory. The guard must sit "
        "above every mutation — relocated below the unlink loop it still raises and still looks like a guard"
    )
    assert sorted(store.keys()) == tracked_before, (
        "SC-2 violated: a purge refused on process identity dropped the key from tracking anyway"
    )
    assert _KEY_A in store, "SC-2 violated: the refused key must still be tracked"


# ---------------------------------------------------------------------------
# SC-2 — the ordering control (B-10).
# ---------------------------------------------------------------------------

#: A message no production code path can produce, so the ordering test's
#: ``match=`` proves the raise came from the injected validator rather than from
#: the real rule happening to refuse the key for its own reasons.
_ORDERING_SENTINEL = "ordering-control-sentinel"


def test_purge_validates_before_it_mutates(
    make_store: MakeStore, tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 15-03 / STORE-04 / SC-2 / T-15-11 — the B-10 control.

    The key here is **legal**; the validator is made to raise anyway. That is
    what turns the test into a statement about *order* rather than about the
    key rule: whatever the validator does, nothing may have been touched by the
    time it does it.

    Why the control is needed rather than obvious: the natural way to write a
    purge inverts mutate and validate — build the names, unlink them, and let
    the builder's own validation raise on the way past — and a raise-only
    assertion stays green under that inversion. This one does not. **Mutation
    proof recorded in the SUMMARY:** with the unlink loop moved above the
    validation call (and the paths built by concatenation, which is what such an
    implementation would do), this test goes red on the snapshot while
    ``pytest.raises`` still passes.

    Note that patching ``paths.validate_store_key`` reaches every path builder
    too, since :func:`~GSEGUtils.lazy_disk_cache.paths._build` calls the same
    module global. That is a feature: it means the assertion holds against the
    *whole* validate-first layer, not only against ``purge``'s own call to it.
    """
    store = _two_key_store(make_store, tmp_cache_dir)
    before = _snapshot(tmp_cache_dir)
    tracked_before = sorted(store.keys())

    def boom(key: str, cache_dir: Path | None = None) -> None:
        raise StoreKeyError(f"{_ORDERING_SENTINEL}: refusing {key!r}")

    monkeypatch.setattr(paths_mod, "validate_store_key", boom)

    with pytest.raises(StoreKeyError, match=_ORDERING_SENTINEL):
        store.purge(_KEY_A)

    assert _snapshot(tmp_cache_dir) == before, (
        "SC-2 violated: purge mutated the cache directory before its validation raised. Validation is the "
        "first statement in the method precisely so a refusal cannot arrive after the damage"
    )
    assert sorted(store.keys()) == tracked_before, (
        "SC-2 violated: purge dropped the key from tracking before its validation raised"
    )


# ---------------------------------------------------------------------------
# SC-3 — the ABA hazard, forced with an explicit collection.
# ---------------------------------------------------------------------------


def test_the_aba_hazard_cannot_fire_under_a_forced_collection(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-03 / STORE-04 / SC-3 / T-15-12.

    A ``weakref.finalize`` deletes whatever occupies **its recorded path at
    collection time**, not the object it was registered for. Purge a key and
    re-add it and the recorded path now names a *different* entry's file, so a
    surviving finalizer from the first lifetime is a delayed deletion of the
    second entry's data — attributable to nothing in the caller's code and
    arriving at whichever arbitrary GC happens to collect the original. The
    detach in ``purge`` is the whole of what prevents it.

    Two things make this test different from
    ``test_a_purged_keys_stale_finalizer_cannot_eat_a_later_entry_under_the_same_key``,
    which 15-02 wrote, rather than a restatement of it:

    * the replacement carries a **different array**, so the readback assertion
      distinguishes "the new entry's data" from "the old entry's data", which an
      equal-payload version cannot;
    * both finalizers are asserted on directly — the original's dead, the
      replacement's alive — which pins that the detach was scoped to the purged
      entry rather than applied to the key's registration generally.

    The collection is forced with an explicit ``del`` plus ``gc.collect()``, the
    idiom this suite uses throughout (``test_lazy_disk_cache.py:380-505``), never
    a scope exit: CPython's refcount would usually collect at the ``del`` anyway,
    but "usually" is not an assertion and a reference cycle would defer it.

    **The replacement is deliberately not offloaded**, and the reason is measured
    rather than assumed: under ``purge_disk_on_gc=True``,
    ``offload(pickle_container=True)`` collects the entry during the offload and
    its own live finalizer takes the ``.dat`` right there — leaving
    ``['<key>.meta.json', '<key>.npy']`` and nothing for a stale finalizer to
    eat. ``add_data_to_store`` materialises the ``.dat`` eagerly, so the file
    this test needs on disk is already there without the offload.
    """
    key = "aba_forced"
    replacement = _PAYLOAD.copy() + 100.0
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)
    store.add_data_to_store(key, _PAYLOAD.copy())
    original_entry = store[key]
    original_finalizer = original_entry._finalizer
    assert original_finalizer.alive, "precondition: a purge_disk_on_gc=True entry registers a finalizer"

    store.purge(key)
    store.add_data_to_store(key, replacement)
    replacement_entry = store[key]
    dat = paths_mod.get_memmap_path(tmp_cache_dir, key)
    assert dat.exists(), "precondition: re-adding the key allocates a fresh <key>.dat"
    assert original_entry is not replacement_entry, "precondition: the re-add produced a genuinely new entry"
    # Recorded, not asserted, *here*: the file assertion below is the headline
    # property and must be the one that fires first when the detach is removed.
    # Captured before the collection because a stale finalizer that has already
    # fired also reports ``alive is False`` — after the collection the flag no
    # longer distinguishes "detached" from "fired and ate the file".
    detached_before_collection = not original_finalizer.alive

    del original_entry
    gc.collect()

    assert dat.exists(), (
        "SC-3 violated (ABA): forcing collection of the purged entry deleted the LATER entry's <key>.dat. "
        "purge must detach the stale finalizer before unlinking, because the finalizer deletes whatever "
        "occupies its recorded path at collection time"
    )
    assert np.array_equal(np.asarray(store[key]), replacement), (
        "SC-3 violated: after collecting the purged entry the key no longer reads back the replacement array"
    )
    assert detached_before_collection, (
        "purge left the purged entry's finalizer live. The file survived this run only because the collection "
        "happened to be harmless; the hazard is still armed"
    )
    assert replacement_entry._finalizer.alive, (
        "the detach must be scoped to the purged entry: the replacement registered under the same key still "
        "needs its own live finalizer, or the key's artefacts would leak forever after any purge"
    )


# ---------------------------------------------------------------------------
# STORE-04 / idempotency probe — the second call changes nothing.
# ---------------------------------------------------------------------------


def test_purging_the_same_key_twice_changes_nothing_the_second_time(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-03 / STORE-04 — the idempotency probe, answered rather than assumed.

    **The answer:** ``purge`` is idempotent *in effect* — the second call changes
    nothing — and it signals the already-gone state with ``KeyError`` rather than
    a ``bool`` return. Two reasons, both of which are decisions rather than
    accidents. A ``-> bool`` would make a typo'd key a silent no-op, which is the
    failure mode a destructive verb can least afford; and it would put the
    removal verbs out of step with ``__delitem__`` and ``pop``, which raise on a
    missing key, so a reader could no longer carry one mental model across the
    three.

    The property is only true *because* D-10 rejected re-tracking the key on
    failure: a purge that put the key back would leave the second call with
    something to do, and the two calls would differ.
    """
    key = "twice"
    store = _offloaded_store(make_store, tmp_cache_dir, key)

    store.purge(key)
    after_first = _snapshot(tmp_cache_dir)

    with pytest.raises(KeyError):
        store.purge(key)

    assert _snapshot(tmp_cache_dir) == after_first, (
        "idempotency violated: the second purge of an already-purged key changed the cache directory"
    )

    with pytest.raises(KeyError):
        store.purge("never_inserted_at_all")

    assert _snapshot(tmp_cache_dir) == after_first, (
        "purging a key that was never inserted and is not on disk must change nothing"
    )
    assert list(store.keys()) == [], "neither failed call may have re-tracked anything"


# ---------------------------------------------------------------------------
# D-02 — the orphan case, stated against what a reopen would have done.
# ---------------------------------------------------------------------------


def test_purge_reaches_an_orphan_a_fresh_store_would_have_re_adopted(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-03 / STORE-04 / D-02.

    ``del store[key]`` untracks *temporarily*: the reopen rescan globs ``*.npy``
    and re-adopts anything with a codec pair, so the key comes back on the next
    fresh store over the same directory. That is what makes the state an
    **orphan** rather than a deletion, and it is measured here on both sides of
    the purge rather than described — a fresh store sees the key before, and does
    not after.

    A "tracked only" reading of ``purge`` would make exactly this state
    unpurgeable through the one verb built to purge it, which is why D-02 takes
    the looser reading of "present".
    """
    key = "orphan_readopt"
    store = _offloaded_store(make_store, tmp_cache_dir, key)
    del store[key]
    assert key not in store, "precondition: __delitem__ drops tracking"
    assert key in make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False), (
        "precondition: the artefacts are still on disk, so a fresh store re-adopts the key — that is the "
        "orphan state D-02 is about"
    )

    store.purge(key)

    _assert_all_six_gone(tmp_cache_dir, key)
    assert key not in make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False), (
        "D-02 violated: a fresh store still re-adopts the key, so the orphan's artefacts were not removed"
    )
    assert sorted(p.name for p in tmp_cache_dir.iterdir()) == [], (
        "the cache directory must be empty after purging its only (orphaned) key"
    )


# ---------------------------------------------------------------------------
# D-03 — the override record, and the shape it is logged in.
# ---------------------------------------------------------------------------


def test_the_override_record_passes_the_key_as_a_lazy_logging_argument(
    make_store: MakeStore, tmp_cache_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Plan 15-03 / STORE-04 / D-03 / CWE-117.

    ``purge_disk_on_gc=False`` is iof3D's configured mode, so this is the
    configuration the primitive is actually used in, and the one that accumulates
    the most artefacts. The override is permitted (the flag governs *implicit,
    GC-time* deletion, not write protection) **and** recorded, which is what keeps
    a destructive override of a configured durability intent from being silent.

    What this test adds over 15-02's message assertion is the *shape*: the key
    must arrive as a lazy ``%s`` **argument** and must not appear in the format
    string. An f-string-interpolated key would render identically in ``caplog``
    and would carry a caller-controlled string straight into the log record's
    template — the CWE-117 log-injection shape Phase 14's CR-01 fixed on the
    rescan warning. Only asserting on ``record.args`` versus ``record.msg`` can
    tell the two apart.
    """
    key = "durable_args"
    store = _offloaded_store(make_store, tmp_cache_dir, key, purge_disk_on_gc=False)

    with caplog.at_level(logging.INFO, logger=_STORE_LOGGER):
        store.purge(key)

    _assert_all_six_gone(tmp_cache_dir, key)
    override_records = [
        record
        for record in caplog.records
        if record.levelno == logging.INFO and record.args is not None and key in tuple(record.args)  # type: ignore[arg-type]
    ]
    assert override_records, (
        "D-03 transparency violated: no INFO record carried the key as a logging argument. "
        f"Records seen: {[(r.msg, r.args) for r in caplog.records]}"
    )
    assert all(key not in str(record.msg) for record in override_records), (
        "CWE-117: the key was interpolated into the log record's format string instead of being passed as a "
        "lazy %s argument, so a caller-controlled string is now part of the template"
    )


# ---------------------------------------------------------------------------
# D-09 — the legacy pickle survives, and stays unreachable.
# ---------------------------------------------------------------------------


def test_the_surviving_legacy_pickle_is_not_re_adopted_by_a_fresh_store(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-03 / STORE-04 / D-09 — the deferred consequence, made concrete.

    D-09 keeps ``<key>.pkl`` out of the six derived paths, and 15-02 pinned that
    it survives. The half worth adding is what that survival *means* for a
    reader: a ``.pkl`` is unreadable by design (``_load_entry`` treats a legacy
    pickle without a codec pair as a cache miss and says so at INFO), so after a
    "complete" purge the directory still shows the key's name while no store can
    reach it. Permanently unreachable garbage in pre-0.5 cache directories —
    recorded as deferred rather than hidden, and asserted here so the deferral is
    a measured state rather than a claim.
    """
    key = "legacy_unreachable"
    store = _offloaded_store(make_store, tmp_cache_dir, key)
    pkl = paths_mod.get_legacy_pickle_path(tmp_cache_dir, key)
    pkl.write_bytes(b"pre-0.5 pickle")

    store.purge(key)

    _assert_all_six_gone(tmp_cache_dir, key)
    assert pkl.exists() and pkl.read_bytes() == b"pre-0.5 pickle", "D-09 violated: purge touched the legacy pickle"

    reopened = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False)

    assert key not in reopened, "the surviving .pkl must not make the key re-adoptable"
    with pytest.raises(KeyError):
        _ = reopened[key]


# ---------------------------------------------------------------------------
# SC-4 — the removal verbs stay in-memory-only.
# ---------------------------------------------------------------------------


def test_clear_pop_and_del_never_reach_purge(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-03 / STORE-04 / SC-4.

    Asserted from the *purge* side, complementing 15-01's axis tests: the three
    ``MutableMapping`` removal routes untrack and nothing more. If any of them
    ever grew a call to ``purge``, the byte snapshot would change and this test
    would say which route did it.

    The distinction is the phase's whole reason for existing. ``del`` / ``pop`` /
    ``clear`` are *untrack temporarily* — undone by the next reopen rescan —
    while ``purge`` is the removal that sticks. Collapsing the two would make
    every ``del store[key]`` in existing downstream code silently destructive.
    """
    keys = ("by_del", "by_pop", "by_clear")
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False)
    for index, key in enumerate(keys):
        store.add_data_to_store(key, _PAYLOAD.copy() + index)
    store.offload(pickle_container=True)
    before = _snapshot(tmp_cache_dir)
    assert len(before) == 9, f"precondition: three offloaded keys own three artefacts each, saw {sorted(before)}"

    del store[keys[0]]
    _ = store.pop(keys[1])
    store.clear()

    assert list(store.keys()) == [], "precondition: all three routes untracked their keys"
    assert _snapshot(tmp_cache_dir) == before, (
        "SC-4 violated: one of __delitem__, pop or clear unlinked an artefact. The removal verbs are "
        "in-memory-only; purge is the only route that touches disk"
    )
    reopened = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False)
    assert sorted(reopened.keys()) == sorted(keys), (
        "SC-4 violated: the untracked keys were not re-adopted by a fresh store, so something removed their "
        "codec pairs after all"
    )


# ---------------------------------------------------------------------------
# T-15-14 — prefix adjacency, asserted over exact names only (review finding F6).
# ---------------------------------------------------------------------------


def test_purging_foo_leaves_every_exact_artefact_of_foo_bar(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-03 / STORE-08 adjacency probe / T-15-14 / review finding F6.

    Two keys whose derived names share a prefix must neither merge nor collide.
    ``purge`` unlinks six enumerated builder-produced paths and never globs, so
    the property should hold — and the point of the test is that the *assertion*
    must be able to detect it if it stopped.

    **Why every assertion here is over exact names and not over a prefix.**
    ``"foo."`` is a prefix of ``"foo.bar.npy"``. A check of the form "no name
    beginning ``foo.`` remains" is therefore not a weaker version of the right
    assertion, it is a *different and false* one: it fires on every one of
    ``foo.bar``'s artefacts, which must all survive. Filtering the family back
    out restores a hand-written exclusion list that can itself be wrong, at which
    point it is a worse spelling of the exact-name checks. The instrument is
    withdrawn phase-wide.

    SC-1's whole-directory listing check reads as "nothing derived from *this*
    key", and a listing *equality* is a valid instrument only in a directory
    known to hold one key — which is why 15-02's headline test may use the
    equality form and why this one uses it against ``foo.bar``'s exact expected
    names rather than against the empty list.
    """
    survivor_payload = _PAYLOAD.copy() + 7.0
    assert "foo.bar" in ACCEPTED_KEYS, "precondition: 'foo.bar' is a legal store key under the Phase-14 rule"
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False)
    store.add_data_to_store("foo", _PAYLOAD.copy())
    store.add_data_to_store("foo.bar", survivor_payload)
    store.offload(pickle_container=True)

    foo_six = _six_artefact_paths(tmp_cache_dir, "foo")
    bar_six = _six_artefact_paths(tmp_cache_dir, "foo.bar")
    bar_present = {artefact: path for artefact, path in bar_six.items() if path.exists()}
    assert set(bar_present) == {"<key>.npy", "<key>.meta.json", "<key>.dat"}, (
        f"precondition: the adjacent key must own its three offloaded artefacts, saw {sorted(bar_present)}"
    )

    store.purge("foo")

    for artefact, path in foo_six.items():
        assert not path.exists(), f"SC-1 violated: foo's {artefact} survived its own purge at {str(path)!r}"
    for artefact, path in bar_present.items():
        assert path.exists(), (
            f"T-15-14 violated: purging 'foo' removed the adjacent key's {artefact} at {str(path)!r}. "
            "purge must reach only artefacts derived from the exact key it was given"
        )
    assert np.array_equal(np.asarray(store["foo.bar"]), survivor_payload), (
        "T-15-14 violated: the adjacent key no longer loads its own data after its neighbour was purged"
    )
    assert sorted(path.name for path in tmp_cache_dir.iterdir()) == sorted(
        path.name for path in bar_present.values()
    ), (
        "the directory must hold exactly the adjacent key's artefacts and nothing else — the equality is the "
        "secondary instrument that catches a seventh artefact no exact-name check enumerated"
    )


# ---------------------------------------------------------------------------
# STORE-08 encoding probe — whose definition of key equality applies.
# ---------------------------------------------------------------------------


def test_two_unicode_normalization_forms_are_two_distinct_keys(
    make_store: MakeStore, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Plan 15-03 / STORE-08 encoding probe / T-15-14b.

    **The contract this pins: key equality is byte equality of the key string as
    the filesystem stores it.** The package normalises nothing — Phase 14's rule
    refuses control characters, separators, absolute paths, reserved names and a
    trailing space or dot, and *nothing else*, which is asserted here rather than
    assumed by running the published predicate over both forms. So two keys
    differing only in Unicode normalization form are two legal, distinct keys
    with two distinct artefact sets, and purging one leaves the other whole.

    That is a statement of what **is**, not an endorsement. A caller who obtains
    one key from a filename read off disk and another from a string typed into a
    config can hold two keys that render identically and behave as two. Folding
    them would be a change to the *key contract* Phase 14 settled and is out of
    scope here; naming the consequence is what keeps it from being discovered
    downstream instead.

    The premise is filesystem-dependent, so it is **probed at runtime rather than
    guessed from the platform**: both names are created as empty files and the
    resulting entry count is counted. A normalizing filesystem (APFS, HFS+)
    folds them into one, and there the test skips with that reason — a recorded
    platform fact rather than a flaky assertion.
    """
    nfc = unicodedata.normalize("NFC", "café")
    nfd = unicodedata.normalize("NFD", "café")
    assert nfc != nfd, "precondition: the two normalization forms must be distinct Python strings"

    probe_dir = tmp_path / "normalization_probe"
    probe_dir.mkdir()
    (probe_dir / nfc).touch()
    (probe_dir / nfd).touch()
    if len(list(probe_dir.iterdir())) != 2:
        pytest.skip(
            "this filesystem normalizes Unicode filenames (APFS/HFS+ fold NFC and NFD into one name), so two "
            "keys differing only in normalization form cannot have two distinct artefact sets here"
        )

    assert paths_mod.is_valid_store_key(nfc), "the key rule must accept the NFC form — it normalizes nothing"
    assert paths_mod.is_valid_store_key(nfd), "the key rule must accept the NFD form — it normalizes nothing"

    nfd_payload = _PAYLOAD.copy() + 13.0
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False)
    store.add_data_to_store(nfc, _PAYLOAD.copy())
    store.add_data_to_store(nfd, nfd_payload)
    store.offload(pickle_container=True)
    nfd_present = {
        artefact: path for artefact, path in _six_artefact_paths(tmp_cache_dir, nfd).items() if path.exists()
    }
    assert set(nfd_present) == {"<key>.npy", "<key>.meta.json", "<key>.dat"}, (
        f"precondition: the NFD key must own its three offloaded artefacts, saw {sorted(nfd_present)}"
    )

    store.purge(nfc)

    _assert_all_six_gone(tmp_cache_dir, nfc)
    for artefact, path in nfd_present.items():
        assert path.exists(), (
            f"T-15-14b violated: purging the NFC key removed the NFD key's {artefact} at {str(path)!r}. "
            "The two are distinct keys because the filesystem stores them as distinct names"
        )
    assert np.array_equal(np.asarray(store[nfd]), nfd_payload), (
        "T-15-14b violated: the NFD key no longer loads its own data after the NFC key was purged"
    )
    assert nfd in store and nfc not in store, "only the purged key may have been dropped from tracking"


# ===========================================================================
# D-10 — the partial-unlink contract: attempt all, collect, raise once, and the
# key stays dropped.
#
# NOTHING BELOW ASSERTS AN UNCHANGED STORE OR AN UNCHANGED DIRECTORY. That is
# the D-10/SC-2 boundary restated at the point it would be easiest to cross:
# POSIX gives no atomicity across N unlinks, so a partial directory state is the
# *stated contract* here, not a discovered surprise. The properties that are
# asserted are the ones D-10 actually promises — which artefacts survived, that
# they are named in one raised aggregate, that the aggregate is an ``OSError``,
# that the key stays dropped, and that the order leaves a residue the reader
# already treats as a cache miss.
# ===========================================================================


def _fail_unlink_for(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    """Monkeypatch :meth:`pathlib.Path.unlink` to fail for ``target`` and delegate otherwise.

    The failure-injection template this suite already uses for the ENOSPC
    offload tests (``test_lazy_disk_cache.py:536-600``): capture the real
    callable first, gate the replacement on a predicate, delegate everything the
    predicate does not select. Gating on the exact file name rather than on a
    call count keeps the injection independent of the order under test, which
    matters because the order is itself asserted two tests below.
    """
    real_unlink = Path.unlink

    def failing_unlink(self: Path, missing_ok: bool = False) -> None:
        if self == target:
            raise OSError(errno.EACCES, "Permission denied")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", failing_unlink)


def test_a_failed_npy_unlink_raises_one_named_aggregate_and_the_key_stays_dropped(
    make_store: MakeStore, tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 15-03 / STORE-04 / D-10 / T-15-13.

    Three properties in one call, each of which was a decision:

    * **One aggregate, not the first exception.** Failures are collected rather
      than aborted on, so one unreadable artefact does not strand the other five.
    * **``StorePurgeIncompleteError`` subclasses ``OSError``**, asserted here
      rather than trusted from the class statement. A caller of a deleting
      operation already writes ``except OSError``; an ``ExceptionGroup`` would
      not be caught by it, so the migrating consumer's existing handler would
      silently stop working at the moment it was needed.
    * **The key stays dropped.** Re-tracking on failure was rejected because the
      re-tracked entry would point at a half-deleted artefact set, so
      ``store[key]`` afterwards could load garbage — and it would make ``purge``
      non-idempotent, which the idempotency test above depends on.

    The surviving ``<key>.npy`` is *inert*: ``_load_entry`` requires **both** the
    ``.npy`` and the ``.meta.json``, so nothing can re-adopt from it. That is why
    this case and the ``.dat`` case below are two different failures rather than
    one failure with two suffixes.
    """
    key = "npy_fails"
    store = _offloaded_store(make_store, tmp_cache_dir, key)
    six = _six_artefact_paths(tmp_cache_dir, key)
    npy = six["<key>.npy"]
    _fail_unlink_for(monkeypatch, npy)

    with pytest.raises(StorePurgeIncompleteError, match=re.escape(repr(str(npy)))) as excinfo:
        store.purge(key)

    assert isinstance(excinfo.value, OSError), (
        "StorePurgeIncompleteError must be catchable by a downstream `except OSError` — that is why it is not "
        "an ExceptionGroup"
    )
    assert key not in store.keys(), "D-10 violated: the key was re-tracked after a partial unlink failure"
    assert key not in store._store, "D-10 violated: the key survived in the backing dict after a failed unlink"
    assert npy.exists(), "the artefact whose unlink was made to fail must still be on disk"
    for artefact, path in six.items():
        if artefact == "<key>.npy":
            continue
        assert not path.exists(), (
            f"D-10 violated: {artefact} survived at {str(path)!r}. Failures are collected, not aborted on, so "
            "one unreadable artefact must not strand the other five"
        )


def test_a_failed_dat_unlink_leaves_a_stale_payload_that_a_re_added_key_never_serves(
    make_store: MakeStore, tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 15-03 / STORE-04 / D-10 / T-15-13b — review finding F4.

    Not the ``.npy`` case with a different suffix. The residues differ in kind:

    * a surviving ``<key>.npy`` is **store-owned and inert** — ``_load_entry``
      requires both halves of the codec pair, so the store cannot re-adopt from
      it;
    * a surviving ``<key>.dat`` is **entry-owned and survives on its own terms**,
      and sitting under a key nobody tracks any more it is precisely the
      ABA-sensitive residue this phase's finalizer detach exists to keep from
      being eaten.

    So the assertion this case adds — and the reason it is worth its own test —
    is the **re-add**: after the failed purge, a new entry under the same key
    must read back *its own* array and never the stale bytes the surviving
    ``.dat`` still holds.

    That last assertion is deliberately written on **the array a consumer reads
    back**, never on which ``np.memmap`` mode produced it, so it stays honest
    across 15-05's change to the ``.dat`` write route rather than pinning an
    implementation detail that plan rewrites.
    """
    key = "dat_fails"
    original = _PAYLOAD.copy()
    store = _offloaded_store(make_store, tmp_cache_dir, key)
    six = _six_artefact_paths(tmp_cache_dir, key)
    dat = six["<key>.dat"]
    assert np.array_equal(np.frombuffer(dat.read_bytes(), dtype=np.float32), original), (
        "precondition: the surviving .dat holds the FIRST key lifetime's bytes, which is what makes it stale"
    )
    _fail_unlink_for(monkeypatch, dat)

    with pytest.raises(StorePurgeIncompleteError, match=re.escape(repr(str(dat)))) as excinfo:
        store.purge(key)

    # (1) and (2): the aggregate names the survivor, and the key stays dropped.
    assert isinstance(excinfo.value, OSError)
    assert key not in store.keys(), (
        "D-10 violated: the key was re-tracked because its entry-owned payload survived. The key stays dropped "
        "even then — this is the half a reader is most likely to expect the opposite of"
    )
    assert key not in store._store, "D-10 violated: the key survived in the backing dict"
    # (3) everything whose unlink did not fail is gone; the payload is not.
    assert dat.exists(), "the .dat whose unlink was made to fail must still be on disk"
    for artefact, path in six.items():
        if artefact == "<key>.dat":
            continue
        assert not path.exists(), f"D-10 violated: {artefact} survived at {str(path)!r}"

    # (4) the point of covering this case separately: re-adding the same key.
    monkeypatch.undo()
    replacement = original + 50.0
    store.add_data_to_store(key, replacement)

    assert np.array_equal(np.asarray(store[key]), replacement), (
        "T-15-13b violated: a key re-added over a stale entry-owned .dat did not read back its own array"
    )
    assert not np.array_equal(np.asarray(store[key]), original), (
        "T-15-13b violated: the re-added key served the bytes the FAILED purge left behind. A surviving .dat "
        "under an untracked key must never become a later entry's data"
    )

    # Nothing from the first key's lifetime resurrects once the new entry is
    # collected and the directory is reopened by a fresh store.
    store.offload(pickle_container=True)
    del store[key]
    gc.collect()
    reopened = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False)

    assert np.array_equal(np.asarray(reopened[key]), replacement), (
        "T-15-13b violated: a fresh store over the same directory re-adopted the key and read back something "
        "other than the replacement array"
    )


def test_the_unlink_order_puts_every_sidecar_and_tmp_name_before_both_payloads(
    make_store: MakeStore, tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 15-03 / STORE-04 / D-10 / T-15-13 — the order, recorded rather than assumed.

    Why the order is the contract and not a preference. Unlinking ``<key>.npy``
    first and then failing on ``<key>.meta.json`` leaves a sidecar with no array:
    the reopen rescan globs ``*.npy`` and so will not rediscover the key, and
    ``_load_entry`` requires both halves and so refuses it — a file that is
    neither reachable nor collectable. The reverse order leaves *at most* a
    payload without its sidecar, which is a state the reader **already** treats
    as a cache miss. Since POSIX offers no atomicity across N unlinks, choosing
    which partial state is possible is the only control available.

    Asserted by **relative position**, not by equality against a six-element
    list, so adding a seventh artefact to the set does not falsify a test that
    is about ordering.
    """
    key = "ordered"
    store = _offloaded_store(make_store, tmp_cache_dir, key)
    six = _six_artefact_paths(tmp_cache_dir, key)
    recorded: list[str] = []
    real_unlink = Path.unlink

    def recording_unlink(self: Path, missing_ok: bool = False) -> None:
        recorded.append(self.name)
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", recording_unlink)
    recorded.clear()

    store.purge(key)

    monkeypatch.undo()
    for artefact, path in six.items():
        assert path.name in recorded, (
            f"purge did not attempt to unlink {artefact}; missing_ok=True means every one of the six is "
            f"attempted whether or not it is on disk. Recorded: {recorded}"
        )
    position = {artefact: recorded.index(path.name) for artefact, path in six.items()}
    for sidecar in ("<key>.meta.json", "<key>.meta.json.tmp", "<key>.npy.tmp", "<key>.dat.tmp"):
        for payload in ("<key>.npy", "<key>.dat"):
            assert position[sidecar] < position[payload], (
                f"D-10 violated: {sidecar} was unlinked after {payload} (positions {position[sidecar]} and "
                f"{position[payload]} in {recorded}). Sidecars and .tmp names go first so a partial failure "
                "leaves a state the reader already treats as a cache miss"
            )


def test_purging_a_tracked_key_with_no_artefacts_on_disk_raises_nothing(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-03 / STORE-04 / D-10 — ``missing_ok=True`` pinned end to end.

    A bare ``path.unlink()`` raises ``FileNotFoundError`` — itself an
    ``OSError`` — for every artefact that is not there, which for a tracked key
    that was never written to disk is *all six*. The purge would then collect six
    failures and raise ``StorePurgeIncompleteError`` for a key that is, in every
    sense the caller cares about, gone. This is the test that pins the flag
    rather than the comment that claims it.

    ``enable_caching=False`` is how the zero-artefact state is reached: with
    caching on, ``add_data_to_store`` materialises the ``.dat`` memmap eagerly,
    so even an un-offloaded key owns one file.
    """
    key = "nofiles"
    store = make_store(tmp_cache_dir, enable_caching=False, purge_disk_on_gc=False)
    store.add_data_to_store(key, _PAYLOAD.copy())
    assert key in store, "precondition: the key is tracked"
    assert sorted(path.name for path in tmp_cache_dir.iterdir()) == [], (
        "precondition: enable_caching=False writes no artefact, so the key owns nothing on disk"
    )

    store.purge(key)

    assert key not in store, "a purge of a tracked, artefact-less key must still drop the key"
    assert sorted(path.name for path in tmp_cache_dir.iterdir()) == [], "and must not have created anything"
