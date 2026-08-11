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
import gc
import logging
import os
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from GSEGUtils.lazy_disk_cache import StoreKeyError, StorePurgeRefusedError
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
