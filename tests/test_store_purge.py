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

# ``tests/`` is not a package, but pytest's prepend import mode puts it on
# ``sys.path``, so a sibling test module is importable by its plain name. The
# refusal matrix is imported rather than restated so this file and Phase 14's
# rule cannot drift into two disagreeing definitions of "illegal key".
from test_store_key_rules import REFUSED_KEYS

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
