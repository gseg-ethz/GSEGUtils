"""Plan 15-08 / gaps G-1 and G-2 / SC-1 and SC-3 / STORE-04.

This module pins the two verification gaps the phase shipped with: **G-1**, the ABA
hazard surviving on three of the four routes an entry can leave the mapping, and
**G-2**, ``purge`` reporting a complete removal while the key's payload is still in
the cache directory (the adopted symlink) or permanently orphaned outside it (a
foreign ``cache_path``). Every case still open at HEAD carries
``pytest.mark.xfail(strict=True)`` with a reason naming ``15-09`` -- the plan obliged
to close the defect and remove the marker -- and *strict* is the operative word: the
moment the defect closes, an unexpected pass turns this suite red until the marker
comes out, so no marker here can rot into a permanent excuse.

**Why the 821 tests that already existed could not see any of this** (§ WR-05,
independently confirmed by ``15-VERIFICATION.md`` § SC-3): every shipped
finalizer/ABA test binds the entry straight from the subscript -- ``store[key]`` --
immediately before purging, and that is the *single* route on which
``self._store.get(key)`` returns a live object. A net that never leaves that route cannot observe the defect no matter
how many assertions it carries. Hence the shape below -- the sequence is the test and
the **route is the only thing that varies**.

The three narrow tests this supersedes in *reach* (``test_store_purge.py``'s
``test_purge_detaches_the_live_entrys_finalizer_without_flipping_its_purge_intent``,
``test_a_purged_keys_stale_finalizer_cannot_eat_a_later_entry_under_the_same_key``
and ``test_the_aba_hazard_cannot_fire_under_a_forced_collection``) are not wrong and
are not deleted: they remain valid narrow assertions on the ``tracked`` route.

``pytest-randomly`` shuffles test order, so nothing here carries cross-test state.
"""

import copy
import gc
import os
import pickle
import weakref
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pytest

from GSEGUtils.lazy_disk_cache import paths as paths_mod
from GSEGUtils.lazy_disk_cache.disk_backed_ndarray import DiskBackedNDArray
from GSEGUtils.lazy_disk_cache.disk_backed_store import DiskBackedStore

#: A store factory as injected by ``conftest.make_store``. Annotated structurally
#: rather than by importing the conftest ``MakeStore`` protocol -- ``tests/`` is not
#: a package, so a relative import of ``conftest`` would not resolve
#: (``test_store_purge.py:53`` makes the same choice for the same reason).
MakeStore = Callable[..., DiskBackedStore[DiskBackedNDArray]]

#: The first and second arrays are deliberately **different**, so the read-back
#: assertion can tell "the later entry's data" from "the earlier entry's data". An
#: equal-payload version passes whichever file survived.
_FIRST = np.arange(4, dtype=np.float32)
_SECOND = np.arange(4, dtype=np.float32) + 100.0

#: Every open marker's reason names ``15-09``, so the plan obliged to remove them can
#: find all of them with one ``grep -c '15-09'``.
#:
#: ``_G1_OPEN`` is **gone**, with its three markers: ``15-09`` Task 1 closed G-1 on all
#: four routes in one step, because the registry that finds a key's live entries is
#: populated at *insertion* and cleared by no drop route -- so the route an entry
#: leaves the mapping by stopped mattering the moment the first registration site
#: landed. ``strict=True`` made that immediate rather than optional: the ``offload``
#: and ``pop`` parameters reported ``XPASS(strict)``, i.e. hard failures, until their
#: markers came out.
_G2A_OPEN = "G-2a open at 0956838 - closed by 15-09; remove this marker there"
_G2B_OPEN = "G-2b open at 0956838 - closed by 15-09; remove this marker there"

_POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="planting a <key>.dat symlink is POSIX-only")


# ---------------------------------------------------------------------------
# The SC-1 instrument. One helper, shared, prefix-safe.
# ---------------------------------------------------------------------------


def assert_nothing_derived_from(
    cache_dir: Path,
    key: str,
    *,
    expected_survivors: Iterable[str] = (),
    expected_files: Iterable[str] = (),
) -> None:
    """Assert SC-1's universal quantifier for ``key``: nothing derived from it is left.

    Two checks, and the second is the one that catches G-2.

    1. **None of the six builder-produced names exists.** Membership is derived
       through ``paths.get_{meta,meta_tmp,npy,npy_tmp,memmap,memmap_tmp}_path`` --
       the same seam ``purge`` builds through (D-14) -- so this test and the
       implementation cannot drift into two vocabularies. ``is_symlink()`` is checked
       alongside ``exists()`` because a *dangling* ``<key>.dat`` link is still
       something derived from the key, and ``exists()`` alone reports ``False`` for it.
    2. **Nothing unaccounted-for is left in the directory.** The listing minus the
       artefacts of every key in ``expected_survivors`` and every name in
       ``expected_files`` must be empty. This is what makes the helper able to see a
       *residual payload* -- a file whose name does not derive from the key but whose
       bytes are the key's data, which is exactly the adopted-symlink defect.

    **Why a prefix glob is the wrong instrument.** The published contract page
    (``docs/source/LazyDiskCache.rst:546-549``) warns that globbing the key followed by
    a dot and a star *"also matches a longer key's artefacts, so ``feat`` and
    ``feat2`` are not independent under a naive check."* Both naive forms fail, in
    different directions: a bare ``startswith(key)`` claims ``feat2.dat`` belongs to
    ``feat``, and the dotted glob claims ``feat.bar.dat`` does. The builder-derived set
    is immune to both, and
    ``test_the_listing_helper_does_not_confuse_a_prefix_neighbour`` is the standing
    control that stops a future edit from weakening this back into a glob.

    Parameters
    ----------
    cache_dir : pathlib.Path
        The store's cache directory, listed non-recursively.
    key : str
        The purged key. None of its six derived artefacts may remain.
    expected_survivors : Iterable[str], optional
        Other **store keys** whose artefacts are legitimately still present. Their six
        derived names each become an accounted-for file.
    expected_files : Iterable[str], optional
        Literal file **names** that are legitimately still present (a legacy
        ``<key>.pkl`` under D-09, say). Nothing is guessed: a helper that inferred
        which files were legitimate would hide the very defect this one exists to
        catch, so every survivor is declared by the caller.
    """
    derived = {
        "<key>.meta.json": paths_mod.get_meta_path(cache_dir, key),
        "<key>.meta.json.tmp": paths_mod.get_meta_tmp_path(cache_dir, key),
        "<key>.npy": paths_mod.get_npy_path(cache_dir, key),
        "<key>.npy.tmp": paths_mod.get_npy_tmp_path(cache_dir, key),
        "<key>.dat": paths_mod.get_memmap_path(cache_dir, key),
        "<key>.dat.tmp": paths_mod.get_memmap_tmp_path(cache_dir, key),
    }
    survived = sorted(name for name, path in derived.items() if path.exists() or path.is_symlink())
    assert not survived, (
        f"SC-1 violated for key {key!r}: {survived} still exist after purge. Every artefact whose name "
        f"derives from the key must be gone -- that is the rule the contract page states and the one a "
        f"directory listing is supposed to confirm"
    )

    accounted: set[str] = set()
    for survivor_key in expected_survivors:
        accounted.update(path.name for path in _derived_names(cache_dir, survivor_key))
    accounted.update(expected_files)
    unaccounted = sorted(path.name for path in cache_dir.iterdir() if path.name not in accounted)
    assert not unaccounted, (
        f"SC-1 violated for key {key!r}: the cache directory still holds {unaccounted}, which belongs to "
        f"neither a declared survivor key {sorted(expected_survivors)!r} nor a declared survivor file "
        f"{sorted(expected_files)!r}. A purge the method reported as complete left the key's data in the "
        f"directory under a name that does not derive from the key"
    )


def _derived_names(cache_dir: Path, key: str) -> tuple[Path, ...]:
    """Return the six builder-produced artefact paths for ``key``."""
    return (
        paths_mod.get_meta_path(cache_dir, key),
        paths_mod.get_meta_tmp_path(cache_dir, key),
        paths_mod.get_npy_path(cache_dir, key),
        paths_mod.get_npy_tmp_path(cache_dir, key),
        paths_mod.get_memmap_path(cache_dir, key),
        paths_mod.get_memmap_tmp_path(cache_dir, key),
    )


def _plant_adopted_symlink(cache_dir: Path, key: str, payload_name: str) -> Path:
    """Plant a real payload inside ``cache_dir`` and point ``<key>.dat`` at it.

    Planted **before the store exists**, which is what makes Phase 14's D-31
    resolved-containment check admit the shape, as it is meant to: the target is
    inside the cache directory, so the resolved final component is contained.
    """
    payload = cache_dir / payload_name
    payload.write_bytes(b"")
    link = paths_mod.get_memmap_path(cache_dir, key)
    link.symlink_to(payload)
    return payload


def _apply_route(store: DiskBackedStore[DiskBackedNDArray], key: str, route: str) -> None:
    """Drive ``key`` out of the mapping by ``route``, asserting the intermediate state.

    The intermediate assertion is not decoration. Without it a route that silently
    stopped producing its state -- an ``offload`` that no longer clears the reference,
    a ``pop`` that no longer deletes -- would degenerate into a fourth copy of
    ``tracked`` and pass for the wrong reason, which is the failure mode this whole
    module exists to have closed.
    """
    if route == "tracked":
        assert store.store[key] is not None, "route=tracked precondition: the entry stays tracked and live"
        return
    if route == "offload":
        store.offload(key, pickle_container=True)
        assert store.store[key] is None, (
            "route=offload precondition: offload(pickle_container=True) must clear the in-memory reference, "
            "leaving the mapping holding None -- that None is precisely why purge's detach is unreachable here"
        )
        return
    if route == "del":
        del store[key]
    elif route == "pop":
        store.pop(key)
    else:  # pragma: no cover - guards a typo in the parametrize list
        raise AssertionError(f"unknown route {route!r}")
    assert key not in store, f"route={route!r} precondition: the key must be untracked after the drop"


# 15-09 removed the three `_G1_OPEN` marks that stood here: the detach now reaches
# every route, so all four parameters are unmarked and green.
@pytest.mark.parametrize(
    "route",
    [
        pytest.param("tracked", id="tracked"),
        pytest.param("offload", id="offload"),
        pytest.param("del", id="del"),
        pytest.param("pop", id="pop"),
    ],
)
def test_the_aba_hazard_cannot_fire_on_any_route_out_of_the_mapping(
    make_store: MakeStore, tmp_cache_dir: Path, route: str
) -> None:
    """Plan 15-08 / G-1 / SC-3 / STORE-04 -- the whole hazard, on every route into ``purge``.

    Four routes reach ``purge`` with an entry that still holds a live
    ``weakref.finalize`` recorded against ``<key>.dat``. Until ``15-09``, ``purge``
    detached it by looking the entry up in the mapping and gating that lookup on the
    key being tracked -- and that lookup answered "nothing" on three of them: an
    **offloaded** entry is stored as ``None`` in a mapping typed
    ``dict[str, Optional[T]]``, and the **untracked orphan** left by ``del`` / ``pop``
    is exactly the state D-02 widened the *existence* check to reach
    (``tracked or any(p.exists())``). The existence check was widened and the detach
    was not -- that was the gap in one sentence.

    ``15-09`` closed it with a weak per-key registry of live entries that no drop
    route clears (D-15-G1), so the detach no longer asks the mapping. All four
    parameters are unmarked and green; ``tracked`` remains the positive control that
    proves this body is not vacuous, and it is the one route the three shipped tests
    in ``test_store_purge.py`` take.
    """
    key = "aba_route"
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)
    store.add_data_to_store(key, _FIRST.copy())

    # THE ENTRY AND ITS FINALIZER ARE REACHED HERE, THROUGH THE TRACKED MAPPING, AND
    # BEFORE THE ROUTE RUNS. Never through `store[key]` afterwards: on the `del` and
    # `pop` routes that subscript falls back to `_load_entry` and re-adopts the key
    # from disk, handing back a DIFFERENT object with a DIFFERENT finalizer -- and the
    # test would then assert liveness on the wrong one, passing for the wrong reason
    # on precisely the routes the defect lives on. That is the trap that produced
    # WR-05, and it is why three blockers survived 821 passing tests. Holding the
    # entry is legitimate *here* only because the subject of the assertion is the
    # finalizer's liveness rather than the detach path.
    original_entry = store.store[key]
    assert original_entry is not None, "precondition: add_data_to_store tracks a live entry"
    original_finalizer = original_entry._finalizer
    assert original_finalizer.alive, "precondition: a purge_disk_on_gc=True entry registers a finalizer"

    _apply_route(store, key, route)

    store.purge(key)

    assert original_finalizer.alive is False, (
        f"G-1 / SC-3 violated on route={route!r}: purge left the original entry's weakref.finalize ARMED "
        f"against <key>.dat. It will unlink whatever occupies that path at an arbitrary later collection -- "
        f"including a later entry created under the same key. The detach must reach every live entry the "
        f"key has had, through the weak registry, and not only the one the mapping still points at"
    )

    # Re-add under the same key with a DIFFERENT array. `add_data_to_store`
    # materialises the `.dat` eagerly, so the file a stale finalizer would eat is on
    # disk without an offload (an offloaded replacement is collected during the
    # offload and takes its own `.dat` right there, leaving nothing to eat).
    store.add_data_to_store(key, _SECOND.copy())
    dat = paths_mod.get_memmap_path(tmp_cache_dir, key)
    assert dat.exists(), "precondition: re-adding the key materialises a fresh <key>.dat"

    # An explicit del plus gc.collect(), never a scope exit: CPython's refcount would
    # usually collect at the del anyway, but "usually" is not an assertion and a
    # reference cycle would defer it.
    del original_entry
    gc.collect()

    assert dat.exists(), (
        f"G-1 / SC-3 violated on route={route!r} (ABA): forcing collection of the purged entry deleted the "
        f"LATER entry's <key>.dat. A weakref.finalize deletes whatever occupies its RECORDED PATH at "
        f"collection time, not the object it was registered for"
    )
    assert np.array_equal(np.asarray(store[key]), _SECOND), (
        f"G-1 / SC-3 violated on route={route!r}: <key>.dat exists but does not hold the replacement array. "
        f"A file that exists holding the wrong bytes is a different defect wearing the same green tick"
    )


def test_the_registry_does_not_keep_an_entry_alive(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-09 / D-15-G1 / STORE-05 axis 3 -- the registry is weak, measured.

    ``15-09`` gives the store a per-key registry of its live entries so ``purge`` can
    disarm a finalizer the mapping no longer points at. The prohibition attached to
    that decision is that it must **not** hold a strong reference: an entry the caller
    has dropped must still be collected, and its finalizer must still fire under the
    default ``purge_disk_on_gc=True``.

    A grep cannot establish this -- ``weakref.ref(`` appearing in the source says
    nothing about a strong reference smuggled in elsewhere -- so it is asserted
    behaviourally, on both halves: the weak reference dies, and the ``<key>.dat`` the
    finalizer was registered against is gone afterwards.
    """
    key = "weakly_held"
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)
    store.add_data_to_store(key, _FIRST.copy())

    entry = store.store[key]
    assert entry is not None, "precondition: add_data_to_store tracks a live entry"
    observer = weakref.ref(entry)
    dat = paths_mod.get_memmap_path(tmp_cache_dir, key)
    assert dat.exists(), "precondition: the entry materialised its <key>.dat"

    # Drop BOTH strong references -- the caller's and the mapping's. `pop` is
    # in-memory only and unlinks nothing (STORE-05), so whatever removes the file
    # below is the finalizer and not the drop route.
    store.pop(key, None)
    del entry
    gc.collect()

    assert observer() is None, (
        "D-15-G1 violated: the entry survived a forced collection after every caller-visible reference was "
        "dropped, so the store-side registry is holding a STRONG reference. The registry must find entries, "
        "never keep them"
    )
    assert not dat.exists(), (
        "STORE-05 axis 3 violated: the collected entry's finalizer did not unlink its <key>.dat under the "
        "default purge_disk_on_gc=True. A registry that merely DEFERRED the collection would show up exactly "
        "here, so this half is asserted as well as the weakref's death"
    )


def _assert_purge_detaches(store: DiskBackedStore[DiskBackedNDArray], key: str, entry: DiskBackedNDArray) -> None:
    """Drive the purge-then-ABA sequence on ``store`` and assert ``entry`` was disarmed.

    Factored out because three of the tests below differ only in *which store object*
    they drive -- the original, its unpickled twin, its shallow copy -- and the
    sequence they drive is the same one
    ``test_the_aba_hazard_cannot_fire_on_any_route_out_of_the_mapping`` established.
    """
    finalizer = entry._finalizer
    assert finalizer.alive, "precondition: a purge_disk_on_gc=True entry registers a finalizer"

    store.purge(key)

    assert finalizer.alive is False, (
        f"purge({key!r}) on this store left the entry's weakref.finalize ARMED against <key>.dat. The registry "
        f"did not survive whatever boundary this store crossed, so the detach found nothing to disarm"
    )


def test_a_pickled_store_still_detaches_on_purge(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-09 / D-15-G1 / STORE-05 concurrency -- the loky round-trip.

    Two properties, and the first is the one a forgotten ``__getstate__`` line breaks
    outright: ``pickle.dumps(store)`` must not raise. A ``weakref.ref`` is not
    picklable and the registry rides ``self.__dict__.copy()`` unless it is explicitly
    removed, so leaving it in would raise
    ``TypeError: cannot pickle 'weakref.ReferenceType' object`` -- and the joblib path
    force-offloads and pickles on **every** dispatch, making that the common failure
    rather than a rare one.

    The second is that the reconstructed store's ``purge`` still detaches. The
    reconstructed store reloads its entries from disk through ``_load_entry``, so it
    gets fresh registrations by the adoption route whether or not ``__setstate__``
    re-registers -- which is exactly why the copy test below is the one that pins the
    re-registration, and why this test cannot substitute for it.
    """
    key = "pickled"
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)
    store.add_data_to_store(key, _FIRST.copy())

    revived = pickle.loads(pickle.dumps(store))

    assert key in revived, "precondition: the round-trip must carry the key across"
    revived_entry = revived.store[key]
    assert revived_entry is not None, "precondition: the reconstructed store reloaded the entry from disk"
    _assert_purge_detaches(revived, key, revived_entry)
    assert_nothing_derived_from(tmp_cache_dir, key)


def test_a_copied_store_still_detaches_on_purge(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-09 / D-15-G1 -- the regression ``__setstate__``'s re-registration prevents.

    ``copy.copy(store)`` travels ``__reduce_ex__`` -> ``__getstate__`` ->
    ``__setstate__`` in-process, and ``__getstate__`` hands over the **same entry
    objects**. So a copy that came back with an empty registry would have a ``purge``
    that no longer detaches even on the ``tracked`` route -- the one route that worked
    *before* ``15-09``. That makes the re-registration a guard against a silent
    regression of existing behaviour, not a nicety attached to a new feature, and this
    test is what says so: delete the re-registration and this goes red while
    ``test_a_pickled_store_still_detaches_on_purge`` stays green.

    The **original** entry is held and asserted on, deliberately: it is the object the
    copy shares, and it is the one whose finalizer would outlive the copy's purge.

    ``enable_caching=False`` is **required**, not incidental, and the reason is a
    correction to the premise as ``15-09`` stated it: ``__getstate__`` opens with
    ``if self._enable_caching: self.offload(pickle_container=True)``, so with caching
    **on** every entry is offloaded to ``None`` before the snapshot is taken and
    ``__setstate__`` reloads a genuinely *different* object from disk. The
    same-object shape the re-registration exists for is therefore reachable only on
    the caching-off configuration -- which is also the one where ``__setstate__``
    skips its reload loop entirely, leaving the restored mapping as the only source
    of registrations. ``purge_disk_on_gc=True`` still arms a finalizer here: it is
    gated on ``_cache_path`` and the flag, never on ``enable_caching``.
    """
    key = "copied"
    store = make_store(tmp_cache_dir, enable_caching=False, purge_disk_on_gc=True)
    store.add_data_to_store(key, _FIRST.copy())
    original_entry = store.store[key]
    assert original_entry is not None, "precondition: add_data_to_store tracks a live entry"

    twin = copy.copy(store)

    assert twin.store[key] is original_entry, (
        "precondition: __getstate__ hands over the same entry OBJECT, which is what makes an empty registry "
        "on the copy a regression rather than a missing feature"
    )
    _assert_purge_detaches(twin, key, original_entry)


def test_a_store_reopened_over_an_existing_directory_purges_a_key_it_never_registered(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-09 / D-15-G1 / D-02 -- the reopen rescan is not a registration route.

    The rescan in ``__init__`` installs ``None``, not an entry, so a store reopened
    over an existing cache directory carries an **empty** registry by construction.
    That must not make the key unpurgeable: ``purge`` reaches it through the
    untracked-orphan path D-02 widened the existence check for, and the detach loop
    simply finds nothing to disarm -- which is correct, because there is no live entry
    in this process whose finalizer could fire.
    """
    key = "reopened"
    first = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False)
    first.add_data_to_store(key, _FIRST.copy())
    first.offload(pickle_container=True)
    del first
    gc.collect()

    reopened = make_store(tmp_cache_dir, enable_caching=False, purge_disk_on_gc=False)
    assert key in reopened, "precondition: the rescan re-adopts the key from its codec pair"
    assert reopened._entry_registry == {}, (
        "precondition: the rescan installs None rather than an entry, so it registers nothing -- if this ever "
        "becomes non-empty the comment at the rescan's `self._store[key] = None` has become wrong"
    )

    reopened.purge(key)

    assert_nothing_derived_from(tmp_cache_dir, key)


# ---------------------------------------------------------------------------
# G-2a — the adopted symlink, reachable through the library's own write route.
# ---------------------------------------------------------------------------


def test_the_listing_helper_does_not_confuse_a_prefix_neighbour(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-08 / SC-1 -- the helper's own positive control.

    Purge ``feat`` in a directory that also holds the artefacts of two neighbours a
    naive check gets wrong in *different* directions: ``feat2`` defeats a bare
    ``startswith(key)``, and the dotted key ``feat.bar`` defeats a glob of the key
    followed by a dot and a star. Both are legal store keys (``test_store_key_rules.py``
    accepts ``foo.bar``). The helper must pass here -- if a future edit weakens it
    into either naive form, this test is what goes red.
    """
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)
    store.add_data_to_store("feat", _FIRST.copy())
    store.add_data_to_store("feat2", _FIRST.copy())
    store.add_data_to_store("feat.bar", _FIRST.copy())
    for neighbour in ("feat2", "feat.bar"):
        assert paths_mod.get_memmap_path(tmp_cache_dir, neighbour).exists(), (
            f"precondition: neighbour {neighbour!r} must have a materialised <key>.dat to be confusable at all"
        )

    store.purge("feat")

    assert_nothing_derived_from(tmp_cache_dir, "feat", expected_survivors=("feat2", "feat.bar"))
    for neighbour in ("feat2", "feat.bar"):
        assert paths_mod.get_memmap_path(tmp_cache_dir, neighbour).exists(), (
            f"purging 'feat' removed the prefix-adjacent neighbour {neighbour!r}'s <key>.dat"
        )


@_POSIX_ONLY
def test_an_adopted_symlink_dat_is_written_through_not_replaced(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-08 / D-17 / STORE-03 -- 15-05's adoption behaviour, held as a control.

    ``lazy_disk_cache.py:508-524`` renames onto ``self._cache_path.resolve()`` rather
    than onto the bare name, precisely so an adopted ``<key>.dat`` keeps being a link
    and the caller's payload keeps receiving data. This test asserts only that.

    It exists so that a "fix" for G-2a which made ``purge`` pass by breaking adoption
    at *write* time -- replacing the link with a regular file, orphaning the caller's
    payload silently -- would show up as a regression rather than as progress.
    """
    key = "adopted_control"
    payload = _plant_adopted_symlink(tmp_cache_dir, key, "innocuous_control.bin")
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)

    store.add_data_to_store(key, _FIRST.copy())

    link = paths_mod.get_memmap_path(tmp_cache_dir, key)
    assert link.is_symlink(), (
        "adoption broken: the library replaced the adopted <key>.dat symlink with a regular file, so the "
        "payload the caller adopted has silently stopped receiving data (D-17 / STORE-03)"
    )
    assert np.array_equal(np.fromfile(payload, dtype=np.float32), _FIRST), (
        f"adoption broken: the payload behind the adopted link does not hold the key's data. The library "
        f"must write THROUGH the link -- read back {np.fromfile(payload, dtype=np.float32)!r}"
    )


@_POSIX_ONLY
@pytest.mark.xfail(strict=True, reason=_G2A_OPEN)
def test_purge_removes_the_payload_behind_an_adopted_symlink_dat(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-08 / G-2a / SC-1 / STORE-04 -- the shape the library itself calls first-class.

    ``lazy_disk_cache.py:510-513`` names it: *"An ``<key>.dat`` that is a symlink
    pointing at a real payload file inside the cache directory is the legitimate
    **adopted entry** that D-17 and STORE-03 exist to protect."* The library then
    writes the key's data through it.

    ``purge`` builds six names and calls ``artefact.unlink(missing_ok=True)``
    (``disk_backed_store.py:1188``), which removes the **link** and never the resolved
    target. So it reports a complete removal while the key's exact payload is still
    sitting in the cache directory -- and SC-1's binding verification clause, *"verified
    by listing the directory afterwards"*, reads ``['innocuous.bin']``.

    Reconstructs the verifier's ``sc1_reachable.py``.
    """
    key = "adopted"
    payload = _plant_adopted_symlink(tmp_cache_dir, key, "innocuous.bin")
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)
    store.add_data_to_store(key, _FIRST.copy())

    # Assert the SHAPE before asserting the defect: a test that quietly failed to set
    # up an adopted entry would pass by testing the ordinary path instead.
    link = paths_mod.get_memmap_path(tmp_cache_dir, key)
    assert link.is_symlink(), "precondition: <key>.dat must still be a symlink after the library's own write"
    assert np.array_equal(np.fromfile(payload, dtype=np.float32), _FIRST), (
        "precondition: the library must have written the key's data through the link into the payload"
    )

    store.purge(key)  # returns cleanly today; that is half the defect, so it is recorded, not routed around

    assert_nothing_derived_from(tmp_cache_dir, key)
    assert not payload.exists(), (
        f"G-2a / SC-1 violated: purge unlinked the LINK and left the key's payload at {payload.name!r} "
        f"inside the cache directory, holding the key's exact bytes. purge must resolve <key>.dat and "
        f"remove what it points at when the target is inside the cache directory"
    )


# ---------------------------------------------------------------------------
# G-2b — a foreign `_cache_path`, entirely inside the public API.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=_G2B_OPEN)
def test_purge_does_not_orphan_a_foreign_cache_path_entry(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-08 / G-2b / SC-1 / STORE-04 -- the leak ``purge`` creates itself.

    An entry constructed with a ``cache_path`` outside the store's cache directory and
    inserted with ``store[key] = entry`` writes its ``.dat`` out there.
    ``disk_backed_store.py:1096-1106`` builds six paths from
    ``(self._cache_dir, key)`` and nothing in ``purge`` ever reads the entry's own
    ``_cache_path``, so the method raises nothing -- not even
    ``StorePurgeIncompleteError``, because the six names it built never existed --
    drops the key, and **detaches the finalizer that was the only thing that would
    ever have removed the file**. A file GC would have cleaned becomes permanently
    orphaned, by the verb whose whole purpose is removal. Reproduces the verifier's
    ``sc1_repro.py``.

    **The path is set at construction, not by assignment**: the ``cache_path`` setter
    is sealed (``lazy_disk_cache.py:792-812``, D-01) and assigning to it raises, so a
    test that reached for the setter would fail with the seal's own error and prove
    nothing about ``purge``.

    **Deliberately no assertion on an exception type.** ``15-09`` introduces a
    dedicated refusal error class for this shape; naming that symbol from a wave-7
    test would raise ``AttributeError`` at *import* and turn a strict-xfail into a
    collection error -- reported as an error rather than an ``xfail``, and it would
    take the whole module down with it. So this test pins the three invariants that
    need no new type name, and ``15-09`` Task 3 adds the typed companion once the
    symbol exists. The split is *behaviour pinned first, contract pinned second*, and
    it is deliberate rather than an oversight.
    """
    key = "foreign"
    elsewhere = tmp_cache_dir.parent / "elsewhere"
    elsewhere.mkdir()
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)

    entry = DiskBackedNDArray(
        _FIRST.copy(),
        enable_caching=True,
        cache_path=elsewhere / key,
        purge_disk_on_gc=True,
    )
    store[key] = entry

    foreign_dat = elsewhere / f"{key}{paths_mod.MEMMAP_SUFFIX}"
    assert foreign_dat.exists(), "precondition: the entry must have written its .dat outside the cache directory"
    assert foreign_dat.stat().st_size > 0, "precondition: the foreign .dat must hold real bytes"
    assert_nothing_derived_from(tmp_cache_dir, key)

    # The subject is the invariants, not the control flow: today purge returns
    # cleanly, in 15-09 it is expected to raise, and this test must pass in both.
    observed: str = "no exception"
    try:
        store.purge(key)
    except Exception as exc:  # noqa: BLE001 - the type is recorded, never asserted on
        observed = type(exc).__name__

    assert entry._finalizer.alive is True, (
        f"G-2b violated (purge raised: {observed}): purge detached the finalizer for a file it did not "
        f"remove. That finalizer was the ONLY thing that would ever have unlinked the foreign .dat, so "
        f"purge has converted a GC-cleanable file into a permanent leak it created itself"
    )
    assert key in store, (
        f"G-2b violated (purge raised: {observed}): purge dropped the key while removing none of its data. "
        f"A refusal must leave the key tracked -- SC-2's no-op property, extended to this refusal"
    )

    # Drop BOTH references, and drop the store's through `pop`, which is in-memory
    # only and unlinks nothing (STORE-05). Written world-agnostically on purpose: at
    # HEAD the key is already untracked here, and after 15-09 the store still holds
    # it, so a bare `del entry` would not be enough to collect the entry there.
    store.pop(key, None)
    del entry
    gc.collect()

    assert not foreign_dat.exists(), (
        f"G-2b violated (purge raised: {observed}): after dropping the entry and forcing a collection the "
        f"foreign .dat is STILL on disk -- a permanent leak. purge must not disarm the only cleanup that "
        f"exists for a file it declined to remove"
    )


# ---------------------------------------------------------------------------
# STORE-04 / idempotency and STORE-05 / boundary — the controls, green today.
# ---------------------------------------------------------------------------


def test_purging_twice_is_safe_and_the_second_call_raises_key_error(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-08 / STORE-04 / D-02 -- the idempotency probe.

    ``purge`` is idempotent *in effect*: the second call changes nothing on disk. It
    is not idempotent *in signalling*, and that is D-02's deliberate contract -- a key
    present neither in tracking nor on disk raises ``KeyError``, because a silent
    no-op on a typo'd key would make the removal verbs disagree with ``__delitem__``
    and ``pop`` about missing keys.

    The prefix-adjacent neighbour is added **between** the two purges, so the repeat
    is the only thing that could have touched it. That is the assertion a naive
    prefix-based implementation of ``purge`` would fail.
    """
    key = "feat"
    neighbour = "feat2"
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False)
    store.add_data_to_store(key, _FIRST.copy())

    store.purge(key)
    assert_nothing_derived_from(tmp_cache_dir, key)

    store.add_data_to_store(neighbour, _SECOND.copy())
    neighbour_dat = paths_mod.get_memmap_path(tmp_cache_dir, neighbour)
    assert neighbour_dat.exists(), "precondition: the neighbour must be on disk before the repeat purge"

    with pytest.raises(KeyError):
        store.purge(key)

    assert neighbour_dat.exists(), (
        "the repeat purge of 'feat' removed the prefix-adjacent neighbour 'feat2'.dat. The six names are "
        "built from the exact key (D-14), so no neighbour can ever be in the set"
    )
    assert np.array_equal(np.asarray(store[neighbour]), _SECOND), (
        "the repeat purge left the neighbour's file in place but corrupted its contents"
    )
    assert_nothing_derived_from(tmp_cache_dir, key, expected_survivors=(neighbour,))


def test_the_three_axes_are_still_in_memory_only_beside_this_net(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-08 / STORE-05 -- the boundary probe, sited here on purpose.

    ``del``, ``pop`` and ``clear`` each drop tracking and unlink **nothing**;
    ``purge`` is the only removal that reaches the filesystem. ``STORE-05`` requires
    that *where the data lives*, *whether the key is tracked* and *whether the file
    outlives the object* stay three separate axes.

    ``tests/test_store_lifecycle_axes.py`` already owns this property and remains its
    canonical home -- this is a **deliberate second assertion**, not a duplicate that
    escaped review. It is sited in the module that is about to teach ``purge`` to
    reach further (``15-09`` makes it resolve symlinks and consult a registry), so a
    change that accidentally made a *drop* route durable fails right here, next to the
    change that caused it, rather than in a file nobody was looking at.

    ``purge_disk_on_gc=False`` throughout: with it ``True`` the entries' own
    finalizers would unlink the ``.dat`` files at the collection that follows each
    drop, and the test would be measuring the GC axis instead of the tracking axis.
    """
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False)
    keys = ("axis_del", "axis_pop", "axis_clear")
    for key in keys:
        store.add_data_to_store(key, _FIRST.copy())
    dats = {key: paths_mod.get_memmap_path(tmp_cache_dir, key) for key in keys}
    for key, dat in dats.items():
        assert dat.exists(), f"precondition: {key!r} must have a materialised <key>.dat"

    del store["axis_del"]
    gc.collect()
    assert dats["axis_del"].exists(), (
        "STORE-05 violated: `del store[key]` unlinked the key's <key>.dat. It drops tracking and must "
        "unlink nothing -- purge is the durable counterpart, one method away"
    )

    store.pop("axis_pop")
    gc.collect()
    assert dats["axis_pop"].exists(), (
        "STORE-05 violated: `store.pop(key)` unlinked the key's <key>.dat. pop routes through "
        "__delitem__ and is in-memory only"
    )

    store.clear()
    gc.collect()
    assert dats["axis_clear"].exists(), (
        "STORE-05 violated: `store.clear()` unlinked a key's <key>.dat. clear routes through "
        "__delitem__ for every key and is in-memory only"
    )
    for key, dat in dats.items():
        assert dat.exists(), f"STORE-05 violated: {key!r}'s <key>.dat did not survive the three drop routes"
