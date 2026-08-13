"""Plan 15-11 / round-3 criticals CR2-01..CR2-04 / D-15-G5, D-15-G6, D-15-G7 / STORE-04.

**The rule this whole module is about: the set ``purge`` disarms is the set ``purge``
deleted.** Cache-directory membership is the *authorization* question -- may I delete
this? -- and it is not the *bookkeeping* question -- did I delete this? What the store
does not own it does not touch, and what it did not remove it does not disarm. Both
halves are load-bearing in opposite directions: disarming a finalizer for a file the
method declined to remove destroys that file's only cleanup, and following a link to a
file the key does not own destroys somebody else's data.

``15-11`` opens the cases; ``15-12`` is obliged to close four of them and ``15-13`` the
fifth. Every open case carries ``pytest.mark.xfail(strict=True)`` with a reason naming
the plan that must remove it, so a marker cannot rot into a permanent excuse: the moment
the defect closes, the unexpected pass turns this suite red until the marker comes out.

**Two facts make this module necessary rather than tidy.**

1. All fourteen round-2 findings -- four of them critical, three of those reproduced as
   data destruction or as a permanent leak manufactured by the removal verb itself --
   live under a suite that reports **844 passed, 0 xfailed**. A green suite bounds only
   what its tests reach.
2. ``test_store_purge_identity.py``'s own net, which was written for exactly this class
   of defect one round ago, cannot see a single one of these shapes -- because **every
   one of its cases pairs a key with its own** ``<key>.dat``. An entry whose backing
   file is in the cache directory under some *other* name, one entry registered under
   *two* keys, an entry with no configured ``cache_path`` at all, a ``.dat`` with no
   key that owns it, and a link aimed at *another key's* artefact are five states that
   net has no case for. Its instrument
   (:func:`assert_nothing_derived_from`) is nonetheless exactly right, so it is imported
   rather than copied -- one vocabulary, no drift.

The markers, as this round assigns them:

* **D-15-G5** (``DESIGN-DECISIONS.md`` entry 82) -- the ``mkstemp`` backing file is
  ephemeral **by policy**, so an ordinarily-constructed entry must not veto the removal
  of the store-owned artefacts that are unambiguously the store's to delete.
* **D-15-G6** (entry 83) -- the detach is gated on what this call deleted, never on
  territory; and the reopen rescan globs ``.dat`` as well as ``.npy``.
* **D-15-G7** (``15-13``) -- ``purge`` refuses to follow a link whose resolved target is
  another key's store artefact.

``pytest-randomly`` shuffles test order, so nothing here carries cross-test state.
"""

import copy
import gc
import os
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

# The one prefix-safe SC-1 listing instrument, imported from the module that owns it
# (`15-08`, recorded in `15-08-SUMMARY.md` as "called by 15-09 and 15-10"). `tests/` is
# not a package, so this resolves through pytest's prepend import mode and through
# mypy's `explicit_package_bases` -- both measured before the choice was made. A local
# copy would be a second vocabulary for "nothing derived from this key survives", which
# is the exact drift D-14 removed from the implementation side. Ruff's isort rules sort
# it into the third-party block, above `GSEGUtils`, which is where a name it cannot
# classify as first-party belongs; the placement is the formatter's, not a choice.
from test_store_purge_identity import assert_nothing_derived_from

from GSEGUtils.lazy_disk_cache import StorePurgeForeignArtefactError
from GSEGUtils.lazy_disk_cache import paths as paths_mod
from GSEGUtils.lazy_disk_cache.disk_backed_ndarray import DiskBackedNDArray
from GSEGUtils.lazy_disk_cache.disk_backed_store import DiskBackedStore
from GSEGUtils.lazy_disk_cache.lazy_disk_cache import LazyDiskCacheConfig

#: A store factory as injected by ``conftest.make_store``. Annotated structurally rather
#: than by importing the conftest ``MakeStore`` protocol -- the sibling purge modules
#: make the same choice for the same reason.
MakeStore = Callable[..., DiskBackedStore[DiskBackedNDArray]]

#: Distinct payloads, so a read-back assertion can tell one key's data from another's.
_FIRST = np.arange(4, dtype=np.float32)
_SECOND = np.arange(4, dtype=np.float32) + 100.0

#: The shared reason prefix for every case that is open at this plan's HEAD, so the
#: closing plans can find them all with one grep -- the device `15-08` used and the
#: reason `15-09`'s five removals were mechanically checkable. The closing plan is named
#: on the decorator line itself rather than in this constant, because the marker counts
#: in `15-12` and `15-13` read a bounded window around the decorator and a reason spread
#: across four lines would undercount.
_OPEN = "round-3 ownership defect, open at 0a2dab2"

_POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="planting a <key>.dat symlink is POSIX-only")


# ---------------------------------------------------------------------------
# CR2-01 — the leak `purge` still creates, with `elsewhere/` swapped for a name
# inside the cache directory.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=f"{_OPEN} (CR2-01) - closed by 15-12; remove this marker there")
def test_purge_does_not_disarm_the_finalizer_for_an_in_cache_file_it_does_not_remove(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-11 / CR2-01 / D-15-G6 / SC-1 / STORE-04 -- G-2b surviving inside the cache dir.

    ``15-09`` closed *"the leak purge creates itself"* for the half where the entry's
    backing file lands **outside** the cache directory, and left it open for the half
    where it lands **inside** it. ``_reconcile_artefact_targets`` computes each live
    registered entry's resolved ``_cache_path`` into what its own docstring calls "the
    key's effective artefact set" -- and uses it for the **refusal only**. It never
    reaches ``_unlink_artefacts``. The detach loop, by contrast, is unconditional over
    the registered entries. So for a contained path that is not one of the six built
    names, ``purge`` does not refuse, does not unlink, **does** detach, drops the key,
    and returns cleanly reporting a complete removal.

    The measured transcript at ``0a2dab2``, which this case reconstructs::

        entry cache_path      : .../cache/other.dat
        purge -> returned cleanly (no exception)
        finalizer alive after : False
        after gc, other.dat   : True    <-- PERMANENT LEAK

    **The shape is asserted before the defect is.** A case that quietly failed to give
    the entry a contained, non-derived backing file would pass by exercising the
    ordinary path instead.

    **The path is fixed at construction, never by assignment**: the ``cache_path``
    setter is sealed (D-01), so a case that reached for the setter would fail with the
    seal's own error and prove nothing about ``purge``.
    """
    key = "feat"
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)

    # `_init_from_config` re-suffixes the supplied path, so `.../cache/other` becomes
    # `.../cache/other.dat`: inside the cache directory, and not one of `feat`'s six
    # built names. That combination is the whole finding.
    entry = DiskBackedNDArray(
        _FIRST.copy(),
        enable_caching=True,
        cache_path=tmp_cache_dir / "other",
        purge_disk_on_gc=True,
    )
    store[key] = entry

    other = tmp_cache_dir / f"other{paths_mod.MEMMAP_SUFFIX}"
    assert other.exists(), "precondition: the entry must have written its backing file inside the cache directory"
    assert other.stat().st_size > 0, "precondition: the in-cache backing file must hold real bytes"
    assert entry._finalizer.alive, "precondition: a purge_disk_on_gc=True entry registers a finalizer"
    assert_nothing_derived_from(tmp_cache_dir, key, expected_files=(other.name,))

    # The subject is the invariants, not the control flow: at HEAD purge returns
    # cleanly, and D-15-G6 keeps it returning cleanly while changing what it disarms.
    # The observed name is recorded and quoted, never asserted on.
    observed: str = "no exception"
    try:
        store.purge(key)
    except Exception as exc:  # noqa: BLE001 - the type is recorded, never asserted on
        observed = type(exc).__name__

    assert entry._finalizer.alive is True, (
        f"CR2-01 / D-15-G6 violated (purge raised: {observed}): purge detached the finalizer for a file it did "
        f"not remove. That finalizer was the ONLY cleanup {other.name!r} had, so purge has converted a "
        f"GC-reclaimable file into a permanent leak it created itself -- the same defect G-2b names, with "
        f"'elsewhere/' swapped for a name inside the cache directory. The detach must be gated on what this "
        f"call DELETED, not on whose territory the file sits in"
    )
    assert_nothing_derived_from(tmp_cache_dir, key, expected_files=(other.name,))

    # Drop BOTH references and force a collection. `pop` is in-memory only and unlinks
    # nothing (STORE-05), so whatever removes the file below is the finalizer.
    store.pop(key, None)
    del entry
    gc.collect()

    assert not other.exists(), (
        f"CR2-01 / D-15-G6 violated (purge raised: {observed}): after dropping the entry and forcing a "
        f"collection {other.name!r} is STILL inside the cache directory -- a permanent leak, and a directory "
        f"listing after a purge the method reported as complete that is not empty. SC-1's binding clause is "
        f"verified by listing the directory afterwards, and it reads [{other.name!r}]"
    )


def test_the_ordinary_tracked_key_purge_still_detaches_and_removes_everything(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-11 / D-15-G6 / STORE-04 -- the CONTROL that forbids a fix-by-never-detaching.

    Green at this plan's HEAD, and it must stay green. ``15-12`` gates the detach on the
    removal set; the degenerate way to satisfy the case above is to gate it on nothing
    and detach never, which would silently re-open the ABA hazard ``15-09`` closed on
    all four routes. This case is what goes red if that is what lands.

    The two cases only say the rule **together**: *disarm exactly what you deleted.*
    Neither says it alone -- the one above forbids disarming what was not deleted, this
    one requires disarming what was.
    """
    key = "ordinary"
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)
    store.add_data_to_store(key, _FIRST.copy())

    # Bound through the tracked mapping BEFORE the purge, never through the subscript
    # afterwards: on a dropped key that subscript re-adopts from disk and hands back a
    # different object with a different finalizer (WR-05, the trap `15-08` recorded).
    entry = store.store[key]
    assert entry is not None, "precondition: add_data_to_store tracks a live entry"
    finalizer = entry._finalizer
    assert finalizer.alive, "precondition: a purge_disk_on_gc=True entry registers a finalizer"
    assert paths_mod.get_memmap_path(tmp_cache_dir, key).exists(), (
        "precondition: add_data_to_store materialises the key's <key>.dat eagerly"
    )

    store.purge(key)

    assert finalizer.alive is False, (
        "D-15-G6 violated in the other direction: purge removed the key's <key>.dat and left its finalizer "
        "ARMED. A stale weakref.finalize deletes whatever occupies its recorded path at an arbitrary later "
        "collection -- including a later entry created under the same key (the ABA hazard, G-1/SC-3). Gating "
        "the detach on the removal set must NARROW it, never disable it"
    )
    assert_nothing_derived_from(tmp_cache_dir, key)


# ---------------------------------------------------------------------------
# CR2-02 — one entry, two keys, and the damage lands on the key nobody named.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=f"{_OPEN} (CR2-02) - closed by 15-12; remove this marker there")
def test_purging_one_key_leaves_a_shared_entrys_finalizer_armed_for_the_other_key(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-11 / CR2-02 / D-15-G6 / STORE-04 -- the cross-key detach.

    Nothing prevents one entry object from being registered under two keys:
    ``store["alpha"] = e; store["beta"] = e`` is **two ordinary validated writes**,
    both of which ``__setitem__``'s own docstring blesses as a write route. No private
    seam is touched anywhere in this case, and that is the point -- the shape is
    reachable by an ordinary consumer.

    ``purge("beta")`` then walks the entries registered under ``beta`` and detaches the
    shared entry's finalizer, while that entry is still the live, tracked entry of
    ``alpha`` and ``alpha.dat`` was never in ``beta``'s artefact set. Measured at
    ``0a2dab2``::

        alpha.dat before      : True   finalizer: True
        purge('beta') ->  finalizer alive: False   alpha still tracked: True   alpha.dat: True
        after gc alpha.dat    : True   <-- alpha's live entry lost its only cleanup

    This is the counterexample entry 83 uses to settle *territory* against *deletion*:
    ``alpha.dat`` is inside the cache directory, so territory says **disarm**; it was
    never in ``beta``'s artefact set, so deletion says **don't**; and deletion is
    correct. The two rules converge for the single-key case, which is exactly why the
    territory version would look like it worked.

    It is **distinct from deferred item D-15-07**, which is about a caller left holding
    a half-live entry over an inode ``purge`` *did* unlink. Here the file was never
    unlinked, the entry is still a first-class tracked member of the store, and the
    damage lands on a key the caller never named.
    """
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)
    store.add_data_to_store("alpha", _FIRST.copy())

    entry = store.store["alpha"]
    assert entry is not None, "precondition: add_data_to_store tracks a live entry"
    store["beta"] = entry

    alpha_dat = paths_mod.get_memmap_path(tmp_cache_dir, "alpha")
    assert alpha_dat.exists(), "precondition: alpha's <key>.dat must be materialised to be losable at all"
    assert entry._finalizer.alive, "precondition: a purge_disk_on_gc=True entry registers a finalizer"
    assert "alpha" in store and "beta" in store, (
        "precondition: both ordinary __setitem__ writes must leave their key tracked -- the shape is one "
        "entry object registered under two keys, and it is unreachable if either write did not land"
    )

    observed: str = "no exception"
    try:
        store.purge("beta")
    except Exception as exc:  # noqa: BLE001 - the type is recorded, never asserted on
        observed = type(exc).__name__

    assert entry._finalizer.alive is True, (
        f"CR2-02 / D-15-G6 violated (purge raised: {observed}): purging 'beta' disarmed the cleanup of an "
        f"entry that is still the live, tracked entry of 'alpha'. purge('beta') removed no file belonging "
        f"to 'alpha' and had no business touching its finalizer -- the damage lands on a key the caller "
        f"never named, which is what makes this a data-loss shape rather than a curiosity"
    )
    assert "alpha" in store, (
        f"CR2-02 violated (purge raised: {observed}): purging 'beta' dropped 'alpha' from tracking. Only the "
        f"named key may be dropped"
    )
    assert alpha_dat.exists(), (
        f"CR2-02 violated (purge raised: {observed}): purging 'beta' unlinked alpha's <key>.dat. The six "
        f"names purge builds come from the EXACT key (D-14), so no other key's artefact can ever be in the "
        f"removal set"
    )

    # The half that makes it a leak rather than a curiosity: drop every reference and
    # force a collection. `pop` is in-memory only and unlinks nothing (STORE-05), so
    # whatever removes the file below is the surviving finalizer.
    store.pop("alpha", None)
    store.pop("beta", None)
    del entry
    gc.collect()

    assert not alpha_dat.exists(), (
        f"CR2-02 violated (purge raised: {observed}): after dropping every reference and forcing a "
        f"collection alpha's <key>.dat is STILL on disk. purge('beta') destroyed the only cleanup a file "
        f"belonging to a DIFFERENT key had, so a key the caller never named now leaks permanently"
    )


# ---------------------------------------------------------------------------
# CR2-04 — an ordinarily-constructed entry, and a refusal decided by the GC.
# ---------------------------------------------------------------------------


def _purge_an_ordinarily_constructed_key(
    store: DiskBackedStore[DiskBackedNDArray], cache_dir: Path, key: str, *, hold_reference: bool
) -> tuple[str, list[str]]:
    """Run the CR2-04 sequence for ``key``; return the outcome name and the survivors.

    The sequence is identical on both calls and **only the caller's liveness varies**,
    which is what makes the two results comparable at all: if the two disagree, the
    difference is the garbage collector and nothing else.

    ``enable_caching=True`` is **required and measured**, not decoration: the store's
    ``offload`` skips an entry whose ``cache_enabled`` is ``False`` (the constructor
    default), so without it no store-owned codec pair is ever written and the case would
    assert the removal of files that never existed. ``pickle_container=True`` is
    likewise required twice over -- it is what writes the ``.npy`` + ``.meta.json``
    pair through the store's own codec, and what clears the in-memory reference so that
    dropping the caller's is enough to collect the entry.

    Returns
    -------
    tuple[str, list[str]]
        The exception type name raised by ``purge`` (or ``"completed"``), and the sorted
        names of the store-owned artefacts still on disk afterwards.
    """
    entry = DiskBackedNDArray(_FIRST.copy(), enable_caching=True)
    store[key] = entry

    backing = entry.cache_path
    assert backing is not None, "precondition: an entry constructed without a cache_path is still allocated one"
    assert not Path(backing).is_relative_to(cache_dir), (
        "precondition: an entry constructed with no cache_path must be backed OUTSIDE the store's cache "
        "directory -- containment is asserted negatively rather than by matching a temp-directory literal, "
        "because TMPDIR is a supported knob (entry 82) and a site pointing it at node-local scratch is the "
        "configuration this policy was designed to compose with"
    )

    store.offload(key, pickle_container=True)
    npy = paths_mod.get_npy_path(cache_dir, key)
    meta = paths_mod.get_meta_path(cache_dir, key)
    assert npy.exists() and meta.exists(), (
        f"precondition: the store-owned codec pair for {key!r} must be on disk before the purge, or the case "
        f"would assert the removal of files that were never written"
    )

    if not hold_reference:
        del entry
        gc.collect()

    outcome = "completed"
    try:
        store.purge(key)
    except Exception as exc:  # noqa: BLE001 - the type is recorded, never asserted on
        outcome = type(exc).__name__
    return outcome, sorted(p.name for p in (npy, meta) if p.exists())


def test_an_ordinarily_constructed_entry_does_not_make_its_key_unpurgeable(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-11 / CR2-04 / D-15-G5 / STORE-04 -- parts 1 and 3 of a three-part finding.

    ``store[key] = DiskBackedNDArray(arr)`` supplies no ``cache_path``, so
    ``_init_from_config`` allocates one with ``tempfile.mkstemp`` in the system temp
    directory. The reconcile finds that path outside the cache directory and refuses --
    **forever**, while the entry lives. Measured at ``0a2dab2``::

        entry cache_path : <system temp>/tmpXXXXXXXX.dat
        cache listing    : ['feat.meta.json', 'feat.npy']
        purge -> raised StorePurgeForeignArtefactError
        after drop+gc    : ['feat.meta.json', 'feat.npy']   <-- store-owned pair still there

    **Part 1, asserted here.** The refusal's prescribed remedy does not work. Both the
    exception message and the contract page say *"drop the entry and let garbage
    collection reclaim it"*, and GC runs ``_purge_memmap_file``, which by explicit
    design removes **only** the entry's memmap and never the store-owned ``<key>.npy``
    + ``<key>.meta.json``. Those two are therefore unreachable by any removal verb the
    package exposes -- a leak created by the refusal itself.

    **Part 3, asserted here as its own message.** The refusal is not a property of the
    key or of the disk state; it is a property of **object liveness**. Same store, same
    on-disk artefacts, same call -- and if the caller has dropped its reference and GC
    has run, the weak registry is empty and the identical call succeeds. *A destructive
    verb whose refusal flips on a garbage collection is not a contract a caller can
    program against.* That assertion is also what stops a fix passing this case by
    accident on the GC-empty path.

    **Part 2, the explicit-outside half, is deliberately NOT asserted here** -- it is
    filed as ``D-15-19`` and stays open past this round. An entry given an explicit
    path outside the cache directory is STORE-02/D-17's genuine foreign-artefact case
    and must keep being refused; only the *allocated* path is ephemeral by policy.

    **The policy is asserted, never the mechanism.** Entry 82 records that provenance is
    not recoverable from the path and that only the constructor can know it, but the
    attribute that records it belongs to ``15-12``. A red test that imported the fix's
    vocabulary could not be red before the fix.
    """
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)

    held_outcome, held_survivors = _purge_an_ordinarily_constructed_key(
        store, tmp_cache_dir, "held", hold_reference=True
    )
    dropped_outcome, dropped_survivors = _purge_an_ordinarily_constructed_key(
        store, tmp_cache_dir, "dropped", hold_reference=False
    )

    assert held_outcome == "completed", (
        f"CR2-04 / D-15-G5 violated: purging an ordinarily-constructed key raised {held_outcome} while the "
        f"caller held the entry. The entry's own allocated backing file is not the store's to manage, but "
        f"the <key>.npy + <key>.meta.json pair is unambiguously the store's to delete -- vetoing their "
        f"removal makes the key permanently unpurgeable and is a behaviour regression: this call succeeded "
        f"before the refusal landed"
    )
    assert not held_survivors, (
        f"CR2-04 violated: the store-owned artefacts {held_survivors} survive the purge. GC cannot reclaim "
        f"them -- the finalizer removes only the entry's memmap -- so the refusal's prescribed remedy "
        f"('drop the entry and let garbage collection reclaim it') leaves them unreachable by every removal "
        f"verb this package exposes"
    )
    assert held_outcome == dropped_outcome, (
        f"CR2-04 violated: the same call on the same key shape produced {held_outcome!r} with the entry "
        f"held and {dropped_outcome!r} with the reference dropped and a collection forced. Nothing about "
        f"the key or the disk state differs -- only whether the garbage collector had run. A destructive "
        f"verb whose refusal flips on a garbage collection is not a contract a caller can program against, "
        f"and nothing in the docstring, the exception text or the contract page mentions it"
    )
    assert not dropped_survivors, (
        f"CR2-04 violated: the store-owned artefacts {dropped_survivors} survive the purge on the "
        f"reference-dropped path too, so the outcome agrees for the wrong reason"
    )


# ---------------------------------------------------------------------------
# D-15-G5 — the provenance flag itself, and the half of CR2-04 that stays open.
# ---------------------------------------------------------------------------


def test_the_ephemeral_flag_is_derived_at_construction_and_is_not_configurable(tmp_cache_dir: Path) -> None:
    """Plan 15-12 / D-15-G5 / entry 82 / T-15-51 -- provenance is recorded, never stated.

    Three properties, and the third is the security-relevant one:

    1. an entry that ALLOCATED its backing file reports ``ephemeral is True``; one
       given an explicit path reports ``False``. Provenance is not recoverable from
       the path afterwards, which is why the constructor records it.
    2. the flag is **read-only**. There is no setter, so assignment raises
       :exc:`AttributeError` -- the same answer the sealed ``cache_path`` gives.
    3. the flag is **not a configuration field** (T-15-51). Were it configurable a
       caller could supply an explicit path and *declare* it ephemeral, and ``purge``
       would act on the lie -- declining to remove a file it is responsible for.

    **Measured, not assumed, for property 3:** ``LazyDiskCacheConfig`` and the
    constructor both **silently ignore** an unknown ``ephemeral=`` keyword rather than
    rejecting it, so the field is absent by way of an ignored extra rather than a
    raised error. That is the weaker of the two outcomes, which is exactly why the
    assertion below is on the DERIVED VALUE and not on an exception: an ignored extra
    that nonetheless flipped the flag would be the vulnerability, and a raise would
    merely be a nicer way of not having one.
    """
    allocated = DiskBackedNDArray(_FIRST.copy(), enable_caching=False)
    explicit = DiskBackedNDArray(_FIRST.copy(), enable_caching=False, cache_path=tmp_cache_dir / "explicit")

    assert allocated.ephemeral is True, (
        "D-15-G5 violated: an entry constructed with no cache_path allocated its own backing file with "
        "tempfile.mkstemp and must report itself ephemeral -- that file is fire-and-forget under an "
        "OS-managed lifetime and is not the store's to manage"
    )
    assert explicit.ephemeral is False, (
        "D-15-G5 violated: an entry given an explicit cache_path did NOT allocate its own backing file, so it "
        "is not ephemeral. Collapsing the two would make purge ignore a file it is responsible for"
    )

    with pytest.raises(AttributeError):
        explicit.ephemeral = True  # type: ignore[misc]
    assert explicit.ephemeral is False, "the refused assignment must not have landed"

    # Property 3, behaviourally: the extra keyword is accepted and ignored on both
    # routes, and neither route changes the derived value.
    lied_config = LazyDiskCacheConfig(cache_path=tmp_cache_dir / "lied", ephemeral=True)  # type: ignore[call-arg]
    assert not hasattr(lied_config, "ephemeral"), (
        "T-15-51: LazyDiskCacheConfig grew an `ephemeral` field. Provenance must be derived in "
        "_init_from_config from `config.cache_path is None` and never stated by a caller"
    )
    lied_entry = DiskBackedNDArray(
        _FIRST.copy(),
        enable_caching=False,
        cache_path=tmp_cache_dir / "lied_entry",
        ephemeral=True,  # type: ignore[call-arg]
    )
    assert lied_entry.ephemeral is False, (
        "T-15-51 violated: a caller supplied an explicit cache_path AND declared the entry ephemeral, and the "
        "declaration won. purge would then skip an entry whose in-cache file it is responsible for deleting -- "
        "the lie the derived-not-configured rule exists to make unrepresentable"
    )


def test_the_ephemeral_flag_survives_a_pickle_round_trip(tmp_cache_dir: Path) -> None:
    """Plan 15-12 / D-15-G5 / entry 82 -- the pickle property is measured, not argued.

    Entry 82 rests on ``__getstate__`` returning ``self.__dict__.copy()`` popping only
    ``_lock`` and ``_finalizer``, so a plain instance attribute survives pickle with no
    pickle-protocol code at all. That is a claim about the current implementation, so it
    is asserted rather than left in a docstring: if ``__getstate__`` is ever narrowed to
    an explicit allow-list of keys, this case is what says so.

    Both entries are built ``enable_caching=False`` and are read by nothing afterwards.
    ``__getstate__`` calls ``disable_purge()`` **unconditionally** and offloads when
    caching is enabled, so a shared entry would be mutated by the round-trip -- the same
    isolation ``15-09`` established for the copy mechanism (its Deviation 3).
    """
    allocated = DiskBackedNDArray(_FIRST.copy(), enable_caching=False)
    explicit = DiskBackedNDArray(_FIRST.copy(), enable_caching=False, cache_path=tmp_cache_dir / "pickled")

    assert pickle.loads(pickle.dumps(allocated)).ephemeral is True, (
        "D-15-G5 violated: the ephemeral flag did not survive a pickle round-trip for an allocated backing "
        "file. An unpickled entry that forgot its provenance would be treated as store-managed by purge, "
        "which is the joblib worker path this package is built around"
    )
    assert pickle.loads(pickle.dumps(explicit)).ephemeral is False, (
        "D-15-G5 violated: an explicit-path entry came back from a pickle round-trip reporting itself "
        "ephemeral, so purge would skip a file it is responsible for"
    )


def test_copying_an_entry_preserves_ephemeral_and_the_sources_measured_side_effects(tmp_cache_dir: Path) -> None:
    """Plan 15-12 / D-15-G5 / R3-04 -- the copy conclusion is read off the COPY.

    This entry is built for this assertion alone and is read by nothing afterwards,
    because ``copy.copy`` routes through ``__getstate__``/``__setstate__`` and
    ``__getstate__`` calls :meth:`~LazyDiskCache.disable_purge` **unconditionally** --
    which detaches the SOURCE's finalizer and flips its ``purge_disk_on_gc`` to
    ``False`` before the state the copy is rebuilt from is even returned. A case that
    shared its entry with another assertion would be reading a provenance flag off an
    object the check itself mutated. ``enable_caching=False`` is the isolation ``15-09``
    already established for exactly this mechanism (its Deviation 3 records a copy test
    written the naive way failing its own precondition on the force-offload); it is
    reused rather than replaced with a second one.

    The source's side effects are **measured and pinned, not presumed absent**. The
    observed transition on this code is ``_finalizer.alive: True -> False`` and
    ``purge_disk_on_gc: True -> False``, and the copy comes back with a freshly
    registered finalizer because ``__setstate__`` routes through ``enable_purge`` (D-19).
    An "assert no side effects" written blind would be false against the code.
    """
    entry = DiskBackedNDArray(_FIRST.copy(), enable_caching=False, purge_disk_on_gc=True)
    assert entry.ephemeral is True, "precondition: this entry must have allocated its own backing file"

    alive_before = entry._finalizer.alive
    purge_intent_before = entry.purge_disk_on_gc
    assert alive_before is True and purge_intent_before is True, (
        "precondition: a purge_disk_on_gc=True entry starts with an armed finalizer and the intent recorded"
    )

    duplicate = copy.copy(entry)

    assert duplicate.ephemeral is True, (
        "D-15-G5 violated: the copy of an entry that allocated its own backing file reports itself "
        "non-ephemeral. The conclusion is read off the COPY deliberately -- the source has been mutated by "
        "this very call, so reading it would measure disable_purge() rather than the flag"
    )

    assert entry._finalizer.alive is False, (
        f"the SOURCE's finalizer was expected to be detached by copy.copy: __getstate__ calls disable_purge() "
        f"unconditionally before returning the state. Observed alive {alive_before} -> "
        f"{entry._finalizer.alive}. A run showing the source untouched means __getstate__ stopped calling "
        f"disable_purge(), which is a change to the pickle contract D-18/D-19 own and not a result to absorb"
    )
    assert entry.purge_disk_on_gc is False, (
        f"the SOURCE's purge_disk_on_gc was expected to be flipped by copy.copy, by the same unconditional "
        f"disable_purge(). Observed {purge_intent_before} -> {entry.purge_disk_on_gc}"
    )
    assert duplicate.purge_disk_on_gc is True, (
        "the COPY must carry the user's original purge intent: __getstate__ overrides the dumped flag with "
        "the snapshot taken before disable_purge(), and __setstate__ re-registers through enable_purge (D-19)"
    )


def test_an_explicit_cache_path_outside_the_cache_directory_still_refuses(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-12 / D-15-G5 / D-15-19 / STORE-02 / D-17 -- the ephemeral skip is NARROW.

    The measurable statement that ``15-12`` stopped refusing on *allocated* paths and not
    on *explicit* ones. An entry given an explicit ``cache_path`` outside the cache
    directory is STORE-02 / D-17's genuine foreign artefact and must keep raising
    :exc:`StorePurgeForeignArtefactError` **exactly**.

    This case is also the standing evidence for the half of ``CR2-04`` this round
    deliberately leaves open. While the refusal stands, the store-owned ``<key>.npy`` +
    ``<key>.meta.json`` for this key are reachable by no removal verb the package
    exposes -- asserted below, so ``15-15`` files ``D-15-19`` against a measurement
    rather than against a plan. Splitting the refusal to remove them would change what a
    refusal MEANS, from *nothing was touched* to *some things were*, one round after
    D-15-G2a locked that semantics; the residual is preferred to reopening it.
    """
    key = "explicit_outside"
    elsewhere = tmp_cache_dir.parent / "elsewhere_explicit"
    elsewhere.mkdir()

    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)
    entry = DiskBackedNDArray(_FIRST.copy(), enable_caching=True, cache_path=elsewhere / key, purge_disk_on_gc=True)
    assert entry.ephemeral is False, (
        "precondition: an explicitly supplied path is NOT ephemeral, or this case would be testing the skip "
        "instead of the refusal it exists to preserve"
    )
    store[key] = entry
    store.offload(key, pickle_container=True)

    foreign_dat = elsewhere / f"{key}{paths_mod.MEMMAP_SUFFIX}"
    npy = paths_mod.get_npy_path(tmp_cache_dir, key)
    meta = paths_mod.get_meta_path(tmp_cache_dir, key)
    assert foreign_dat.exists(), "precondition: the entry must have written its .dat outside the cache directory"
    assert npy.exists() and meta.exists(), "precondition: the store-owned codec pair must be on disk"

    with pytest.raises(StorePurgeForeignArtefactError):
        store.purge(key)

    assert key in store, "SC-2: the refusal must leave the key tracked"
    assert entry._finalizer.alive is True, "SC-2: the refusal must leave the entry's finalizer armed"
    assert foreign_dat.exists(), "the refusal must not reach outside the cache directory"

    # D-15-19, stated as a measurement: the store-owned pair survives the refusal and is
    # reachable by no removal verb while the refusal stands. Recorded here so the
    # residual is evidenced rather than asserted.
    assert npy.exists() and meta.exists(), (
        "the refusal is a no-op by contract (D-15-G2a), so the store-owned codec pair necessarily survives "
        "it. That is the open half of CR2-04, filed as D-15-19 -- this assertion is its evidence, and it must "
        "be UPDATED rather than deleted if a later round makes the pair reachable"
    )


# ---------------------------------------------------------------------------
# D-15-G6's rescan half — a `.dat` with no key that owns it.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason=f"{_OPEN} (rescan half) - closed by 15-12; remove this marker there")
def test_a_dat_only_orphan_is_enumerable_by_a_fresh_store(make_store: MakeStore, tmp_cache_dir: Path) -> None:
    """Plan 15-11 / D-15-G6 rescan half / D-15-02 / STORE-04.

    The orphan is produced through **the library's own opt-out sequence**, never by
    writing a file by hand, because entry 83 records that sequence as *correct
    behaviour*: ``disable_purge()`` then ``del store[key]`` instructs "do not delete" at
    every step, and ``__delitem__`` is documented as dropping tracking and unlinking
    nothing. What is left behind is not a bug and is not undeletable -- ``purge``
    derives its paths from the key, so a fresh store in a fresh process removes it with
    no entry object in existence.

    The genuine residual is **enumerability**. The reopen rescan globs the ``.npy``
    suffix only, so a ``.dat``-only leftover is invisible: a fresh store tracks ``[]``
    while an offloaded codec pair tracks correctly. An orphan nobody can list is an
    orphan nobody can purge, because purging it requires already knowing the key --
    and a cache-directory orphan, unlike the ephemeral scratch file, has no janitor.
    Extending the glob closes D-15-02's cheap half with no new API.

    The third assertion below was **measured before it was written**, not predicted. A
    ``.dat``-only key is *tracked but not loadable*, which is a state this round
    introduces and which the plan deliberately did not presume the error type of. The
    tracked-with-no-codec-pair read was simulated against ``0a2dab2`` by installing the
    key the extended glob will install, and the observed outcome was ``KeyError``:
    ``_load_entry`` requires **both** halves of the codec pair and reports their absence
    as a cache miss, which the subscript surfaces as a missing key. That is the right
    answer and it is pinned rather than left to the fix to choose.
    """
    key = "orphan"
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)
    store.add_data_to_store(key, _FIRST.copy())

    entry = store.store[key]
    assert entry is not None, "precondition: add_data_to_store tracks a live entry"
    entry.disable_purge()
    del store[key]
    del entry
    gc.collect()

    # Defensive and explicit: the sequence above writes no codec pair, but the shape
    # this case is about is `<key>.dat` ALONE, so anything else is removed rather than
    # assumed absent.
    paths_mod.get_npy_path(tmp_cache_dir, key).unlink(missing_ok=True)
    paths_mod.get_meta_path(tmp_cache_dir, key).unlink(missing_ok=True)
    dat = paths_mod.get_memmap_path(tmp_cache_dir, key)
    assert sorted(p.name for p in tmp_cache_dir.iterdir()) == [dat.name], (
        f"precondition: the opt-out sequence must leave EXACTLY {dat.name!r} behind -- the case is about a "
        f"cache directory holding nothing but a .dat, and any other file would give the rescan a second "
        f"route to the key"
    )
    del store
    gc.collect()

    fresh = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=False)

    assert key in fresh.keys(), (
        f"D-15-G6 / D-15-02 violated: a fresh store over a cache directory holding {dat.name!r} tracks "
        f"{fresh.keys()!r} -- the key is invisible to enumeration, so nobody who did not already know it can "
        f"discover the orphan. The rescan must glob the memmap suffix as well as the codec one"
    )

    try:
        fresh[key]
    except KeyError:
        pass
    except Exception as exc:  # noqa: BLE001 - a measured pin, so the observed type is reported
        raise AssertionError(
            f"reading a tracked key with no codec pair raised {type(exc).__name__}, not KeyError. Measured "
            f"at 0a2dab2 the answer is KeyError: _load_entry requires BOTH halves of the codec pair and "
            f"reports their absence as a cache miss. A different type here means the tracked-but-not-loadable "
            f"state acquired a new contract that nothing published"
        ) from exc
    else:
        raise AssertionError(
            "reading a tracked key with no codec pair returned a value. The rescan installs a tracking "
            "entry, not data -- a `.dat` is an opaque memmap with no shape or dtype sidecar, so there is "
            "nothing to reconstruct an array from and the read must fail with KeyError"
        )

    fresh.purge(key)

    assert_nothing_derived_from(tmp_cache_dir, key)
    assert not list(tmp_cache_dir.iterdir()), (
        f"the consequence enumerability buys did not follow: purging the now-listable orphan left "
        f"{sorted(p.name for p in tmp_cache_dir.iterdir())!r} in the cache directory"
    )


# ---------------------------------------------------------------------------
# CR2-03 — a link aimed at another key's artefact, and the adopted link that
# must keep working beside it.
# ---------------------------------------------------------------------------


def _plant_dat_symlink(cache_dir: Path, key: str, target: Path) -> Path:
    """Point ``<key>.dat`` at ``target`` and return the link path.

    Planted **before the store exists** wherever the shape needs to be admitted at
    construction, which is what makes Phase 14's D-31 resolved-containment check accept
    it, as it is meant to: the target is inside the cache directory, so the resolved
    final component is contained.
    """
    link = paths_mod.get_memmap_path(cache_dir, key)
    link.symlink_to(target)
    return link


@_POSIX_ONLY
@pytest.mark.xfail(strict=True, reason=f"{_OPEN} (CR2-03) - closed by 15-13; remove this marker there")
def test_purge_refuses_to_follow_a_link_aimed_at_another_keys_artefact(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-11 / CR2-03 / D-15-G7 / STORE-04 -- the cross-key destruction primitive.

    ``15-09`` made ``purge`` follow each built path to its resolved target and unlink
    that target first, and the containment gate requires only that the target be
    *inside the cache directory* -- it does not require the target to have anything to
    do with the key. So an in-cache ``<keyA>.dat`` symlink aimed at another key's real,
    store-written artefacts makes ``purge(keyA)`` destroy them. Measured at
    ``0a2dab2``::

        before: ['aggressor.dat', 'victim.meta.json', 'victim.npy']
        # store.purge("aggressor")
        after : ['victim.meta.json']
        victim.npy survived: False
        store['victim'] ->  KeyError 'victim'

    ``purge("aggressor")`` returned **cleanly** and silently orphaned a live key. The
    published limit covers only *"two keys whose ``<key>.dat`` links point at the same
    payload"* and argues the shape needs an operator planting two links at one target.
    **One** planted link is enough, and the destroyed file is a store-owned artefact
    rather than a shared adopted payload -- so the rule immediately above that limit,
    *"everything belonging to this key and nothing else"*, is false for this input.

    The precondition -- write access to the cache directory -- is the module's own
    accepted residual D-17 / T-14-18, and the *adopted entry* feature exists precisely
    to make in-cache links a supported shape, so this is reachable by a legitimate
    operator action and not only by an attacker.

    **No exception type is asserted, and the omission is deliberate.** ``15-13``
    chooses how the refusal is spelled; naming a symbol it has not written yet would
    raise at *import* and take the whole module down with a collection error, which is
    the trap ``15-08`` recorded and avoided. The typed companion belongs to ``15-13``.

    **Whether the aggressor link itself survives is likewise not asserted.** The case is
    about ownership -- who may destroy ``victim.npy`` -- and unlinking an in-cache link
    is an in-cache operation whatever it points at, so both answers are defensible and
    the choice is ``15-13``'s.
    """
    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)
    store.add_data_to_store("victim", _FIRST.copy())
    store.offload("victim", pickle_container=True)

    victim_npy = paths_mod.get_npy_path(tmp_cache_dir, "victim")
    assert victim_npy.exists(), (
        "precondition: 'victim' must have a genuine store-written .npy -- the finding is about destroying a "
        "store-owned artefact, not a planted decoy"
    )
    link = _plant_dat_symlink(tmp_cache_dir, "aggressor", victim_npy)
    assert link.is_symlink() and link.resolve() == victim_npy.resolve(), (
        "precondition: <aggressor>.dat must resolve to victim's .npy, or the case exercises the ordinary path"
    )

    observed: str = "no exception"
    try:
        store.purge("aggressor")
    except Exception as exc:  # noqa: BLE001 - the type is recorded, never asserted on
        observed = type(exc).__name__

    assert victim_npy.exists(), (
        f"CR2-03 / D-15-G7 violated (purge raised: {observed}): purging 'aggressor' destroyed 'victim's "
        f"store-written .npy through a planted link. purge must refuse to follow a resolved target that is "
        f"another key's built artefact -- containment answers 'may I delete inside my own directory', which "
        f"is not the same question as 'is this file mine to delete'"
    )
    assert np.array_equal(np.asarray(store["victim"]), _FIRST), (
        f"CR2-03 violated (purge raised: {observed}): 'victim' is no longer readable as the array that was "
        f"written to it. A file that survives holding the wrong bytes, or a key that survives tracking but "
        f"raises on read, is the same silent orphaning wearing a different green tick"
    )


@_POSIX_ONLY
def test_purge_still_follows_an_adopted_link_to_a_payload_that_is_not_a_store_artefact(
    make_store: MakeStore, tmp_cache_dir: Path
) -> None:
    """Plan 15-11 / D-17 / STORE-03 / D-15-G7 -- the boundary the refusal must NOT cross.

    Green at this plan's HEAD, and it must stay green. The legitimate adopted entry that
    D-17 and STORE-03 exist to protect is identified by **what the target is NAMED**,
    not by the fact that a link was followed: the library writes the key's data through
    an adopted ``<key>.dat``, and ``15-09`` made ``purge`` remove the payload behind it
    precisely so a "complete" removal does not leave the key's exact bytes sitting in
    the cache directory under another name.

    **This is the case that goes red if ``15-13``'s refusal is written too wide.** A
    blanket "never follow a link" -- or a check on the fact of the link rather than on
    the artefact-shaped name of its target -- satisfies the CR2-03 case above by
    refusing everything, and would reopen G-2a as a residual payload the moment it
    landed. The payload here is deliberately named with an extension that belongs to no
    store artefact vocabulary, so a refusal that keys on the target's NAME passes both
    cases and a refusal that keys on the link does not.
    """
    key = "adopted_round3"
    payload = tmp_cache_dir / "innocuous_round3.bin"
    payload.write_bytes(b"")
    link = _plant_dat_symlink(tmp_cache_dir, key, payload)

    store = make_store(tmp_cache_dir, enable_caching=True, purge_disk_on_gc=True)
    store.add_data_to_store(key, _SECOND.copy())

    assert link.is_symlink(), (
        "precondition: <key>.dat must still be a symlink after the library's own write, or adoption is "
        "already broken and this case is measuring something else"
    )
    assert np.array_equal(np.fromfile(payload, dtype=np.float32), _SECOND), (
        "precondition: the library must have written the key's data THROUGH the link into the payload"
    )

    store.purge(key)

    assert not payload.exists(), (
        f"the adopted payload {payload.name!r} survived the purge, so the key's exact bytes are still in the "
        f"cache directory under a name that does not derive from the key. That is G-2a reopened -- a refusal "
        f"aimed at CR2-03 must key on the TARGET'S NAME being another key's artefact, never on the fact that "
        f"a link was followed"
    )
    assert_nothing_derived_from(tmp_cache_dir, key)
    assert not list(tmp_cache_dir.iterdir()), (
        f"the purge left {sorted(p.name for p in tmp_cache_dir.iterdir())!r} in the cache directory"
    )
