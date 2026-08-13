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

import gc
import os
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

from GSEGUtils.lazy_disk_cache import paths as paths_mod
from GSEGUtils.lazy_disk_cache.disk_backed_ndarray import DiskBackedNDArray
from GSEGUtils.lazy_disk_cache.disk_backed_store import DiskBackedStore

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
