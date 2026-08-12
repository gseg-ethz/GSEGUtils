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

import gc
from pathlib import Path
from typing import Callable

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
_G1_OPEN = "G-1 open at 0956838 - closed by 15-09; remove this marker there"


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


# 15-09 removes the three `_G1_OPEN` marks below once the detach reaches every route.
@pytest.mark.parametrize(
    "route",
    [
        pytest.param("tracked", id="tracked"),
        pytest.param("offload", id="offload", marks=pytest.mark.xfail(strict=True, reason=_G1_OPEN)),
        pytest.param("del", id="del", marks=pytest.mark.xfail(strict=True, reason=_G1_OPEN)),
        pytest.param("pop", id="pop", marks=pytest.mark.xfail(strict=True, reason=_G1_OPEN)),
    ],
)
def test_the_aba_hazard_cannot_fire_on_any_route_out_of_the_mapping(
    make_store: MakeStore, tmp_cache_dir: Path, route: str
) -> None:
    """Plan 15-08 / G-1 / SC-3 / STORE-04 -- the whole hazard, on every route into ``purge``.

    Four routes reach ``purge`` with an entry that still holds a live
    ``weakref.finalize`` recorded against ``<key>.dat``. ``purge`` detaches it through
    ``entry = self._store.get(key) if tracked else None``
    (``disk_backed_store.py:1155``), and that expression is ``None`` on three of them:
    an **offloaded** entry is stored as ``None`` in a mapping typed
    ``dict[str, Optional[T]]``, and the **untracked orphan** left by ``del`` / ``pop``
    is exactly the state D-02 widened the *existence* check to reach
    (``tracked or any(p.exists())``, ``:1140-1142``). The existence check was widened
    and the detach was not -- that is the gap in one sentence.

    ``tracked`` is unmarked and green today: it is the positive control that proves
    this body is not vacuous, and it is the one route the three shipped tests take.
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
        f"including a later entry created under the same key. purge detaches only when "
        f"`self._store.get(key)` returns a live object, which on this route it does not"
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
