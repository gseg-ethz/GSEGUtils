"""Route-policy matrix for the disk-backed store's key containment.

Covers STORE-01 (lexical key validation at every route that writes into
``_store``, extended by D-11 to the read route), and is extended later in the
phase with STORE-02 (resolved containment at every path builder) and STORE-03
(symlink tolerance). This file is created by Plan 14-03 with the six-route
policy matrix and the SC-1 worker-safety proof; Plan 14-06 extends it with the
builder-containment and symlink tests; Plan 14-07 extends it with the migration
snippet's round-trip.

The point of this file is the *differences* between the routes. Six distinct
policies apply to twelve routes, and each difference is a decision:

===================== ============================================== ======
Route                 Policy                                         Basis
===================== ============================================== ======
``__setitem__``       raise, with no filesystem syscall              SC-1
``add_data_to_store`` raise, unconditionally                         SC-1
``__getitem__``       raise, before ``_load_entry`` touches disk     D-11
``get``               return the default                             D-11a
``pop(key, default)`` return the default; remove a key that is there D-29
``pop(key)``          raise, and still remove a key that is there    D-29
``setdefault``        raise — it is a write route                    D-25
``clear``             remove every tracked key, or raise             D-29
``popitem``           left inherited; its ``KeyError`` carries a key D-29
``__init__`` rescan   warn and skip, leaving the file untouched      D-09
``__setstate__``      raise, never a silently short store            D-10
``store`` property    not a write route at all — a read-only view    D-19
``__getstate__``      not a write route at all — a detached snapshot D-19
===================== ============================================== ======

The table has thirteen rows for twelve routes because ``pop``'s two forms took
one row until D-29 split them: they now differ **only** in what they return, and
writing that as one row is how the round-3 draft came to claim a delete it did
not perform.

The ``pop`` and ``setdefault`` rows are one decision seen from both sides
(D-25), and they are in the table because the phase *changed* one of them
without noticing. ``pop`` reaches the subscript by the same route ``get`` does,
so D-11 and D-12 between them turned ``pop(key, default)`` from returning the
default into raising — silently, one accessor over from the change that
motivated the ``get`` override. It is restored. ``setdefault`` moves the same
way and is **deliberately** left raising, because it *inserts*: refusing an
illegal key at a write route is the whole point of this phase. The asymmetry is
a choice, and it is written down in three places — here, in ``pop``'s docstring
and in the migration note — precisely so the next reader does not read it as an
oversight and helpfully "finish" it.

The ``clear`` and ``popitem`` rows are D-29, and they are here because round 3
published a five-route enumeration marked *exhaustive* that named neither.
Both are supplied by ``MutableMapping`` and neither appeared anywhere in
``disk_backed_store.py``, so the set was invisible from the diff that changed
their behaviour — which is the mechanism worth carrying away, not the two
names. ``clear`` was ``while True: self.popitem()`` under ``except KeyError:
pass``, and measured over four tracked keys one of which had no readable
payload it returned, **raising nothing**, with three keys still in the store.
It is overridden. ``popitem`` is **deliberately left inherited**, and the
reason is measured rather than asserted: its ``KeyError`` on a non-empty store
carries the key (``args == ('ghost',)``) while its ``KeyError`` on an empty one
does not (``args == ()``), so the two cases were always distinguishable and
``clear``'s bare ``except KeyError`` is what threw the distinction away. The
defect was in the handler, not in the signal. See ``clear``'s docstring for the
three override shapes that were rejected.

The last two rows are the odd ones out on purpose, and the difference is the
decision: the other six routes *validate* a write, while these two have no
write to validate. Plan 14-08 closed the ``store`` property structurally rather
than adding a seventh validation site, so SC-1's "every route" is an invariant
rather than an enumeration of the routes someone remembered.

The final row is that same argument **finished**, and it is here because the
enumeration was not the whole set of protocol routes. Plan 14-08's structural
closure was premature as stated: ``__getstate__``'s shallow
``self.__dict__.copy()`` still aliased the entry mapping, so the snapshot it
returned was a live write route into ``_store`` through a public protocol
method — and ``copy.copy(store)``, which travels the same protocol, handed back
a store sharing the original's key set. Plan 14-12 detaches the snapshot
(CR-02). A claim quantified over "every route" that is maintained by
enumerating routes degrades silently at the next route somebody adds, which is
precisely how this one was missed.
"""

import ast
import copy
import inspect
import os
import pickle
import re
import tempfile
import textwrap
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Optional, cast

import numpy as np
import pytest

import GSEGUtils.lazy_disk_cache as lazy_disk_cache_package
from GSEGUtils.lazy_disk_cache import (
    StoreContainmentError,
    StoreKeyError,
    get_legacy_pickle_path,
    get_meta_path,
    get_npy_path,
    is_valid_store_key,
)
from GSEGUtils.lazy_disk_cache import paths as paths_mod
from GSEGUtils.lazy_disk_cache.disk_backed_ndarray import DiskBackedNDArray
from GSEGUtils.lazy_disk_cache.disk_backed_store import DiskBackedStore
from GSEGUtils.lazy_disk_cache.lazy_disk_cache import LazyDiskCacheConfig

#: The canonical illegal key: it is the reproduced pc2img escape, and it is
#: refused by the separator clause rather than by its dots (D-06).
ILLEGAL_KEY = "../victim"

#: A store factory as injected by ``conftest.make_store``. Annotated
#: structurally rather than by importing the conftest ``MakeStore`` protocol,
#: so this module does not depend on conftest being importable as a module.
MakeStoreFn = Callable[..., DiskBackedStore[DiskBackedNDArray]]


def _store_with_cache_path(
    cache_path: Optional[Path], *, fallback_root: Optional[Path] = None
) -> DiskBackedStore[DiskBackedNDArray]:
    """Build an empty store over ``cache_path`` — or over none at all.

    ``cache_path=None`` is the second half of the ``route_add_data``
    parametrisation: the store then falls back to a ``mkdtemp`` directory, and
    ``add_data_to_store``'s ``if self._cache_dir`` conditional is exercised in
    the state its author intended it to gate.

    **WR-09 — why ``fallback_root`` exists.** That ``mkdtemp`` fallback lands
    in the *system* temporary directory, outside pytest's managed tree, and
    nothing ever removes it: one leaked directory per invocation, surviving the
    whole session. ``fallback_root`` is the caller-supplied base that fixes it —
    for the duration of construction ``tempfile.tempdir`` points at it, so the
    fallback directory is created under the per-test ``tmp_path`` and pytest
    removes it with everything else. The fallback *branch* is still exercised
    exactly as before: ``cache_path`` really is ``None``, so ``__init__`` really
    does take the ``mkdtemp`` arm. Only where that directory lands changes.

    Passing ``cache_path=None`` without a ``fallback_root`` is refused rather
    than tolerated, so the leak cannot come back by someone adding a call site
    and not knowing about this.
    """
    if cache_path is None and fallback_root is None:
        raise AssertionError(
            "_store_with_cache_path(None) leaks a mkdtemp directory outside pytest's managed "
            "tree; pass fallback_root=<a tmp_path-derived directory> (WR-09)"
        )
    cfg = LazyDiskCacheConfig(
        enable_caching=True,
        cache_path=cache_path,
        purge_disk_on_gc=False,
        automatic_offloading=False,
    )
    if cache_path is None:
        previous_tempdir = tempfile.tempdir
        tempfile.tempdir = str(fallback_root)
        try:
            return DiskBackedStore[DiskBackedNDArray](config=cfg, factory=DiskBackedNDArray)
        finally:
            tempfile.tempdir = previous_tempdir
    return DiskBackedStore[DiskBackedNDArray](config=cfg, factory=DiskBackedNDArray)


def _entry(cache_path: Optional[Path] = None) -> DiskBackedNDArray:
    """Return a small, fully-constructed cache entry usable as a store value."""
    return DiskBackedNDArray(
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        enable_caching=False,
        cache_path=cache_path,
        automatic_offloading=False,
        purge_disk_on_gc=False,
    )


# ---------------------------------------------------------------------------
# route_setitem — raise, and do not track (SC-1)
# ---------------------------------------------------------------------------


def test_route_setitem_refuses_an_illegal_key_and_does_not_track_it(
    make_store: MakeStoreFn, tmp_cache_dir: Path
) -> None:
    """Plan 14-03 / STORE-01 / SC-1.

    ``__setitem__`` raises ``StoreKeyError`` for an illegal key **and** leaves
    the store untouched. The second assertion is the one that matters: a raise
    that still left the key tracked would be worse than no guard at all,
    because the entry would then be offloaded to the escaping path on the next
    ``offload()`` while the caller believes the insert failed.
    """
    store = make_store(tmp_cache_dir)

    with pytest.raises(StoreKeyError, match="Invalid store key"):
        store[ILLEGAL_KEY] = _entry()

    assert ILLEGAL_KEY not in store, "a refused key was still tracked by the store"
    assert store.keys() == [], "the refused insert left residue in the store"


# ---------------------------------------------------------------------------
# route_add_data — raise unconditionally, with and without a cache path (SC-1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("configured_cache_path", [True, False], ids=["with-cache-path", "without-cache-path"])
def test_route_add_data_refuses_an_illegal_key_unconditionally(
    configured_cache_path: bool, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Plan 14-03 / STORE-01 / SC-1.

    ``add_data_to_store`` validates **unconditionally**. Parametrised with and
    without a configured ``cache_path`` to pin that the ``if self._cache_dir``
    conditional further down that method can never gate the check.

    That conditional is dead code: ``Path`` defines no ``__bool__``, so a
    ``Path`` is always truthy, and ``__init__`` assigns one unconditionally
    (falling back to ``mkdtemp`` when no ``cache_path`` was configured). Both
    parametrisations must therefore refuse — if one ever accepts, the hole has
    been resurrected by someone "fixing" the conditional.
    """
    store = _store_with_cache_path(
        tmp_cache_dir if configured_cache_path else None,
        fallback_root=tmp_path,
    )

    with pytest.raises(StoreKeyError, match="Invalid store key"):
        store.add_data_to_store(ILLEGAL_KEY, np.zeros(3, dtype=np.float32))

    assert ILLEGAL_KEY not in store, "a refused key was still tracked by the store"


# ---------------------------------------------------------------------------
# route_getitem — raise before touching disk (D-11)
# ---------------------------------------------------------------------------


def test_route_getitem_refuses_before_load_entry_is_reached(
    make_store: MakeStoreFn, tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 14-03 / STORE-01 / D-11.

    ``__getitem__`` refuses an illegal key **before** ``_load_entry`` runs, so
    no path is built from it and the legacy-``.pkl`` ``exists()`` probe never
    fires on an unvalidated path.

    Asserting only that the call raises would pass even with the guard placed
    *after* the load — the load would probe disk, miss, and the guard would
    then raise on the way out. So ``_load_entry`` is replaced with a callable
    that records its own invocation, and the test asserts it was never reached.
    """
    store = make_store(tmp_cache_dir)
    real_load_entry = DiskBackedStore._load_entry
    calls: list[str] = []

    def recording_load_entry(self: DiskBackedStore[Any], key: str) -> Any:
        calls.append(key)
        raise AssertionError(f"_load_entry was reached with an unvalidated key {key!r}")

    monkeypatch.setattr(DiskBackedStore, "_load_entry", recording_load_entry)
    try:
        with pytest.raises(StoreKeyError, match="Invalid store key"):
            store[ILLEGAL_KEY]
    finally:
        # explicitly (belt-and-suspenders; monkeypatch teardown does this too)
        monkeypatch.setattr(DiskBackedStore, "_load_entry", real_load_entry)

    assert calls == [], "the guard runs after the load — _load_entry saw the illegal key"


# ---------------------------------------------------------------------------
# route_get — the D-11 amendment: the default is still returned
# ---------------------------------------------------------------------------


def test_route_get_returns_the_default_while_the_subscript_still_raises(
    make_store: MakeStoreFn, tmp_cache_dir: Path
) -> None:
    """Plan 14-03 / STORE-01 / the D-11 amendment.

    ``DiskBackedStore`` subclasses ``MutableMapping``, whose inherited ``get``
    is ``try: return self[key] except KeyError: return default``. D-12 makes
    ``StoreKeyError`` a ``ValueError`` — deliberately not a ``KeyError``,
    because ``add_data_to_store`` already raises that one for "key exists" — so
    the moment D-11 makes ``__getitem__`` validate, the inherited ``get`` stops
    catching the refusal. ``get`` is therefore overridden to catch
    ``(KeyError, StoreKeyError)``.

    The three assertions stay in **one** test on purpose: they are one
    contract. Split across three tests, one can regress while the others stay
    green — which is exactly how this gap went unnoticed when D-11 was written,
    since D-11 enumerated ``__contains__`` and ``__delitem__`` and never
    reached ``.get()``.

    **The fourth assertion (Plan 14-10 / WR-03) belongs in this same function
    for that same reason**, and it is the one that reverses a direction. The
    first three say *swallow*; the fourth says *do not*, and the line between
    them is the whole of D-12's two-type split. ``StoreContainmentError``
    subclasses ``StoreKeyError``, so the catch tuple above swallowed it too —
    the library's own accessor doing precisely what the contract page tells
    consumers not to do: *"A per-item handler that skips one bad key should not
    silently swallow the second kind."* A refused key is evidence about the
    caller; a containment violation is evidence about the **environment**, and
    degrading it to a ``None`` turns a planted symlink into an ordinary cache
    miss inside the caller's loop. Asserted here rather than in a fifth test
    function because a reader who changes the catch tuple must meet all four
    halves of the contract at once.

    The containment error is driven through the module-level ``Sneaky`` helper
    (Plan 14-08): the lexical layer accepts its characters, and the builder
    inside ``_load_entry`` refuses what it actually formats to. That shape is
    also the honest reachability story — for a plain ``str`` key the containment
    layer cannot fire from this path once the lexical layer has passed, so this
    carve-out matters for the ``str``-subclass shape and for any future
    lexical-layer regression, which is exactly when the signal is most wanted.
    """
    store = make_store(tmp_cache_dir)
    sentinel = object()

    assert store.get(ILLEGAL_KEY, sentinel) is sentinel, "get raised (or returned something else) for an illegal key"
    assert (ILLEGAL_KEY in store) is False, "__contains__ must stay an unguarded, dict-backed membership test"
    with pytest.raises(StoreKeyError, match="Invalid store key"):
        store[ILLEGAL_KEY]

    with pytest.raises(StoreContainmentError, match=re.escape(paths_mod.CLAUSE_ESCAPES)):
        store.get(Sneaky("safe"), sentinel)


def test_route_get_still_returns_the_default_for_a_legal_but_absent_key(
    make_store: MakeStoreFn, tmp_cache_dir: Path
) -> None:
    """Plan 14-03 / STORE-01 / the D-11 amendment.

    The override must preserve the ordinary miss path rather than swallow
    everything: a *legal* key that is simply absent still returns the default,
    via the ``KeyError`` half of the catch tuple (raised by ``_load_entry``).
    """
    store = make_store(tmp_cache_dir)
    sentinel = object()

    assert store.get("absent_but_legal", sentinel) is sentinel
    assert store.get("absent_but_legal") is None


def test_route_get_is_overridden_on_the_class_not_inherited() -> None:
    """Plan 14-03 / STORE-01 / the D-11 amendment.

    Checked on ``vars()`` rather than ``hasattr``: the inherited
    ``Mapping.get`` satisfies ``hasattr``, so a ``hasattr`` assertion would
    stay green with the amendment silently deleted.
    """
    assert "get" in vars(DiskBackedStore), "get is the inherited Mapping.get, which catches only KeyError"


# ---------------------------------------------------------------------------
# route_pop / route_setdefault — the same mechanism as ``get``, one accessor
# over, and one of them changed silently (Plan 14-14 / D-25 / WR-01)
# ---------------------------------------------------------------------------


def test_route_pop_with_a_default_returns_it_while_the_bare_pop_still_raises(
    make_store: MakeStoreFn, tmp_cache_dir: Path
) -> None:
    """Plan 14-14 / STORE-01 / D-25 / WR-01 / T-14-61 / T-14-62.

    **The identical mechanism that motivated the ``get`` override, one accessor
    over.** ``MutableMapping.pop`` is ``try: value = self[key] except KeyError:
    return default`` — so the moment D-11 made ``__getitem__`` validate and D-12
    made the refusal a ``ValueError`` rather than a ``KeyError``, ``pop`` stopped
    catching it. Measured rather than recalled: at the phase-14 base commit
    ``10683dc`` ``store.pop('../victim', 'DEF')`` returned ``'DEF'``; at
    ``56c8306`` the same call raised ``StoreKeyError``. Nobody decided that, and
    BC-GSEG-006's enumeration of the read-route deltas omits it while sitting
    next to a paragraph that reassures the reader about ``.get()``.

    **All six behaviours live in one function, deliberately**, on the same
    "one contract, one test" grounds as the ``get`` sibling above: split across
    six tests, one half can regress while the others stay green — which is
    precisely how the original gap went unnoticed.

    The three behaviours that are *not* restored matter as much as the three
    that are. ``pop(key)`` with no default is a raise-on-miss in every mapping,
    so it keeps raising and D-25 restores the **defaulting form only**. A
    successful ``pop`` still **deletes**, which is what keeps this an override of
    ``pop`` rather than a second ``get`` under a different name. And a
    containment violation still propagates: ``pop`` reaches the subscript by
    exactly the route ``get`` does, so inverting its two handler clauses would
    restore the WR-03 swallow in a second place — a subclass caught after its
    base is never reached.
    """
    store = make_store(tmp_cache_dir)
    sentinel = object()

    # 1. Illegal key + a supplied default -> the default. This is D-25.
    assert store.pop(ILLEGAL_KEY, sentinel) is sentinel, (
        "pop raised for an illegal key despite a supplied default; the defaulting read routes disagree again"
    )

    # 2. Illegal key, no default -> the refusal type, exactly as the subscript
    #    gives it. The bare form is a write route on success.
    with pytest.raises(StoreKeyError, match="Invalid store key"):
        store.pop(ILLEGAL_KEY)

    # 3. / 4. A legal but absent key keeps ordinary mapping behaviour on both
    #    forms — the override must not swallow everything.
    assert store.pop("absent_but_legal", sentinel) is sentinel
    with pytest.raises(KeyError):
        store.pop("absent_but_legal")

    # 5. A successful pop returns the entry AND removes it. Asserted on the
    #    tracked key set before and after, and on the identity of the value,
    #    because a "fix" that turned the write route into a read would satisfy
    #    every other assertion here.
    entry = _entry()
    store["present"] = entry
    assert store.keys() == ["present"], "the fixture insert did not take, so the delete assertion proves nothing"
    popped = store.pop("present")
    assert popped is entry, "pop returned something other than the stored entry"
    assert store.keys() == [], "pop returned the entry without deleting it — a write route turned into a read"
    assert "present" not in store

    # 6. Containment evidence is never degraded into the default (WR-03, in a
    #    second place). Driven through the module-level ``Sneaky`` helper: the
    #    lexical layer accepts its characters and the builder inside the load
    #    path refuses what it actually formats to.
    with pytest.raises(StoreContainmentError, match=re.escape(paths_mod.CLAUSE_ESCAPES)):
        store.pop(Sneaky("safe"), sentinel)


def test_route_setdefault_still_raises_because_it_is_a_write_route(
    make_store: MakeStoreFn, tmp_cache_dir: Path
) -> None:
    """Plan 14-14 / STORE-01 / D-25 — the divergence, pinned as a decision.

    ``setdefault`` reaches the subscript exactly as ``pop`` does, and it is
    **deliberately left raising**. The asymmetry is the decision rather than the
    leftover: ``pop`` with a default is an *interrogative read* — it asks whether
    the store holds a key and accepts "no" for an answer — whereas
    ``setdefault`` **inserts**, and refusing an illegal key at a write route is
    the whole point of this phase. A ``setdefault`` that returned its default for
    ``'../victim'`` would answer as though the key were fine and then be expected
    to have stored something under it.

    Recorded as its own named test rather than as a sentence inside ``pop``'s,
    because a reader who finds ``pop`` overridden and ``setdefault`` not must be
    able to tell a choice from an oversight by grepping the suite.
    """
    store = make_store(tmp_cache_dir)

    with pytest.raises(StoreKeyError, match="Invalid store key"):
        store.setdefault(ILLEGAL_KEY, _entry())

    assert ILLEGAL_KEY not in store, "a refused setdefault still tracked the key"
    assert store.keys() == [], "the refused setdefault left residue in the store"


# ---------------------------------------------------------------------------
# route_pop / route_clear / route_popitem — removal semantics (D-29)
# ---------------------------------------------------------------------------


def _seed_offloaded(make_store: MakeStoreFn, cache_dir: Path, keys: list[str]) -> DiskBackedStore[DiskBackedNDArray]:
    """Seed a genuine codec pair per key, then return a **reopened** store over them.

    Reopening is what makes the seeded keys *tracked but not resident*: the
    rescan installs ``_store[key] = None`` and every read goes through
    ``_load_entry``. That is the only state in which a key can be present and
    its payload unreadable, and it is the state both round-3 defects live in.
    """
    writer = make_store(cache_dir)
    for i, key in enumerate(keys):
        writer.add_data_to_store(key, np.arange(4, dtype=np.float64) + i)
    writer.offload(pickle_container=True)
    del writer
    return make_store(cache_dir)


def _unlink_artefacts(cache_dir: Path, key: str) -> None:
    """Remove every on-disk artefact belonging to ``key``, leaving it tracked."""
    for path in cache_dir.glob(f"{key}.*"):
        path.unlink()


def test_route_pop_removes_a_tracked_key_whose_payload_cannot_be_loaded(
    make_store: MakeStoreFn, tmp_cache_dir: Path
) -> None:
    """Plan 14-17 / STORE-01 / D-29 / § WR-02a / T-14-76.

    **``dict.pop(k, d)`` removes ``k`` when ``k`` is present. This did not.**
    Measured at the post-14-16 tip, over a key the reopen rescan tracked and
    whose codec pair had then been unlinked::

        'gone' in s (before) : True
        bare pop             : KeyError('gone')
        'gone' in s (after)  : True
        defaulting pop       : 'DEF'
        'gone' in s (after)  : True

    The cause is a single handler that cannot tell *no such key* from *key
    exists, payload unreadable* — ``except (KeyError, StoreKeyError): return
    default`` sees the same exception type for both. A caller draining a store
    with ``pop(k, None)`` therefore completes a removal loop having removed
    nothing and having been told nothing.

    **Both forms are asserted, and they must differ only in what they return.**
    The bare form still raises, because raising on a miss is what a mapping
    ``pop`` does; but it now also removes, because the key *was* there. If only
    the defaulting form were fixed, the bare form would be the route that
    reports a failure and leaves the store in the state that caused it.

    **The containment carve-out is re-asserted here, not assumed.** The refusal
    handler is exactly where WR-03's swallow lived, and D-29 restructures it —
    so the last clause of this test drives a ``StoreContainmentError`` through
    both forms and asserts nothing was removed. A containment violation is
    evidence about the *environment*; the store must be left intact for the
    caller to inspect.
    """
    store = _seed_offloaded(make_store, tmp_cache_dir, ["gone", "kept"])
    _unlink_artefacts(tmp_cache_dir, "gone")
    assert sorted(store.keys()) == ["gone", "kept"], (
        "the fixture did not track both keys, so nothing below proves anything"
    )

    # 1. The bare form: raises, and still removes the key that was there.
    with pytest.raises(KeyError):
        store.pop("gone")
    assert "gone" not in store, (
        "the BARE pop raised and left the key tracked; a route that cannot honour a key must not "
        "also leave the caller holding it (D-29, WR-02a)"
    )

    # 2. The defaulting form: returns the default, and still removes.
    store2 = _seed_offloaded(make_store, tmp_cache_dir, ["gone2"])
    _unlink_artefacts(tmp_cache_dir, "gone2")
    sentinel = object()
    assert store2.pop("gone2", sentinel) is sentinel, "the defaulting pop did not answer with its default"
    assert "gone2" not in store2, (
        "the DEFAULTING pop returned its default and left the key tracked — `dict.pop(k, d)` removes "
        "`k` when present, and a removal loop built on this one removes nothing (D-29, WR-02a)"
    )

    # 3. A genuinely absent key is unchanged on both forms: the fix must
    #    distinguish "present but unreadable" from "not there", not collapse
    #    them the other way.
    assert store2.pop("never_existed", sentinel) is sentinel
    with pytest.raises(KeyError):
        store2.pop("never_existed")

    # 4. Containment evidence is never degraded, and nothing is removed when it
    #    propagates (WR-03, in the handler D-29 restructures).
    #
    #    The Sneaky key is installed as *tracked but not resident* — the state
    #    the reopen rescan produces — because that is the only state in which
    #    the new removal clause could fire on it. A resident entry short-circuits
    #    in `_store.get` and never reaches the builder that refuses; an untracked
    #    key reaches the builder but has nothing to remove. Only this state puts
    #    the containment re-raise and the removal in the same call.
    store3 = make_store(tmp_cache_dir / "containment")
    store3._store[Sneaky("safe")] = None
    for call in (lambda: store3.pop(Sneaky("safe")), lambda: store3.pop(Sneaky("safe"), sentinel)):
        with pytest.raises(StoreContainmentError, match=re.escape(paths_mod.CLAUSE_ESCAPES)):
            call()
        assert store3.keys() == ["safe"], (
            f"a containment violation removed the key (store is now {store3.keys()}); it is evidence "
            "about the environment, not about the key, and the store must be left intact for the "
            "caller to inspect (WR-03)"
        )


def test_route_pop_of_an_offloaded_entry_leaves_its_artefacts_and_the_key_is_re_adopted(
    make_store: MakeStoreFn, tmp_cache_dir: Path
) -> None:
    """Plan 14-17 / D-29 / § WR-02b / T-14-77 — **a characterization test.**

    **This pins a known limitation, not a desired property.** Read that
    sentence before changing anything it asserts. A ``pop`` that succeeds
    removes the key from tracking and **leaves the codec pair on disk**, so the
    very next subscript re-adopts the key and so does the reopen rescan. That
    leaves ``key not in store`` true while ``store[key]`` succeeds, which is a
    ``Mapping`` contract violation. Measured at the post-14-16 tip::

        popped         : DiskBackedNDArray
        keys after pop : []      'a' in s: False
        files on disk  : ['a.dat', 'a.meta.json', 'a.npy']
        RESURRECTED    : DiskBackedNDArray   keys now: ['a']
        reopen adopts  : ['a']

    **Why it is deliberately not fixed here.** Adding an unlink to
    ``__delitem__`` would collapse **STORE-05**, which requires that *where the
    data lives* (offload), *whether the key is tracked* (``__delitem__``,
    in-memory only) and *whether the file outlives the object*
    (``purge_disk_on_gc``) stay three separate axes and are characterization-
    tested — this test is one of those characterizations. And the combined
    drop-key-and-delete-files operation is **STORE-04**'s, which additionally
    requires atomicity, no partial effect on refusal and no stale finalizer able
    to delete a later entry created under the same key. Neither is in this
    phase, and a removal route that half-implemented STORE-04 here would be
    harder to make atomic later than one that does not touch disk at all.

    A test that merely records a bug without saying it is one is how a bug
    becomes a feature; this docstring is the part that keeps that from
    happening.
    """
    store = _seed_offloaded(make_store, tmp_cache_dir, ["a"])
    assert store.keys() == ["a"] and dict(store.store) == {"a": None}, "the fixture did not produce an offloaded entry"

    popped = store.pop("a")
    assert popped is not None, "pop did not return the entry"
    assert "a" not in store and store.keys() == [], "pop did not drop the tracking entry"

    remaining = sorted(p.name for p in tmp_cache_dir.iterdir())
    assert remaining == ["a.dat", "a.meta.json", "a.npy"], (
        f"the on-disk artefacts changed: {remaining}. If an unlink was added to a removal route, it "
        "collapsed STORE-05's three axes and pre-empted STORE-04's atomic primitive — revert it and "
        "take the change to Phase 15"
    )

    revived = store["a"]
    assert revived is not None, "the subscript did not re-adopt the popped key"
    assert store.keys() == ["a"], "the key is re-adopted by the subscript — this is the documented limitation"

    del store
    reopened = make_store(tmp_cache_dir)
    assert reopened.keys() == ["a"], "the reopen rescan also re-adopts the popped key — the second half of WR-02b"


def test_route_clear_cannot_return_with_the_store_non_empty(make_store: MakeStoreFn, tmp_cache_dir: Path) -> None:
    """Plan 14-17 / STORE-01 / D-29 / § WR-04 / T-14-75.

    **A ``clear()`` that returns normally with the store still populated is a
    data-integrity defect, and it shipped.** ``MutableMapping.clear`` is
    ``while True: self.popitem()`` under ``except KeyError: pass``, and
    ``popitem`` reaches the validating subscript — so a tracked key whose
    payload cannot be loaded raises ``KeyError``, which the handler reads as
    *the mapping is empty* and stops on. Measured at the post-14-16 tip over
    four tracked keys, one of them unloadable::

        before clear : ['a', 'ghost', 'b', 'c']
        clear raised : (nothing)
        after clear  : ['ghost', 'b', 'c']

    One key of four dropped, no exception, and the caller told it succeeded.

    **Emptiness is asserted, not a count.** A partial clear that happened to
    drop the right *number* of keys would satisfy an arithmetic assertion, and
    the failure mode here is precisely a partial clear.

    **The unloadable key is placed after a loadable one on purpose.** If it
    came first the inherited implementation would drop nothing, and a test that
    only distinguishes "cleared nothing" from "cleared everything" cannot see a
    clear that stops halfway — which is the shape that actually shipped.
    """
    store = _seed_offloaded(make_store, tmp_cache_dir, ["a", "ghost", "b", "c"])
    _unlink_artefacts(tmp_cache_dir, "ghost")
    before = sorted(store.keys())
    assert before == ["a", "b", "c", "ghost"], "the fixture did not track all four keys"

    store.clear()

    assert store.keys() == [], (
        f"clear() returned with the store non-empty: before={before}, after={sorted(store.keys())}. "
        "A route that cannot honour a key must raise, not drop part of the work and report success "
        "(D-29, WR-04)"
    )

    # Idempotency: a second clear, and a clear on a store that was never
    # populated, are both no-ops rather than raises.
    store.clear()
    assert store.keys() == [], "the second clear() was not a no-op"
    empty = make_store(tmp_cache_dir / "never-populated")
    empty.clear()
    assert empty.keys() == [], "clear() on an already-empty store did not leave it empty"


def test_route_popitem_is_left_inherited_and_its_key_error_carries_the_key(
    make_store: MakeStoreFn, tmp_cache_dir: Path
) -> None:
    r"""Plan 14-17 / D-29 / § WR-04 (``popitem`` half) — **decided by measurement.**

    § WR-04 proposed overriding ``popitem`` alongside ``clear``. Re-measured at
    the post-14-16 tip, its two ``KeyError``\\ s are **distinguishable**::

        popitem() on a NON-EMPTY store, payload unreadable : KeyError args=('ghost',)
        popitem() on an EMPTY store                        : KeyError args=()

    ``MutableMapping.popitem`` raises a bare ``KeyError`` from ``next(iter())``
    when the mapping is empty and propagates ``self[key]``'s ``KeyError(key)``
    otherwise. The information ``clear`` needed was therefore always present;
    ``except KeyError: pass`` discarded it. **The defect was in the handler, not
    in the signal**, and with ``clear`` overridden nothing routes through
    ``popitem`` any more.

    **Three override shapes were considered and all are worse**, which is why
    this is a decision rather than an oversight — see ``clear``'s docstring for
    the same argument in the module. This test exists so that a reader who finds
    ``clear`` overridden and ``popitem`` not can tell the two apart by grepping
    the suite, exactly as ``setdefault``'s test does for ``pop``.

    The reviewer's own caveat also reproduces and is asserted below: an
    already-installed illegal key makes ``popitem`` raise ``StoreKeyError``
    rather than misbehave, which is defensible and is left alone.
    """
    store = _seed_offloaded(make_store, tmp_cache_dir, ["ghost"])
    _unlink_artefacts(tmp_cache_dir, "ghost")

    with pytest.raises(KeyError) as unloadable:
        store.popitem()
    assert unloadable.value.args == ("ghost",), (
        f"popitem's KeyError on a NON-EMPTY store no longer carries the key (args="
        f"{unloadable.value.args!r}); that is the only thing distinguishing it from the empty-store "
        "KeyError, and clear()'s round-3 defect was reading the two as the same event"
    )
    assert store.keys() == ["ghost"], "the inherited popitem removed a key the caller never named"

    empty = make_store(tmp_cache_dir / "empty-store")
    with pytest.raises(KeyError) as exhausted:
        empty.popitem()
    assert exhausted.value.args == (), (
        f"popitem's KeyError on an EMPTY store now carries arguments ({exhausted.value.args!r}); the "
        "two cases have stopped being distinguishable"
    )

    # The reviewer's caveat, reproduced: an already-installed illegal key is
    # refused by the subscript's lexical layer rather than silently popped.
    installed = make_store(tmp_cache_dir / "installed-illegal")
    installed._store[ILLEGAL_KEY] = _entry()
    with pytest.raises(StoreKeyError, match="Invalid store key"):
        installed.popitem()


# ---------------------------------------------------------------------------
# route_rescan — warn and skip, leave the file alone (D-09)
# ---------------------------------------------------------------------------


def test_route_rescan_warns_and_skips_an_illegal_stem_without_raising(
    tmp_cache_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    r"""Plan 14-03 / STORE-01 / D-09.

    Opening a cache directory that holds a now-illegal stem must never become a
    crash — this route sees pre-existing data, not a caller mistake. The
    refused stems are absent from the store, the legal one is still adopted,
    both refused files are untouched on disk, and a WARNING names each refused
    stem together with the cache directory.

    Two refused stems are planted so that two *different* clauses are covered:
    ``..npy`` has stem ``'.'`` (the reserved clause — the free case a pre-fix
    directory produces on its own) and ``evil\..\x.npy`` has stem
    ``evil\..\x`` (the separator clause under Windows semantics, which is a
    perfectly legal POSIX filename and therefore genuinely plantable here).

    Note the plan named ``bad..npy`` as the second planted file. Its stem is
    ``bad.``, which D-06 makes **legal** — ``foo.`` passes — so it would have
    exercised the adoption path rather than the refusal path. Replaced with a
    stem that is genuinely refused, and by a different clause.

    The assertion is on ``record.getMessage()`` — the *rendered* message —
    rather than on ``record.args``. The module's house style is lazy ``%``
    interpolation and stays so, but ``caplog`` renders either style, so keying
    the assertion to ``args`` would pin the formatting idiom instead of the
    behaviour the requirement is about. D-13 asks for ``repr(key)``, so the
    stem appears in the message in its ``repr`` form.
    """
    reserved_stem_file = tmp_cache_dir / "..npy"
    separator_stem_file = tmp_cache_dir / "evil\\..\\x.npy"
    legal_file = tmp_cache_dir / "good.npy"
    for planted in (reserved_stem_file, separator_stem_file, legal_file):
        planted.write_bytes(b"not-a-real-npy")

    with caplog.at_level("WARNING"):
        store = _store_with_cache_path(tmp_cache_dir)

    assert "." not in store.keys(), "the reserved stem was adopted as a live key"
    assert "evil\\..\\x" not in store.keys(), "the separator-bearing stem was adopted as a live key"
    assert "good" in store.keys(), "the rescan skipped a perfectly legal stem"

    assert reserved_stem_file.exists(), "the rescan deleted a file it should only skip"
    assert separator_stem_file.exists(), "the rescan deleted a file it should only skip"

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    for stem in (".", "evil\\..\\x"):
        assert any(repr(stem) in r.getMessage() and str(tmp_cache_dir) in r.getMessage() for r in warnings), (
            f"no WARNING named the refused stem {stem!r} together with the cache directory"
        )


# ---------------------------------------------------------------------------
# route_setstate — raise, never a silently short store (D-10)
# ---------------------------------------------------------------------------


def test_route_setstate_raises_rather_than_returning_a_short_store(
    make_store: MakeStoreFn, tmp_cache_dir: Path
) -> None:
    """Plan 14-03 / STORE-01 / D-10.

    Unpickling is a trust boundary, and a post-fix pickle cannot legitimately
    carry an illegal key — so an illegal key arriving here means a legacy or a
    tampered pickle.

    ↻ CORRECTED by Plan 14-08 (CR-02). This docstring used to locate that
    guarantee in ``__getstate__``, claiming it "snapshots a ``_store`` the
    parent already validated". It does not: it offloads when caching is enabled
    and then copies ``__dict__`` verbatim, validating nothing. The guarantee
    lives in the *write* routes — the five rows above — plus D-19's read-only
    ``store`` view, which together are why an illegal key cannot enter
    ``_store`` after construction. Same false premise as the one corrected in
    ``__setstate__``'s own Notes block; do not restore either.

    The negative half is the point. Warning-and-skipping instead — the policy
    the ``__init__`` rescan deliberately uses — would hand back a store
    silently missing entries: data loss disguised as success, inside a worker.
    So the state below carries one legal key alongside the illegal one, and the
    test asserts the call *raised* rather than quietly reconstructing a store
    holding only the legal key.
    """
    donor = make_store(tmp_cache_dir)
    victim = make_store(tmp_cache_dir)

    state: dict[str, Any] = dict(donor.__dict__)
    state["_store"] = {"legal_key": None, ILLEGAL_KEY: None}

    with pytest.raises(StoreKeyError, match="Invalid store key"):
        victim.__setstate__(state)

    assert ILLEGAL_KEY not in victim.store, "the illegal key survived into the restored store"
    assert "legal_key" not in victim.store, (
        "__setstate__ reconstructed a partial store instead of raising — a silently short store "
        "is the failure D-10 exists to prevent"
    )


# ---------------------------------------------------------------------------
# setitem_no_syscall — SC-1's worker-safety clause
# ---------------------------------------------------------------------------


def test_setitem_no_syscall_a_legal_insert_survives_a_hostile_filesystem(
    make_store: MakeStoreFn, tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 14-03 / STORE-01 / SC-1.

    ``__setitem__`` must perform no filesystem syscall, so it stays safe to
    call inside a ``loky`` worker whose mount namespace may not even contain
    the cache directory. Proven by injection rather than by inspection:
    ``Path.resolve``, ``Path.exists`` and ``os.stat`` are all replaced with
    callables that raise, and a **legal** insert must still succeed.

    This is the test that stops a future "just call ``resolve`` here too" from
    silently making ``__setitem__`` unsafe in a worker. The resolved
    containment layer belongs in the path builders, which is the whole reason
    the lexical and containment layers are separate.

    The store and the entry are both constructed *before* the patch, since
    their own construction legitimately touches disk.
    """
    store = make_store(tmp_cache_dir)
    entry = _entry()

    real_resolve = Path.resolve
    real_exists = Path.exists
    real_stat = os.stat

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("__setitem__ performed a filesystem syscall; SC-1 forbids it on this route")

    monkeypatch.setattr(Path, "resolve", boom)
    monkeypatch.setattr(Path, "exists", boom)
    monkeypatch.setattr(os, "stat", boom)
    try:
        store["legal_key"] = entry
    finally:
        # explicitly (belt-and-suspenders; monkeypatch teardown does this too)
        monkeypatch.setattr(Path, "resolve", real_resolve)
        monkeypatch.setattr(Path, "exists", real_exists)
        monkeypatch.setattr(os, "stat", real_stat)

    assert store.keys() == ["legal_key"], "the legal insert did not land"


# ===========================================================================
# Plan 14-06 / STORE-02 — the mechanism proofs
# ===========================================================================
#
# Everything above pins the *lexical* layer at each route. Everything below
# pins the *containment* layer at each path builder, and pins it by mechanism
# rather than by outcome. An outcome-only test ("an escaping key raises")
# passes against a string prefix test on whichever fixture happens to be
# chosen — which is the whole reason SC-3 asks for the three wrong primitives
# to be asserted wrong explicitly.


def _small_array() -> np.ndarray:
    """Return a fresh small float32 payload; fresh per call, so no test shares state."""
    return np.array([[1.0, 2.0, 3.0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# bypass — SC-3: three wrong primitives, asserted wrong; one right check
# ---------------------------------------------------------------------------

#: A containment primitive under test: does it *claim* ``candidate`` is inside
#: ``cache_dir``?
Primitive = Callable[[Path, Path], bool]

#: Builds ``(cache_dir, candidate)`` for one crafted escape, under ``tmp_path``.
CraftedCase = Callable[[Path], tuple[Path, Path]]


def _string_prefix_says_contained(cache_dir: Path, candidate: Path) -> bool:
    """Rejected primitive 1: ``str(candidate).startswith(str(cache_dir))``."""
    return str(candidate).startswith(str(cache_dir))


def _bare_is_relative_to_says_contained(cache_dir: Path, candidate: Path) -> bool:
    """Rejected primitive 2: ``is_relative_to`` on an *unresolved* candidate."""
    return candidate.is_relative_to(cache_dir)


def _bare_common_path_says_contained(cache_dir: Path, candidate: Path) -> bool:
    """Rejected primitive 3: a bare ``os.path.commonpath`` computation."""
    return os.path.commonpath([str(cache_dir), str(candidate)]) == str(cache_dir)


def _craft_sibling_sharing_the_name_prefix(tmp_path: Path) -> tuple[Path, Path]:
    """Craft ``/…/cache-evil/x.npy`` against ``/…/cache`` — defeats the string test."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sibling = tmp_path / "cache-evil"
    sibling.mkdir(parents=True, exist_ok=True)
    return cache_dir, sibling / "x.npy"


def _craft_unresolved_parent_segment(tmp_path: Path) -> tuple[Path, Path]:
    """Craft ``/…/cache/../victim.npy`` — defeats both purely lexical primitives."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir, cache_dir / ".." / "victim.npy"


@pytest.mark.parametrize(
    ("primitive", "crafted"),
    [
        pytest.param(_string_prefix_says_contained, _craft_sibling_sharing_the_name_prefix, id="string-prefix-test"),
        pytest.param(_bare_is_relative_to_says_contained, _craft_unresolved_parent_segment, id="bare-is-relative-to"),
        pytest.param(_bare_common_path_says_contained, _craft_unresolved_parent_segment, id="bare-common-path"),
    ],
)
def test_bypass_each_obvious_primitive_accepts_an_escape_the_shipped_check_refuses(
    primitive: Primitive, crafted: CraftedCase, tmp_path: Path
) -> None:
    """Plan 14-06 / STORE-02 / SC-3 / T-14-16.

    A table of why each obvious containment primitive was rejected. Each of the
    three is a **complete** bypass, not a partial one, and all three fail in the
    direction of *accepting* an escape:

    * a **string prefix test** accepts a sibling directory that merely shares
      the cache directory's name prefix — containment is a path relationship,
      not a string relationship;
    * a **bare** ``is_relative_to`` on an unresolved path accepts a ``..``
      segment, because it is purely lexical and never collapses the segment;
    * a **bare** ``os.path.commonpath`` has the identical blind spot, for the
      identical reason.

    The wrong answers are asserted **positively** rather than marked
    expected-to-fail. An ``xfail`` marker would let a primitive silently start
    behaving differently — including *correctly*, which would quietly remove
    this group's justification for the extra ``resolve`` call — without anything
    noticing. Asserted positively, the group reads as documentation and a future
    "simplification" of :func:`paths._assert_contained` fails a test rather than
    passing a review.
    """
    cache_dir, candidate = crafted(tmp_path)

    assert primitive(cache_dir, candidate) is True, (
        f"the rejected primitive no longer accepts the crafted escape {str(candidate)!r} against "
        f"{str(cache_dir)!r}; if it genuinely became correct, retire the case rather than the group"
    )

    with pytest.raises(StoreContainmentError, match=paths_mod.CLAUSE_ESCAPES):
        paths_mod._assert_contained(cache_dir, candidate)


# ---------------------------------------------------------------------------
# every_builder — SC-4's builder clause, driven off the registry (T-14-19)
# ---------------------------------------------------------------------------

#: Every module-level path builder ``paths.py`` publishes, discovered rather
#: than listed. ``_build`` is private and deliberately unregistered, so the
#: naming convention (``get_*_path``) is what defines "a builder" here.
PUBLISHED_BUILDER_NAMES = {
    name
    for name, obj in vars(paths_mod).items()
    if name.startswith("get_") and name.endswith("_path") and callable(obj)
}

#: The registry, as parametrisation cases. Sorted for a stable report order
#: under ``pytest-randomly``; the key function keeps the sort off the callables.
BUILDER_CASES = [
    pytest.param(name, builder, id=name)
    for name, builder in sorted(paths_mod.STORE_PATH_BUILDERS.items(), key=lambda item: item[0])
]


@pytest.mark.parametrize(("builder_name", "builder"), BUILDER_CASES)
def test_every_builder_contains_a_legal_key_and_refuses_an_escaping_one(
    builder_name: str, builder: Callable[[Path, str], Path], tmp_cache_dir: Path
) -> None:
    """Plan 14-06 / STORE-02 / SC-4 / T-14-19.

    Every builder puts a legal key's path inside the cache directory and
    refuses an escaping key — **iterated off** ``STORE_PATH_BUILDERS`` rather
    than hand-listed. A hand-written five-case version satisfies the letter of
    SC-4 today and silently loses it at the next addition: a sixth builder added
    in Phase 15 would simply not appear.

    The registry-membership assertion is what gives that its teeth. It ties the
    registry to the set of ``get_*_path`` callables the module actually
    publishes, so an unregistered sixth builder fails this group **by
    omission** — the parametrisation cannot grow to cover it, so the set
    comparison is the thing that notices.

    The containment assertion is on the *parent*, resolved: that is the property
    :func:`paths._assert_contained` actually enforces (D-17), and asserting on
    the fully-resolved path instead would encode the policy this phase
    deliberately did not adopt.
    """
    assert set(paths_mod.STORE_PATH_BUILDERS) == PUBLISHED_BUILDER_NAMES, (
        "STORE_PATH_BUILDERS has drifted from the set of get_*_path builders the module publishes: "
        f"registered={sorted(paths_mod.STORE_PATH_BUILDERS)} published={sorted(PUBLISHED_BUILDER_NAMES)}. "
        "A builder outside the registry is a builder outside this contract test."
    )
    assert len(paths_mod.STORE_PATH_BUILDERS) == len(PUBLISHED_BUILDER_NAMES)

    legal = builder(tmp_cache_dir, "legal_key")
    assert legal.parent.resolve() == tmp_cache_dir.resolve(), (
        f"{builder_name} returned {str(legal)!r}, whose parent is not the cache directory"
    )
    assert legal.name.startswith("legal_key"), f"{builder_name} did not build its path from the key"

    with pytest.raises(StoreKeyError):
        builder(tmp_cache_dir, ILLEGAL_KEY)


# ---------------------------------------------------------------------------
# layer_order — the lexical layer fires first, at every builder (D-03)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("builder_name", "builder"), BUILDER_CASES)
def test_layer_order_a_doubly_illegal_key_raises_the_base_type_at_each_registered_builder(
    builder_name: str, builder: Callable[[Path, str], Path], tmp_cache_dir: Path
) -> None:
    """Plan 14-06 / STORE-02 / the lexical-then-contained ordering.

    ``'../victim'`` is illegal under **both** layers — a separator violation
    lexically, and it would resolve outside the cache directory too — so it is
    the key that discriminates between the two orderings. With the lexical layer
    first it raises the base :exc:`StoreKeyError`; inverted, it would raise the
    :exc:`StoreContainmentError` subclass.

    Asserted with ``type(exc) is StoreKeyError`` rather than ``isinstance``.
    ``StoreContainmentError`` *is* a ``StoreKeyError``, so an ``isinstance``
    assertion accepts the inverted order and cannot detect the very thing this
    test exists to detect.

    Why the ordering is load-bearing rather than cosmetic: it is what lets a
    downstream handler read the exception **type** as evidence of which layer
    fired — a bad key from the caller, versus a symlink planted in the cache
    directory. Inverted, every bad key would look like an attack.
    """
    with pytest.raises(StoreKeyError) as excinfo:
        builder(tmp_cache_dir, ILLEGAL_KEY)

    assert type(excinfo.value) is StoreKeyError, (
        f"{builder_name} raised {type(excinfo.value).__name__} for a key that is illegal under both "
        "layers; the containment layer ran before the lexical one, so the exception type no longer "
        "says which layer fired"
    )


# ---------------------------------------------------------------------------
# base_not_pickled — the resolved base never becomes state (D-03 / T-14-17)
# ---------------------------------------------------------------------------


def test_base_not_pickled_no_attribute_appears_across_a_build_or_a_pickle_round_trip(
    make_store: MakeStoreFn, tmp_cache_dir: Path
) -> None:
    """Plan 14-06 / STORE-02 / D-03 / T-14-17.

    The containment base is ``cache_dir.resolve()``, computed inside
    :func:`paths._assert_contained` on every call. It must never be memoised
    onto the instance: :meth:`DiskBackedStore.__getstate__` is
    ``self.__dict__.copy()``, so a stored resolved base would be pickled
    verbatim and travel into a worker that may be on a different mount
    namespace, where it would be compared against paths that genuinely differ.

    Made observable in two halves. First, the store's *attribute set* is
    snapshotted across an offload — which drives all five builders — and must be
    unchanged. Second, a pickle round-trip must not revive an instance carrying
    an attribute the original lacked, which is the failure the first half would
    miss if the memoisation happened in ``__getstate__`` itself.
    """
    store = make_store(tmp_cache_dir)
    attributes_before = set(store.__dict__)

    store.add_data_to_store("k0", _small_array())
    store.offload(pickle_container=True)

    assert set(store.__dict__) == attributes_before, (
        "driving a path build added instance state: "
        f"{sorted(set(store.__dict__) - attributes_before)}. If that is the resolved containment "
        "base, it will be pickled into a worker (D-03)."
    )

    revived: DiskBackedStore[DiskBackedNDArray] = pickle.loads(pickle.dumps(store))
    assert set(revived.__dict__) - attributes_before == set(), (
        f"the revived store carries state the original lacked: {sorted(set(revived.__dict__) - attributes_before)}"
    )
    np.testing.assert_array_equal(np.asarray(revived["k0"]), _small_array())


# ---------------------------------------------------------------------------
# json_tmp_routed — both temporary names come from the seam (SC-4 / T-14-15)
# ---------------------------------------------------------------------------


def test_json_tmp_routed_both_temporary_names_come_from_the_shared_builders(
    make_store: MakeStoreFn, tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 14-06 / STORE-02 / SC-4 / T-14-15.

    ``.meta.json.tmp`` — SC-4's named target — used to be a bare
    ``self._cache_dir / f"{key}.meta.json.tmp"`` join, safe only *incidentally*
    because the real ``.meta.json`` builder ran three lines earlier and would
    already have raised. Both temporary names must now come from the shared
    builders, so the property is an invariant rather than a filter.

    Proven by **recorders**, not by filenames. Asserting on the resulting
    ``.tmp`` names would pass just as well against an inline construction that
    happens to produce the same string, which is exactly the incidental-safety
    shape SC-4 rejects.

    The patch target is the attribute on the shared ``paths`` **module**.
    ``disk_backed_store`` pins ``from . import paths`` with qualified call
    sites, so it resolves the attribute off the module object at call time and
    sees the patch. The inverse pairing is the failure this is written against:
    a name-level import in the store plus a patch on the module leaves the
    store's bound copy live, the recorders never fire, and the group reports a
    **vacuous pass** over a store that never touched the seam. The
    ``== ["k0"]`` assertions (not ``>= 0``) are what make that failure loud.
    """
    store = make_store(tmp_cache_dir)
    store.add_data_to_store("k0", _small_array())

    real_npy_tmp = paths_mod.get_npy_tmp_path
    real_meta_tmp = paths_mod.get_meta_tmp_path
    npy_tmp_calls: list[str] = []
    meta_tmp_calls: list[str] = []

    def recording_npy_tmp(cache_dir: Path, key: str) -> Path:
        npy_tmp_calls.append(key)
        return real_npy_tmp(cache_dir, key)

    def recording_meta_tmp(cache_dir: Path, key: str) -> Path:
        meta_tmp_calls.append(key)
        return real_meta_tmp(cache_dir, key)

    monkeypatch.setattr("GSEGUtils.lazy_disk_cache.paths.get_npy_tmp_path", recording_npy_tmp)
    monkeypatch.setattr("GSEGUtils.lazy_disk_cache.paths.get_meta_tmp_path", recording_meta_tmp)
    try:
        store.offload(pickle_container=True)
    finally:
        # explicitly (belt-and-suspenders; monkeypatch teardown does this too)
        monkeypatch.setattr("GSEGUtils.lazy_disk_cache.paths.get_npy_tmp_path", real_npy_tmp)
        monkeypatch.setattr("GSEGUtils.lazy_disk_cache.paths.get_meta_tmp_path", real_meta_tmp)

    assert npy_tmp_calls == ["k0"], (
        f"the .npy.tmp name did not come from paths.get_npy_tmp_path (recorded {npy_tmp_calls!r}); "
        "an empty list means the patch did not intercept — suspect the import binding in "
        "disk_backed_store.py before suspecting the routing"
    )
    assert meta_tmp_calls == ["k0"], (
        f"the .meta.json.tmp name did not come from paths.get_meta_tmp_path (recorded "
        f"{meta_tmp_calls!r}); this is the name SC-4 calls out by name"
    )
    assert (tmp_cache_dir / "k0.npy").exists(), "the offload did not actually write the codec pair"
    assert (tmp_cache_dir / "k0.meta.json").exists(), "the offload did not actually write the codec pair"


# ===========================================================================
# Plan 14-06 / STORE-03 — the symlink group
# ===========================================================================
#
# STORE-03 is the requirement that fails INVISIBLY on the CI runner. Its
# ``/tmp`` is a real directory, exactly like this host's, so the break that is
# live on macOS (``mkdtemp`` under ``/var`` → ``/private/var``) and on ETH
# ``/scratch`` is undetectable here unless the symlink is built explicitly.
# Everything below therefore runs against the ``symlinked_cache_dir`` fixture,
# which calls ``.symlink_to()`` itself — and the first group is the one that
# proves it still does.


# ---------------------------------------------------------------------------
# fixture_is_really_symlinked — the guard the other four depend on for meaning
# ---------------------------------------------------------------------------


def test_fixture_is_really_symlinked_or_the_whole_group_tests_nothing(symlinked_cache_dir: Path) -> None:
    """Plan 14-06 / STORE-03 / T-14-20 (test-validity guard).

    Read this group's other four tests as conditional on this one. The ambient
    temporary directory is a **real** directory on this host and on the CI
    runner, so a fixture that quietly stopped creating a link would leave every
    remaining STORE-03 test passing while testing nothing at all — a green suite
    proving the opposite of what it claims.

    Any test named for symlinks that does not itself create one is the warning
    sign; this is the assertion that turns that warning into a failure.
    """
    assert symlinked_cache_dir.is_symlink(), (
        f"the symlinked_cache_dir fixture returned {str(symlinked_cache_dir)!r}, which is not a "
        "symlink. Every other test in this group is now exercising the ordinary real-directory "
        "path while claiming to cover STORE-03 — restore the fixture's .symlink_to() call."
    )
    assert symlinked_cache_dir.resolve() != symlinked_cache_dir, (
        "the fixture is a symlink whose resolved form equals itself, so it points at itself rather "
        "than at a separate real directory; the symlinked-base code path is not being exercised"
    )


# ---------------------------------------------------------------------------
# symlinked_cache_dir — the round-trip (SC-2 / T-14-20)
# ---------------------------------------------------------------------------


def test_symlinked_cache_dir_round_trips_through_a_fresh_store(
    make_store: MakeStoreFn, symlinked_cache_dir: Path
) -> None:
    """Plan 14-06 / STORE-03 / SC-2 / T-14-20.

    Write, offload with the container pickled, reopen and read back — with the
    store constructed on the **link**, never on its target. Constructing on the
    target would make this pass without exercising the symlinked-base path at
    all, which is the same degradation the fixture guard above catches one level
    up.

    This is the single test that would have caught the macOS and ETH
    ``/scratch`` break on a Linux runner. Comparing a candidate against an
    *unresolved* base refuses **every legitimate key** when the cache directory
    is itself a symlink, and that configuration is the default under ``mkdtemp``
    on macOS and normal on ``/scratch``. The failure mode is an availability
    regression, not a security one — which is why it needs a round-trip rather
    than a refusal assertion.
    """
    expected = _small_array()
    writer = make_store(symlinked_cache_dir)
    writer.add_data_to_store("k0", expected)
    writer.offload(pickle_container=True)

    assert (symlinked_cache_dir / "k0.npy").exists(), "the offload wrote nothing through the symlinked cache dir"
    assert (symlinked_cache_dir / "k0.meta.json").exists(), "the sidecar is missing under the symlinked cache dir"

    reader = make_store(symlinked_cache_dir)
    assert "k0" in reader.keys(), "the rescan did not adopt the entry written through the symlinked cache dir"
    np.testing.assert_array_equal(np.asarray(reader["k0"]), expected)


# ---------------------------------------------------------------------------
# adopted_symlink — the legitimate case full resolution would refuse (D-17)
# ---------------------------------------------------------------------------


def test_adopted_symlink_loads_through_both_final_component_links(
    make_store: MakeStoreFn, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Plan 14-06 / STORE-03 / SC-2 / D-17.

    A real codec pair lives *outside* the cache directory and is adopted into it
    through **one final-component symlink per member of the pair**. The entry
    must load, containment must accept both paths, and the array read back must
    equal the array written outside. Full resolution refuses this case, which is
    why the policy is parent-only — and this test is the only thing that says
    so.

    **Both symlinks, not one.** ``_load_entry`` raises ``KeyError(key)`` unless
    the ``.npy`` **and** the ``.meta.json`` both ``exists()``, and only past
    that check does it open the sidecar and load the array. A fixture that
    symlinked only the array would therefore take the cache-**miss** branch, and
    a test asserting merely "it did not blow up" would pass on the miss path
    while claiming to demonstrate STORE-03 — retiring a success criterion
    without exercising it. (``exists()`` follows symlinks, so a dangling link
    fails the same way; the outside pair is built first.)

    **Content equality, not just the absence of an exception.** With both files
    present the load path is real, so asserting the values is what proves bytes
    travelled *through* the symlinks. Without it, a future ``KeyError``-
    swallowing change would leave this test green. Delete the sidecar link and
    this test must go red with a ``KeyError``; a test that cannot fail is not
    evidence.
    """
    expected = _small_array()
    outside = tmp_path / "outside"
    outside.mkdir(parents=True, exist_ok=True)

    donor = make_store(outside)
    donor.add_data_to_store("adopted", expected)
    donor.offload(pickle_container=True)
    outside_npy = outside / "adopted.npy"
    outside_meta = outside / "adopted.meta.json"
    assert outside_npy.exists() and outside_meta.exists(), "the donor store did not write a real codec pair outside"

    adopted_npy = tmp_cache_dir / "adopted.npy"
    adopted_meta = tmp_cache_dir / "adopted.meta.json"
    adopted_npy.symlink_to(outside_npy)
    adopted_meta.symlink_to(outside_meta)
    assert adopted_npy.is_symlink() and adopted_meta.is_symlink(), "the adoption fixture did not create two symlinks"

    # Containment accepts both — parent-only resolution, so the final component
    # being a link to elsewhere is not an escape.
    assert paths_mod.get_npy_path(tmp_cache_dir, "adopted") == adopted_npy
    assert paths_mod.get_meta_path(tmp_cache_dir, "adopted") == adopted_meta

    store = make_store(tmp_cache_dir)
    assert "adopted" in store.keys(), "the rescan did not adopt the symlinked entry"
    np.testing.assert_array_equal(np.asarray(store["adopted"]), expected)


# ---------------------------------------------------------------------------
# parent_only_controls — tolerance is discrimination, not loosening (T-14-18)
# ---------------------------------------------------------------------------


def _control_symlinked_subdirectory_inside_the_cache(cache_dir: Path, outside: Path) -> Path:
    """Build ``<cache>/sub/x.npy`` where ``sub`` is a symlink pointing outside."""
    sub = cache_dir / "sub"
    sub.symlink_to(outside, target_is_directory=True)
    return sub / "x.npy"


def _control_parent_directory_segment(cache_dir: Path, outside: Path) -> Path:
    """Build the path a parent-directory key would produce: ``<cache>/../victim.npy``.

    ``outside`` is deliberately unused (WR-09): this control escapes *upward*
    and needs no directory outside the cache to point at, while its sibling
    does. The parameter is present only so both controls share one callable
    shape and can be parametrised together — do not delete it, and do not
    invent a use for it.
    """
    return cache_dir / ".." / "victim.npy"


@pytest.mark.parametrize(
    "control",
    [
        pytest.param(_control_symlinked_subdirectory_inside_the_cache, id="symlinked-sub-directory"),
        pytest.param(_control_parent_directory_segment, id="parent-directory-segment"),
    ],
)
def test_parent_only_controls_still_refuse_what_they_always_refused(
    control: Callable[[Path, Path], Path], tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Plan 14-06 / STORE-03 / D-17 / T-14-18.

    Without these two controls the adopted-symlink test above reads as evidence
    that containment was **loosened**. With them it reads as evidence that
    containment **discriminates**: parent-only resolution accepts a
    final-component link and still refuses a symlinked *directory* component and
    a parent-directory segment.

    The symlinked sub-directory is what bounds the accepted residual (D-17,
    T-14-18): the residual is the *leaf* case only. A directory component
    pointing outside is resolved away by ``candidate.parent.resolve()`` and
    caught, so "an attacker who can already write inside the cache directory can
    redirect one entry" does not widen into "…can redirect a whole tree".

    Both cases are asserted at the containment layer directly, because a key
    carrying either shape is refused one layer earlier by the lexical rule and
    would never reach here — which is exactly why the containment layer needs
    its own proof.
    """
    outside = tmp_path / "outside"
    outside.mkdir(parents=True, exist_ok=True)
    candidate = control(tmp_cache_dir, outside)

    with pytest.raises(StoreContainmentError, match=paths_mod.CLAUSE_ESCAPES):
        paths_mod._assert_contained(tmp_cache_dir, candidate)


# ---------------------------------------------------------------------------
# symlink_pickle_round_trip — the per-check base survives __getstate__ (D-03)
# ---------------------------------------------------------------------------


def test_symlink_pickle_round_trip_revives_its_entries(make_store: MakeStoreFn, symlinked_cache_dir: Path) -> None:
    """Plan 14-06 / STORE-03 / D-03.

    Guards the interaction between the per-check containment base and
    :meth:`DiskBackedStore.__getstate__`, which offloads *before* serialising:
    the offload writes through the symlinked directory, and the revived store
    must re-derive its base rather than inherit one. A memoised base would
    survive the pickle as a resolved real path while ``_cache_dir`` came back as
    the link, and the two would then disagree for every key.
    """
    expected = _small_array()
    store = make_store(symlinked_cache_dir)
    store.add_data_to_store("k0", expected)

    revived: DiskBackedStore[DiskBackedNDArray] = pickle.loads(pickle.dumps(store))

    assert revived.cache_dir == symlinked_cache_dir, "the revived store no longer points at the symlinked cache dir"
    assert "k0" in revived.keys(), "the revived store lost its entry"
    np.testing.assert_array_equal(np.asarray(revived["k0"]), expected)


# ---------------------------------------------------------------------------
# Plan 14-07 / STORE-07 — the published contract (D-07 / D-18)
# ---------------------------------------------------------------------------

#: The documentation page carrying the published store key contract. The scan
#: snippet below is *extracted from this file and executed*, not re-typed, so
#: the published snippet and the test that proves it correct cannot drift: an
#: edit to one is an edit to the other by construction. That is the whole point
#: — a snippet a consumer runs against their live cache directory is only worth
#: publishing if something asserts it still answers correctly.
CONTRACT_PAGE = Path(__file__).resolve().parents[1] / "docs" / "source" / "LazyDiskCache.rst"

#: The six names the package publishes for the key contract. Kept as a literal
#: tuple rather than derived from ``__all__``: this is the *specification* of
#: the published surface, and deriving it from the thing under test would make
#: the assertion vacuous.
PUBLISHED_SURFACE = (
    "StoreKeyError",
    "StoreContainmentError",
    "is_valid_store_key",
    "get_npy_path",
    "get_meta_path",
    "get_legacy_pickle_path",
)

#: Deliberately withheld (D-07): the raising validator, because the predicate is
#: the published half by decision, and the two temporary-name builders, which are
#: internal construction detail with no downstream caller. A name published by
#: accident in a package on production PyPI cannot be withdrawn without a break,
#: which is why the smaller surface is the one pinned here.
WITHHELD_NAMES = ("validate_store_key", "get_npy_tmp_path", "get_meta_tmp_path")

#: Two stems that are legal today and stay legal. The first is a real pc2img
#: feature name — parentheses, commas, hyphens and an exponent-shaped dot — and
#: is the reason the rule is a property denylist rather than a charset.
LEGAL_STEMS = ("rrim_pack_(range,r16,d8,z1e-05)", "foo.bar")

#: Three stems that used to work and are now refused, one per mechanism: a nested
#: key (the silent-leak bug fix), a single filename carrying backslashes (a
#: perfectly legal POSIX filename, refused under the Windows interpretation), and
#: the **empty** key.
#:
#: The empty stem is Plan 14-10 / WR-04 / D-22, and it does double duty: seeding
#: it writes ``<root>/.npy`` — the pre-fix artefact of the empty key, the file a
#: cache directory written before this phase can genuinely contain — and listing
#: it here is the expectation that the realigned snippet now reports it. Before
#: the realignment the snippet reported ``''`` for that file while the store
#: adopted ``'.npy'``: one byte on disk, two different keys, which is the
#: doc/code drift the snippet's whole design exists to prevent.
ILLEGAL_STEMS = ("tile_03/range", "evil\\..\\x", "")


def _published_namespace() -> dict[str, Any]:
    """Execute the documentation's scan snippet and return its namespace.

    Parses the contract page for the ``code-block:: python`` that defines
    ``refused_keys``, dedents it and executes it. Asserting that exactly one
    such block exists is load-bearing: if a second scan snippet is ever added,
    this test would otherwise silently exercise whichever came first.
    """
    page = CONTRACT_PAGE.read_text(encoding="utf-8")
    blocks = re.findall(r"\.\. code-block:: python\n\n((?:(?: {3}.*)?\n)+)", page)
    matching = [textwrap.dedent(block) for block in blocks if "def refused_keys" in block]
    assert len(matching) == 1, (
        f"expected exactly one scan snippet defining refused_keys in {CONTRACT_PAGE.name}, "
        f"found {len(matching)} — the doc and this test are no longer in step"
    )
    namespace: dict[str, Any] = {}
    exec(compile(matching[0], str(CONTRACT_PAGE), "exec"), namespace)
    return namespace


def _published_scan() -> Callable[[Path], list[str]]:
    """Return the ``refused_keys`` function exactly as the documentation publishes it."""
    return cast(Callable[[Path], list[str]], _published_namespace()["refused_keys"])


def _published_store_key_for() -> Callable[[Path, Path], str]:
    """Return the snippet's own key derivation, ``store_key_for``.

    Extracted from the *same* executed block as :func:`_published_scan`, so the
    derivation this file compares the rescan against is the derivation a
    consumer actually runs — not a re-typed copy of it. ``refused_keys`` alone
    could not serve: it reports only the keys it *refuses*, so it cannot say
    which key a legal file was mapped to, and per-file agreement is exactly the
    property D-22 makes non-coincidental.
    """
    return cast(Callable[[Path, Path], str], _published_namespace()["store_key_for"])


def _seed_cache_directory(root: Path) -> None:
    """Write one ``.npy`` artefact per stem, plus one file the scan must skip."""
    for stem in LEGAL_STEMS + ILLEGAL_STEMS:
        artefact = root / f"{stem}{paths_mod.NPY_SUFFIX}"
        artefact.parent.mkdir(parents=True, exist_ok=True)
        artefact.write_bytes(b"not really an array")
    # The caution the snippet carries, made executable: the sidecar extension
    # collides with mypy's cache format, and an unfiltered scan reported
    # thousands of false hits during this phase's research.
    noise = root / ".mypy_cache" / "3.12" / f"..{paths_mod.NPY_SUFFIX}"
    noise.parent.mkdir(parents=True, exist_ok=True)
    noise.write_bytes(b"type-checker cache, not a store artefact")


def _snapshot(root: Path) -> list[tuple[str, Optional[bytes]]]:
    """Return a content-level fingerprint of everything under ``root``."""
    entries: list[tuple[str, Optional[bytes]]] = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        entries.append((relative, path.read_bytes() if path.is_file() else None))
    return entries


def test_migration_snippet_reports_exactly_the_keys_that_are_now_refused(tmp_path: Path) -> None:
    """Plan 14-07 / STORE-07 / D-18 / SC-5.

    The snippet a consumer is told to run must return the keys that stop
    working — no more and no fewer. Two legal stems are seeded alongside the
    three illegal ones so the test can fail in *both* directions: a snippet that
    refused everything would pass a "finds the bad ones" assertion.

    The mypy-cache file is not decoration either. Its stem is ``..``, which the
    rule refuses, so a snippet that dropped the ``SKIP`` filter would report it
    and fail here.

    ↻ EXTENDED by Plan 14-10 (WR-04 / D-22). The third illegal stem is the empty
    key, whose artefact is the bare ``.npy`` file. It is here because the
    realigned derivation now reports it — the snippet used to report ``''`` for
    that file while the store adopted ``'.npy'``, so the two sides named
    different keys for one byte on disk.
    """
    _seed_cache_directory(tmp_path)

    refused = _published_scan()(tmp_path)

    assert refused == sorted(ILLEGAL_STEMS), (
        f"the published scan returned {refused!r}; expected exactly the two refused stems "
        f"{sorted(ILLEGAL_STEMS)!r}. Legal stems reported here mean the snippet is stricter "
        "than the shipped rule; missing illegal stems mean it is looser."
    )
    # The snippet's verdict must be the library's verdict, not a second opinion.
    assert all(not is_valid_store_key(key) for key in refused)
    assert all(is_valid_store_key(stem) for stem in LEGAL_STEMS)


def test_migration_snippet_is_read_only_and_gives_the_same_answer_twice(tmp_path: Path) -> None:
    """Plan 14-07 / STORE-07 / D-18.

    A consumer will run this against a live cache directory, probably more than
    once and possibly interrupting it. The snippet performs no writes, so a
    second run must return the same list and leave the tree byte-identical —
    which is also what makes an interrupted run safe to re-run.
    """
    _seed_cache_directory(tmp_path)
    scan = _published_scan()

    before = _snapshot(tmp_path)
    first = scan(tmp_path)
    second = scan(tmp_path)
    after = _snapshot(tmp_path)

    assert first == second, f"the scan is not repeatable: {first!r} then {second!r}"
    assert after == before, "the scan mutated the cache directory; it must be read-only"


def test_public_exports_all_resolve_from_the_package_root() -> None:
    """Plan 14-07 / STORE-07 / D-07.

    Pins the surface a downstream consumer actually imports from. Importing
    these from ``GSEGUtils.lazy_disk_cache.paths`` would pass while the
    *published* surface was broken, which is why the module-level import at the
    top of this file names the package and not the submodule.
    """
    imported = {
        "StoreKeyError": StoreKeyError,
        "StoreContainmentError": StoreContainmentError,
        "is_valid_store_key": is_valid_store_key,
        "get_npy_path": get_npy_path,
        "get_meta_path": get_meta_path,
        "get_legacy_pickle_path": get_legacy_pickle_path,
    }
    assert tuple(imported) == PUBLISHED_SURFACE

    declared = set(lazy_disk_cache_package.__all__)
    missing = [name for name in PUBLISHED_SURFACE if name not in declared]
    assert not missing, f"published but absent from __all__: {missing!r}"

    for name, obj in imported.items():
        assert getattr(lazy_disk_cache_package, name) is obj, (
            f"{name!r} resolves from the package root to a different object than the one imported"
        )


def test_public_exports_withhold_the_internal_names() -> None:
    """Plan 14-07 / STORE-07 / D-07.

    The withheld half of the same decision. Each of these three is reachable at
    ``GSEGUtils.lazy_disk_cache.paths.<name>`` for anyone who insists — the
    assertion is about ``__all__``, not about reachability — but publishing one
    by accident is a commitment that cannot be withdrawn without a break.
    """
    declared = set(lazy_disk_cache_package.__all__)
    leaked = [name for name in WITHHELD_NAMES if name in declared]
    assert not leaked, f"internal names leaked into the published surface: {leaked!r}"

    for name in WITHHELD_NAMES:
        assert hasattr(paths_mod, name), f"{name!r} vanished from the paths module entirely"


# ---------------------------------------------------------------------------
# builder_containment — the SECOND layer, asserted where it actually lives
# (Plan 14-08 / CR-01 / V-2 / SC-4 / T-14-27 / T-14-28)
# ---------------------------------------------------------------------------


class Sneaky(str):
    """A ``str`` subclass whose characters and whose ``__str__`` disagree.

    The whole point of the class is the disagreement. ``validate_store_key``
    inspects the *characters* — ``in``, ``ord()`` and ``PurePath(key).parts``
    all read the underlying ``str`` payload, which here spells the perfectly
    legal ``'safe'`` — while every path builder interpolates the key through an
    f-string, and ``f"{key}"`` dispatches to ``__format__``, which for an empty
    format spec returns ``str(self)``. So the value that is *validated* and the
    value that is *joined* are two different strings, and the lexical layer
    cannot see it.

    Defined at module level rather than inside the test that needs it because
    Plan 14-10 reuses it for the ``get`` re-raise assertion; a locally-defined
    class would have to be duplicated there, which is how one contract becomes
    two drifting expressions of itself.
    """

    def __str__(self) -> str:
        """Return the escaping string, regardless of the characters held."""
        return ILLEGAL_KEY


@pytest.mark.parametrize(("builder_name", "builder"), BUILDER_CASES)
def test_builder_containment_every_builder_refuses_an_escape_with_the_lexical_layer_disabled(
    builder_name: str,
    builder: Callable[[Path, str], Path],
    tmp_cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan 14-08 / CR-01 / SC-4 / T-14-27.

    **Why this group exists.** ``_assert_contained`` is the second of the two
    layers, and until this group it had zero coverage at every builder: both
    registry-driven builder groups above use ``ILLEGAL_KEY``, which
    ``validate_store_key`` refuses one line earlier inside ``_build``, so the
    containment call was never on the path of any assertion. Measured, not
    inferred — deleting that call from ``_build`` left the suite at **616
    passed**. With the lexical layer gone, the containment layer is the only
    thing standing between an escaping key and a path above the cache
    directory, so it is the thing that must be asserted directly.

    **The patch target is load-bearing.** ``validate_store_key`` is rebound on
    the ``paths`` *module object*, because ``_build`` looks the name up as a
    module global at call time. Patching a copy imported into this test module
    would leave the production lookup untouched and this group would pass
    vacuously against a builder that still ran the lexical check — green, and
    proving nothing. If this group ever passes while the containment call is
    deleted, suspect the patch target before suspecting the builder.

    Parametrised off ``BUILDER_CASES`` — the same registry the sibling groups
    use — so a sixth builder added in Phase 15 is covered by growth rather than
    by someone remembering to extend a hand-written list.

    The match is on ``paths_mod.CLAUSE_ESCAPES`` so this group *imports* the
    clause wording instead of restating it; a respelling breaks the test rather
    than silently loosening it.
    """
    monkeypatch.setattr(paths_mod, "validate_store_key", lambda *args, **kwargs: None)

    assert paths_mod.is_valid_store_key(ILLEGAL_KEY), (
        "the lexical layer was not actually neutralised, so this case cannot isolate the "
        "containment layer and would pass for the wrong reason"
    )

    with pytest.raises(StoreContainmentError, match=re.escape(paths_mod.CLAUSE_ESCAPES)):
        builder(tmp_cache_dir, ILLEGAL_KEY)


def test_builder_containment_a_str_subclass_defeats_the_lexical_layer_and_is_caught_by_containment(
    tmp_cache_dir: Path,
) -> None:
    """Plan 14-08 / CR-01 / SC-4 / T-14-28.

    The honest end-to-end proof that the two layers are genuinely separate,
    because it needs **no monkeypatch at all**: the lexical layer is fully
    intact and still lets this value through.

    ``is_valid_store_key`` is asserted ``is True`` *positively* rather than by
    the absence of a ``pytest.raises``. The distinction matters: an absence
    would also be satisfied by a predicate that had stopped being called, while
    the positive form records that the lexical layer actively accepts this
    value — which is the premise the rest of the assertion rests on.

    So the containment layer is not redundant defence-in-depth here; for this
    input it is the *only* control, which is what makes CR-01's deletability
    finding a live escape rather than a formality.
    """
    sneaky = Sneaky("safe")

    assert paths_mod.is_valid_store_key(sneaky) is True, (
        "the lexical layer no longer accepts the str subclass, so this case no longer "
        "demonstrates that the two layers are separate"
    )
    assert str(sneaky) == ILLEGAL_KEY, "the subclass no longer disagrees with its characters"

    refused: dict[str, str] = {}
    for builder_name, builder in sorted(paths_mod.STORE_PATH_BUILDERS.items()):
        try:
            built = builder(tmp_cache_dir, sneaky)
        except StoreContainmentError as exc:
            refused[builder_name] = str(exc)
        else:
            pytest.fail(
                f"{builder_name} returned {str(built)!r} for a key whose characters are legal but "
                f"whose __str__ is {ILLEGAL_KEY!r}; the containment layer did not fire"
            )

    assert set(refused) == set(paths_mod.STORE_PATH_BUILDERS), (
        "not every registered builder was exercised: "
        f"refused={sorted(refused)} registered={sorted(paths_mod.STORE_PATH_BUILDERS)}"
    )
    for builder_name, message in refused.items():
        assert paths_mod.CLAUSE_ESCAPES in message, (
            f"{builder_name} refused, but not with the containment clause: {message!r}"
        )


# ---------------------------------------------------------------------------
# contract-page route enumeration — derived from BOTH the class's own namespace
# and MutableMapping's mixins (Plan 14-17 / D-29 / WR-04 / RV-04 / T-14-78,
# T-14-101)
# ---------------------------------------------------------------------------

#: Routes deliberately **not** named in the page's enumeration, each with the
#: reason. These are the only members of the derived union that take no key and
#: mutate nothing, so no key can be refused at them and there is no policy to
#: publish. A silent omission and a decided one must look different, which is
#: what makes this an explicit table rather than a filter buried in a
#: comprehension.
ROUTE_ENUMERATION_ALLOWLIST: dict[str, str] = {
    "keys": "returns a view of the tracked keys; takes no key argument and mutates nothing",
    "items": "returns a view of (key, entry) pairs; takes no key argument and mutates nothing",
    "values": "returns a view of the entries; takes no key argument and mutates nothing",
}

#: Extraction floor. The marker block currently yields far more inline literals
#: than this; the floor exists so that a parser which has stopped seeing the
#: page fails on *itself* rather than passing vacuously over an empty match.
_MIN_ROUTE_PARAGRAPH_LITERALS: int = 20


def _derive_mapping_route_sets() -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(set_a, set_b)`` — the routes this class is answerable for.

    **Two sets, and deriving only the first is the defect this control exists
    to close** (RV-04). Round 3's enumeration was assembled by looking at
    ``disk_backed_store.py``, which is why it named five routes and missed
    ``clear`` and ``popitem``: both are supplied by ``MutableMapping`` and
    appear nowhere in this library, so they are invisible from the module *and*
    from the diff that changed their behaviour.

    * **Set (a) — overridden or policied.** The mapping-API names the class
      defines in its own namespace.
    * **Set (b) — inherited mutators.** The concrete mixins ``MutableMapping``
      supplies (its namespace minus its abstract methods) that the class does
      **not** define, and whose behaviour therefore flows through the overridden
      ``__getitem__`` / ``__setitem__`` / ``__delitem__``.

    Both are read off the ABCs rather than from a literal list, so a future
    Python that adds or renames a mixin changes this derivation instead of
    quietly leaving it stale. A hardcoded list would be a second enumeration,
    and a second enumeration drifts exactly the way the paragraph did.
    """

    def namespace(cls: type) -> frozenset[str]:
        return frozenset(name for name, value in vars(cls).items() if callable(value) and not name.startswith("_abc"))

    universe = namespace(Mapping) | namespace(MutableMapping)
    own = frozenset(vars(DiskBackedStore))
    set_a = universe & own
    mixins = namespace(MutableMapping) - frozenset(MutableMapping.__abstractmethods__)
    set_b = mixins - own
    return set_a, set_b


def _extract_route_paragraph_literals() -> list[str]:
    """Return every ``inline literal`` inside the page's route-enumeration block."""
    page = CONTRACT_PAGE.read_text(encoding="utf-8")
    blocks = re.findall(
        r"\.\. CONTRACT-PAGE-ROUTES: BEGIN\n(.*?)\.\. CONTRACT-PAGE-ROUTES: END",
        page,
        flags=re.DOTALL,
    )
    assert len(blocks) == 1, (
        f"expected exactly one CONTRACT-PAGE-ROUTES marker pair in {CONTRACT_PAGE.name}, found "
        f"{len(blocks)} — the page and this test are no longer in step"
    )
    return re.findall(r"``([^`]+)``", blocks[0])


def test_contract_page_route_paragraph_names_every_overridden_mapping_route() -> None:
    """Plan 14-17 / STORE-07 / D-29 / § WR-04 / RV-04 / T-14-78 / T-14-101.

    **The page's route enumeration keeps its exhaustiveness claim, and this is
    what makes the claim true rather than aspirational.** Round 3 published a
    five-route list headed by that claim; ``clear`` and ``popitem`` were missing
    and both were then found to misbehave. A partial list marked complete is
    harder to find defects in than a list that makes no such claim, so the claim
    now has a maintainer instead of only an author.

    **The derivation is over two sets, and a one-set derivation would have been
    green through the whole of round 3** — that is not a hypothetical, it is
    mutation proof 3 in this plan's SUMMARY, run on this repository. Set (a) is
    what ``DiskBackedStore`` defines; set (b) is what ``MutableMapping`` supplies
    and this class inherits. Round 3's two omissions live in set (b), where a
    control that introspects the class alone cannot see them — which is the same
    blindness, rebuilt as a test, that produced the defect.

    Set (b) is asserted non-empty on purpose: if the derivation breaks, it comes
    out empty and every inherited route silently stops being checked.

    The floor assertion comes first for the reason Plan 14-15's page tests give:
    it is an assertion about the *parser*, and a parser matching nothing would
    make every check below it vacuous and this test green.
    """
    literals = _extract_route_paragraph_literals()

    assert len(literals) >= _MIN_ROUTE_PARAGRAPH_LITERALS, (
        f"extracted only {len(literals)} inline literals from the route enumeration in "
        f"{CONTRACT_PAGE.name} (floor {_MIN_ROUTE_PARAGRAPH_LITERALS}) — the parser has stopped "
        f"seeing the page. Check the '.. CONTRACT-PAGE-ROUTES: BEGIN/END' markers. "
        f"Extracted: {literals}"
    )

    set_a, set_b = _derive_mapping_route_sets()
    assert set_b, (
        "set (b) came out empty — the inherited-mutator derivation is broken, not absent. Every "
        "route MutableMapping supplies would silently stop being checked, which is precisely how "
        "clear() and popitem() went unenumerated for a whole round (RV-04)"
    )

    origin = {name: "a (defined by DiskBackedStore)" for name in set_a}
    origin.update({name: "b (inherited from MutableMapping)" for name in set_b})

    for name in ROUTE_ENUMERATION_ALLOWLIST:
        assert name in origin, (
            f"{name!r} is allowlisted out of the page enumeration but is no longer a member of "
            f"either derived set; a stale exclusion hides a route that has come back"
        )

    required = sorted(set(origin) - set(ROUTE_ENUMERATION_ALLOWLIST))
    for name in required:
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
        assert any(pattern.search(literal) for literal in literals), (
            f"the route enumeration in {CONTRACT_PAGE.name} does not name {name!r}, which is in "
            f"derived set {origin[name]}. Either name it in the paragraph between the "
            f"CONTRACT-PAGE-ROUTES markers, or add it to ROUTE_ENUMERATION_ALLOWLIST with the "
            f"reason it is excluded — a silent omission and a decided one must not look alike. "
            f"Literals currently in the paragraph: {literals}"
        )


# ---------------------------------------------------------------------------
# route_store_property — the public accessor is a read-only view, not a write
# route (Plan 14-08 / CR-02 / V-1 / SC-1 / D-19 / T-14-26)
# ---------------------------------------------------------------------------

#: One illegal key per clause of the lexical rule, so the route is exercised
#: across the whole refusal vocabulary rather than at the single canonical key.
ILLEGAL_SHAPES = (ILLEGAL_KEY, "..", "", "a/b", "/etc/passwd", "x\n")


def test_route_store_property_is_a_read_only_view_not_a_write_route(
    make_store: MakeStoreFn, tmp_cache_dir: Path
) -> None:
    """Plan 14-08 / CR-02 / SC-1 / D-19 / T-14-26.

    Before D-19 this property returned ``self._store`` **by reference**, so
    ``store.store['../victim'] = entry`` installed an illegal key with no
    validation and ``keys()`` reported it — a write route into ``_store`` that
    nothing guarded, falsifying SC-1's *every route* quantifier.

    **The legal-key assertion is the one that carries the decision.** A legal
    key fails to install too, which is what proves the closure is *structural*
    rather than a fifth validation site: there is no write to validate, so
    there is no site to forget. Had this been closed by validation, the legal
    key would have gone in and SC-1 would still have been an enumeration of the
    routes someone remembered.

    **Why two different exception types, deliberately.** ``mappingproxy``
    refuses the two *syntactic* mutations through type slots that raise
    :exc:`TypeError`, and refuses the *method* mutations by simply not having
    them, which raises :exc:`AttributeError`. Both are refusals and neither
    mutates, so this test asserts the type each operation actually produces.
    Do not "tidy" this into a uniform ``TypeError`` — measured on CPython
    3.12, ``proxy.update(...)`` raises ``AttributeError: 'mappingproxy' object
    has no attribute 'update'``, and a uniform assertion goes red.

    The identity assertion matters independently: a view that *was* the
    internal dict would satisfy every read assertion here while leaving the
    write route wide open.
    """
    store = make_store(tmp_cache_dir)
    store["legal_key"] = _entry()
    baseline = store.keys()
    assert baseline == ["legal_key"]

    view = store.store
    assert view is not store._store, (
        "the property handed out the internal mapping itself; the write route CR-02 reported is "
        "still open regardless of what the annotation says"
    )
    assert isinstance(view, MappingProxyType), f"expected a read-only proxy, got {type(view).__name__}"

    # ``cast`` only defeats the *static* check so the runtime behaviour can be
    # exercised. The static half is a real part of the guarantee and is enforced
    # by mypy over this very file: without the cast, mypy rejects the assignment
    # as an unsupported target, which is how a downstream caller meets D-19.
    mutable = cast(Any, view)

    # Every illegal shape fails to install ...
    for illegal in ILLEGAL_SHAPES:
        with pytest.raises(TypeError):
            mutable[illegal] = _entry()
        assert store.keys() == baseline, f"a refused write of {illegal!r} still changed the store"

    # ... and so does a perfectly legal one, which is the structural proof.
    with pytest.raises(TypeError):
        mutable["another_legal_key"] = _entry()
    assert store.keys() == baseline, "a legal key installed through the read-only view"

    # The remaining mutating operations, each with the type it actually raises.
    with pytest.raises(TypeError):
        del mutable["legal_key"]
    for label, operation in (
        ("update", lambda: mutable.update({"another_legal_key": None})),
        ("pop", lambda: mutable.pop("legal_key")),
        ("clear", lambda: mutable.clear()),
        ("setdefault", lambda: mutable.setdefault("another_legal_key", None)),
    ):
        with pytest.raises(AttributeError):
            operation()
        assert store.keys() == baseline, f"the refused {label} still changed the store"

    # The accessor is narrowed, not removed: every read still works.
    assert "legal_key" in view
    assert ILLEGAL_KEY not in view
    assert len(view) == 1
    assert list(view) == ["legal_key"]
    assert view["legal_key"] is store._store["legal_key"]
    assert dict(view) == dict(store._store)


def test_route_getstate_hands_out_a_detached_copy_of_the_entry_mapping(
    make_store: MakeStoreFn, tmp_cache_dir: Path
) -> None:
    """Plan 14-12 / CR-02 / SC-1 / D-19 completed / T-14-50 / T-14-51.

    **The other half of the test directly above, and separating them is how
    this one went unnoticed for a round.** D-19 argued the ``store`` property
    was closed *structurally* — "there is no write to validate, so there is no
    site to forget" — but :meth:`~DiskBackedStore.__getstate__` took a
    **shallow** ``self.__dict__.copy()``, so ``state['_store'] is self._store``:
    a second live route into the same mapping, through a public protocol
    method, and therefore a write route SC-1's *every route* quantifier covers.
    The counterexample was the very line ``__setstate__``'s Notes block quoted
    as evidence.

    The second half has nothing to do with keys and is plainer:
    ``copy.copy`` routes through ``__reduce_ex__`` → ``__getstate__`` →
    ``__setstate__``, so a shallow entry mapping made two nominally independent
    stores one store — insert into the copy and the original grew the key. On a
    class explicitly designed to travel through joblib/loky, that is a
    data-integrity defect rather than a curiosity.

    **The behavioural assertions are not decoration.** An identity-only test is
    satisfied by a ``__getstate__`` that "detached" by handing back an empty
    dict — a silently short store, which is the failure D-10 exists to prevent.
    So the detached mapping must also compare *equal* to the original, and the
    over-tightness control lives in
    ``test_route_setstate_ordinary_round_trips_still_work_under_every_configuration``.

    The detachment is at the **mapping** level only: the entry values are still
    the same objects, asserted here so the docstring's claim is pinned to
    exactly what it says and no more.
    """
    store = make_store(tmp_cache_dir)
    store["legal_key"] = _entry()
    baseline = store.keys()
    assert baseline == ["legal_key"]

    # -- the __getstate__ route ------------------------------------------
    state = store.__getstate__()
    assert state["_store"] is not store._store, (
        "__getstate__ handed out the internal mapping itself; the write route CR-02 reported is "
        "still open, one accessor over from the one D-19 closed"
    )
    assert state["_store"] == store._store, (
        "the snapshot's mapping does not compare equal to the store's own — a snapshot that "
        "detached by dropping entries is a silently short store, not a fix"
    )
    assert state["_store"]["legal_key"] is store._store["legal_key"], (
        "the detachment reached the values; it is a mapping-level copy, not a deep copy"
    )

    state["_store"][ILLEGAL_KEY] = None
    assert store.keys() == baseline, (
        f"writing {ILLEGAL_KEY!r} through the snapshot changed the store's key set to {store.keys()!r}"
    )

    # -- the copy.copy route ---------------------------------------------
    shallow = copy.copy(store)
    assert shallow._store is not store._store, (
        "copy.copy(store) shares its entry mapping with the original; two nominally independent stores are one store"
    )
    shallow["another_legal_key"] = _entry()
    assert store.keys() == baseline, f"inserting into a shallow copy changed the original's key set to {store.keys()!r}"
    assert "another_legal_key" in shallow.keys(), "the copy did not take the insert either, so it is simply broken"


def test_route_setstate_detaches_the_incoming_entry_mapping(make_store: MakeStoreFn, tmp_cache_dir: Path) -> None:
    """Plan 14-18 / D-32 / § WR-03 / SC-1 / T-14-80 / T-14-81.

    **The inverse of the test directly above, and the third surface of one
    protocol pair.** Plan 14-12 detached the snapshot on the way *out*; this
    pins the way *in*. ``__setstate__``'s ``self.__dict__.update(state)``
    installed the caller's mapping **object**, so the restored store's
    ``_store`` *was* ``state["_store"]`` and a caller who retained ``state``
    kept a live, unvalidated write handle into a store that had validated every
    key it was shown.

    **Why this is written as an inverse rather than as a copy.** The
    ``__getstate__`` test mutates the *store* and asserts the *snapshot* is
    unchanged. This one mutates the *snapshot* — after the call — and asserts
    the *store* is unchanged. A test asserting only non-identity would be
    satisfied by any object that is merely a different object, including one
    that shares nothing useful, and would not pin the property that matters.

    **The legal-key assertion is the one that names the property.** Non-aliasing
    is the property; key legality is not. A "fix" that special-cased the illegal
    key — re-validating on the way through, say — would pass the illegal-key
    assertion and fail this one, and a reader needs to see that this test is
    about *sharing* rather than about *validation*. That distinction is D-19's
    whole argument: the route is closed structurally, not guarded.

    The reverse direction is asserted too, so a fix that replaced sharing in one
    direction with sharing in the other is caught rather than congratulated.
    """
    source = make_store(tmp_cache_dir / "source")
    source["legal"] = _entry()
    state = source.__getstate__()

    victim = make_store(tmp_cache_dir / "victim")
    victim.__setstate__(state)
    baseline = sorted(victim.keys())
    assert baseline == ["legal"], (
        f"the round trip did not restore the entry (keys={baseline!r}); a detach that drops entries "
        "is a silently short store, not a fix"
    )

    # (1) The cheap structural check.
    assert victim._store is not state["_store"], (
        "__setstate__ installed the caller's mapping object itself; the restored store shares its "
        "entry mapping with whoever holds the state (§ WR-03, the mirror of the CR-02 defect "
        "Plan 14-12 closed on __getstate__)"
    )

    # (2) The property that actually matters, with an ILLEGAL key.
    state["_store"][ILLEGAL_KEY] = None
    assert sorted(victim.keys()) == baseline, (
        f"writing {ILLEGAL_KEY!r} into the retained state AFTER __setstate__ changed the store's "
        f"key set to {sorted(victim.keys())!r} — an illegal key installed post-validation through "
        "a mapping the caller still holds"
    )

    # (3) The same with a LEGAL key: the property is non-aliasing, not legality.
    state["_store"]["smuggled_legal"] = None
    assert sorted(victim.keys()) == baseline, (
        f"writing a LEGAL key into the retained state after __setstate__ changed the store's key "
        f"set to {sorted(victim.keys())!r}. This assertion is not about key legality — it is what "
        "distinguishes a structural detach from a fix that only re-validates the illegal case"
    )

    # (4) The reverse direction: sharing must not be re-established the other way.
    state_baseline = sorted(state["_store"])
    victim._store["from_the_store_side"] = None
    assert sorted(state["_store"]) == state_baseline, (
        f"mutating the restored store changed the caller's state dict to {sorted(state['_store'])!r}; "
        "sharing was moved rather than removed"
    )


# ---------------------------------------------------------------------------
# route_rescan (Plan 14-10 / WR-04 / D-22 / T-14-39 / T-14-40 / T-14-41)
#
# One key derivation, proven per file by rebuilding the path from the derived
# key — and still never fatal.
# ---------------------------------------------------------------------------

#: The public surface D-21 declined (WR-06). The review proposed an opt-in
#: ``expected_cache_dir`` check; it was considered and refused, because a
#: published parameter in a package on production PyPI cannot be withdrawn
#: without a break.
DECLINED_SURFACE = "expected_cache_dir"

#: The pre-fix artefact of the empty key: a file whose *whole name* is the array
#: suffix. A cache directory written before this phase can genuinely contain one,
#: because the empty key was accepted then.
EMPTY_KEY_ARTEFACT_NAME = paths_mod.NPY_SUFFIX


def _warning_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the rendered text of every WARNING record captured so far.

    Rendered rather than ``record.args`` on purpose, matching the convention the
    D-09 rescan test established: the module's house style is lazy ``%``
    interpolation and stays so, but keying an assertion to ``args`` would pin the
    formatting idiom instead of the behaviour.
    """
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


def test_route_rescan_requires_the_derived_key_to_rebuild_its_own_file(
    tmp_cache_dir: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 14-10 / WR-04 / D-22 / T-14-39.

    **The round-trip guard is the invariant; the aligned derivation alone is a
    coincidence.** Stripping the suffix off the file *name* happens to agree with
    the builders for today's suffix vocabulary, and that agreement is what makes
    the empty-key phantom disappear. But "happens to agree" is a filter, not an
    invariant — the same distinction this phase keeps drawing about paths. The
    guard rebuilds the path from the derived key through the shared builder and
    refuses the file when the result is not the file the key came from.

    **Why this test monkeypatches the builder, stated plainly rather than left
    to look like laziness.** With the derivation and the builders in step, no
    real filename reaches the round-trip branch — which is precisely what makes
    the guard *unobservable* and therefore untestable by seeding alone. Deleting
    it would leave the suite green and the property standing by luck. So the
    disagreement is injected: one key's builder returns a different path, which
    is exactly the drift the guard exists to survive. If this test ever passes
    with the guard deleted, suspect the patch target — ``disk_backed_store``
    calls the builder as a module attribute at call time, so the rebinding must
    be on the ``paths`` module object.

    ↻ **PATCH TARGET MOVED by Plan 14-14 (D-26).** The rescan used to rebuild
    with ``paths.get_npy_path``; it now rebuilds through ``paths._build`` with
    the *instance's* own artefact suffix, so that is where the drift has to be
    injected. Only the seam moved: what this test asserts is unchanged, and it
    still goes red when the round-trip comparison is deleted. That is what
    "D-26 changes which builder the guard calls, not whether the guard exists"
    means in practice.

    The ``ordinary`` control is not decoration: a rescan that adopted *nothing*
    would satisfy "does not adopt the drifting key" while being completely
    broken.
    """
    drifting = tmp_cache_dir / f"drifting{paths_mod.NPY_SUFFIX}"
    ordinary = tmp_cache_dir / f"ordinary{paths_mod.NPY_SUFFIX}"
    for planted in (drifting, ordinary):
        planted.write_bytes(b"not-a-real-npy")

    real_build = paths_mod._build

    def drifting_build(cache_dir: Path, key: str, suffix: str) -> Path:
        """Return a path that is not the file ``key`` was derived from."""
        if key == "drifting":
            return cache_dir / f"{key}_elsewhere{suffix}"
        return real_build(cache_dir, key, suffix)

    monkeypatch.setattr(paths_mod, "_build", drifting_build)

    with caplog.at_level("WARNING"):
        store = _store_with_cache_path(tmp_cache_dir)

    assert "drifting" not in store.keys(), (
        "the rescan adopted a key that does not rebuild its own file — a phantom key the store "
        "advertises through keys() and can never load"
    )
    assert "ordinary" in store.keys(), "the rescan adopted nothing, so the negative assertion proves nothing"
    assert drifting.exists(), "the rescan deleted a file it should only skip"

    # ``repr`` on both halves, matching the message (D-13 / Plan 14-12). This
    # name needs no escaping so a raw match would also pass; it is written this
    # way so the two rescan assertions agree about what the message contains.
    assert any(
        repr(drifting.name) in message and repr("drifting") in message for message in _warning_messages(caplog)
    ), "no WARNING named the skipped file together with the key that was derived for it"


def test_route_rescan_survives_a_builder_that_refuses_during_the_round_trip(
    tmp_cache_dir: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 14-10 / D-22 under D-09's standing prohibition / T-14-40.

    D-22 inherits D-09's policy rather than replacing it: **opening a
    pre-existing cache directory must never become a crash.** The round-trip
    guard calls the shared builder, and that builder *raises* — for a key the
    lexical rule refuses, and potentially with a containment error on an odd
    directory. A guard added under a warn-and-skip policy that could itself
    propagate would violate the very policy it was added under, so the rebuild is
    computed inside a handler that turns any builder refusal into the same
    warn-and-skip.

    That handler is unreachable through the real builders once the predicate has
    passed, which is why the refusal is injected here. The alternative — trusting
    the ordering argument — is what this phase has repeatedly found insufficient.

    ↻ **PATCH TARGET MOVED by Plan 14-14 (D-26)**, from ``paths.get_npy_path`` to
    the shared ``paths._build`` seam the rebuild now calls. The prohibition being
    asserted is untouched: a builder refusal during the round trip is still
    converted into the same warn-and-skip and still cannot reach the caller.
    """

    def refusing_build(cache_dir: Path, key: str, suffix: str) -> Path:
        """Refuse every key, the way the builder would on a hostile directory."""
        raise StoreContainmentError(f"synthetic refusal for {key!r}: the path {paths_mod.CLAUSE_ESCAPES}")

    (tmp_cache_dir / f"ordinary{paths_mod.NPY_SUFFIX}").write_bytes(b"not-a-real-npy")
    monkeypatch.setattr(paths_mod, "_build", refusing_build)

    with caplog.at_level("WARNING"):
        store = _store_with_cache_path(tmp_cache_dir)

    assert store.keys() == [], "a key was adopted despite the round-trip check refusing it"
    assert _warning_messages(caplog), "the refusal was swallowed silently instead of being warned about"


def test_route_rescan_opens_a_directory_holding_every_refused_shape_without_raising(
    tmp_cache_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Plan 14-10 / D-09's standing prohibition, re-asserted under D-22 / T-14-40.

    Every refused shape available at this point in the phase, in one directory,
    opened at once:

    * a nested artefact (the separator clause, and the silent-leak bug fix);
    * the pre-fix artefact of the empty key (the reserved clause, and the shape
      D-22 exists for);
    * a trailing-dot stem and a reserved Win32 device name — both refused only
      since D-20, which Plan 14-09 shipped and this plan depends on.

    Construction must succeed, warn, adopt only the legal artefact and leave
    every file on disk. The legal control is what keeps the assertion from
    passing vacuously.
    """
    planted = {
        "nested": tmp_cache_dir / "tile_03" / f"range{paths_mod.NPY_SUFFIX}",
        "empty-key": tmp_cache_dir / EMPTY_KEY_ARTEFACT_NAME,
        "trailing-dot": tmp_cache_dir / f"foo.{paths_mod.NPY_SUFFIX}",
        "win-device": tmp_cache_dir / f"CON{paths_mod.NPY_SUFFIX}",
        "legal": tmp_cache_dir / f"ordinary{paths_mod.NPY_SUFFIX}",
    }
    for artefact in planted.values():
        artefact.parent.mkdir(parents=True, exist_ok=True)
        artefact.write_bytes(b"not-a-real-npy")

    with caplog.at_level("WARNING"):
        store = _store_with_cache_path(tmp_cache_dir)

    assert store.keys() == ["ordinary"], (
        f"the rescan adopted {store.keys()!r}; only the legal artefact may be tracked, and the "
        "nested one is invisible to a non-recursive glob by design"
    )
    for label, artefact in planted.items():
        assert artefact.exists(), f"the rescan deleted the {label} artefact instead of skipping it"

    # Three top-level refusals (nested is never globbed, legal is adopted).
    assert len(_warning_messages(caplog)) == 3, (
        f"expected one WARNING per refused top-level artefact, got {_warning_messages(caplog)!r}"
    )


#: The forged record a planted name embeds. It is shaped like a real log line
#: on purpose: the failure mode CR-01 reproduced is not a garbled message but a
#: *well-formed* second record asserting the opposite of what happened.
FORGED_RECORD = "\nWARNING:all clear, 0 files skipped\n"


@pytest.mark.parametrize(
    "newline_planted_in",
    ["filename", "cache-directory"],
    ids=["newline-in-filename", "newline-in-cache-directory"],
)
def test_route_rescan_warning_escapes_a_newline_bearing_filename(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, newline_planted_in: str
) -> None:
    r"""Plan 14-12 / CR-01 / D-13 under D-09 / T-14-49 / T-14-52.

    **The rescan sibling of**
    ``test_message_contains_no_newline_for_a_newline_bearing_key``
    (``test_store_key_rules.py``), and it lives here rather than beside its twin
    because its subject is a *route*, not the rule: it needs a real
    :class:`DiskBackedStore`, a real cache directory and a real file planted in
    it, which is what the route-policy matrix in this module is for.

    This message is the **only** one in the subsystem whose input is genuinely
    untrusted — the rescan reads names off disk, written by whoever can write
    into the cache directory, and newlines are legal in POSIX filenames. Under
    D-09's warn-and-skip policy the log is the reader's *only* signal, which is
    what makes forging it worse here than anywhere else: the reproduction
    planted a filename that made one ``logger.warning`` call emit three lines,
    the second of which read ``WARNING:all clear, 0 files skipped`` while a file
    was in fact being skipped.

    **Two seedings, because the message interpolates two components.** One
    plants the newline in the *file* name, the other in the *cache directory*
    name — a directory whose own name embeds a newline, holding an
    ordinarily-refused artefact so the warning fires at all. Reverting either
    format specifier must turn this test red; a single seeding would leave one
    of them a change nothing would notice reverting.

    Asserted on ``record.getMessage()`` — the *rendered* result of the lazy
    ``%``-interpolation — rather than on ``record.msg``: the latter is the
    format string, which is identical with and without the defect.
    """
    if newline_planted_in == "filename":
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        planted = cache_dir / f"evil{FORGED_RECORD}.{paths_mod.NPY_SUFFIX}"
    else:
        cache_dir = tmp_path / f"cache{FORGED_RECORD}dir"
        cache_dir.mkdir()
        planted = cache_dir / EMPTY_KEY_ARTEFACT_NAME
    planted.write_bytes(b"not-a-real-npy")

    with caplog.at_level("WARNING"):
        store = _store_with_cache_path(cache_dir)

    assert store.keys() == [], f"the rescan adopted {store.keys()!r} for a refused artefact"
    assert planted.exists(), "the rescan deleted a file it should only skip"

    messages = _warning_messages(caplog)
    assert len(messages) == 1, f"expected exactly one WARNING for the single planted file, got {messages!r}"
    for message in messages:
        assert "\n" not in message, f"a planted newline split the record across lines: {message!r}"
    assert "\\n" in messages[0], (
        f"the newline was not rendered in its escaped two-character form, so the escaping is a "
        f"strip rather than repr: {messages[0]!r}"
    )


def test_rescan_and_the_published_snippet_derive_the_same_key_for_every_seeded_file(
    tmp_cache_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Plan 14-10 / WR-04 / D-22 / T-14-41 / SC-5.

    **The agreement is asserted per file, not on the aggregate list, and that is
    the whole point.** Two derivations can return the same sorted list of refused
    keys while disagreeing about *which file produced which key* — which is
    exactly the WR-04 reproduction: for the single byte ``<cache>/.npy`` the
    snippet reported ``''`` and the store adopted ``'.npy'``. An aggregate
    comparison is blind to that; a per-file one is not.

    The rescan's derivation is read off its *observable behaviour* rather than
    re-implemented here: for each seeded top-level artefact the store either
    adopted exactly the snippet's key, or logged a WARNING naming that file
    together with that key. Re-implementing it in the test would create a third
    expression of the derivation this task exists to reduce to one.

    The nested artefact is the deliberate asymmetry, asserted rather than
    ignored: the reopen scan globs non-recursively, so the store never sees it,
    while the snippet reports it under its relative path — which is precisely the
    silent leak the contract page tells consumers about, and the reason the
    snippet is recursive and the rescan is not.
    """
    _seed_cache_directory(tmp_cache_dir)
    store_key_for = _published_store_key_for()

    with caplog.at_level("WARNING"):
        store = _store_with_cache_path(tmp_cache_dir)

    adopted = set(store.keys())
    messages = _warning_messages(caplog)

    top_level = sorted(f for f in tmp_cache_dir.glob(f"*{paths_mod.NPY_SUFFIX}") if f.is_file())
    assert len(top_level) == 4, f"the fixture no longer seeds the expected top-level artefacts: {top_level!r}"

    for artefact in top_level:
        expected = store_key_for(artefact, tmp_cache_dir)
        if is_valid_store_key(expected):
            assert expected in adopted, (
                f"the snippet derives {expected!r} for {artefact.name!r} and the rule accepts it, "
                f"but the store did not adopt it; adopted={sorted(adopted)!r}"
            )
            assert f"{expected}{paths_mod.NPY_SUFFIX}" == artefact.name, (
                f"the adopted key {expected!r} does not rebuild {artefact.name!r}"
            )
        else:
            assert expected not in adopted, (
                f"the store adopted {expected!r}, which the published rule refuses — the reopen "
                "scan and the published scan disagree about this file"
            )
            # The filename is matched through ``repr`` because the message
            # renders it that way (D-13, extended to this call site by Plan
            # 14-12 / CR-01). A raw-substring match was correct until then and
            # goes red for exactly one seeded file — ``evil\..\x.npy``, whose
            # backslashes ``repr`` doubles — which is the escaping working.
            assert any(repr(artefact.name) in message and repr(expected) in message for message in messages), (
                f"no WARNING named {artefact.name!r} together with the key {expected!r} the snippet "
                f"derives for it; captured warnings were {messages!r}"
            )

    nested = tmp_cache_dir / "tile_03" / f"range{paths_mod.NPY_SUFFIX}"
    nested_key = store_key_for(nested, tmp_cache_dir)
    assert nested_key == os.path.join("tile_03", "range"), (
        f"the snippet's nested derivation regressed to {nested_key!r} while the empty-key case was fixed"
    )
    assert not is_valid_store_key(nested_key), "the nested key is no longer refused, so the snippet reports nothing"
    assert nested_key not in adopted, "the non-recursive rescan somehow adopted a nested artefact"
    assert nested.exists(), "the rescan reached into a subdirectory and disturbed it"


#: The three artefact-suffix class attributes D-27 withdraws, as data rather
#: than inline literals so the failure message can name *which* one came back.
#: Their presence is what let the rescan's discovery vocabulary drift away from
#: the load and write vocabularies, which is CR-01.
WITHDRAWN_SUFFIX_ATTRS = ("_DBNDArrayFileExt", "_DBNDArrayMetaExt", "_LegacyPickleExt")


def test_route_rescan_has_one_codec_artefact_vocabulary_and_every_adopted_key_loads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Plan 14-16 / CR-01 + IN-02 / D-27 / T-14-70, T-14-71, T-14-72.

    **The property, stated with the qualifier it needs.** ``DiskBackedStore``
    holds **one codec artefact vocabulary** — ``.npy`` / ``.meta.json`` /
    ``.pkl``, all read from :mod:`GSEGUtils.lazy_disk_cache.paths` by the glob,
    the key derivation, the round-trip rebuild, ``_load_entry``, ``_store_entry``
    and ``offload`` alike. It does **not** hold one artefact vocabulary
    unqualified: the ``.dat`` memmap suffix keeps its own, duplicated between
    ``paths.MEMMAP_SUFFIX`` and ``LazyDiskCache._MEMMAP_SUFFIX``, and unifying
    those is STORE-08 / Phase 15 work this round does not do. The test's own name
    carries the qualifier because an overstated claim in a test name outlives
    every document that repeats it.

    **This replaces the D-26 test, and the reason is the assertion that test
    omitted.** ``test_route_rescan_adopts_a_subclass_that_repoints_the_artefact_suffix``
    asserted ``keys()`` and never read an adopted entry. Measured at ``7d798cd``
    with a genuine ``alpha.sub.npy`` + ``alpha.meta.json`` pair under a suffix-
    repointing subclass: ``keys() == ['alpha']``, ``'alpha' in s`` is ``True``,
    **no warning at all**, and ``s['alpha']`` raises ``KeyError``. A store full of
    ``KeyError`` read as green, because ``keys()`` is not evidence about
    retrieval. So the end-to-end half below **subscripts every adopted key and
    compares the payload**.

    **Two halves, and both are needed.** The structural half is a proxy — it
    pins that no second codec vocabulary can be introduced through an instance
    attribute. The end-to-end half is the property that proxy stands for. A
    mutation that repoints the glob and the derivation at a foreign constant
    while leaving the rebuild on the module constant passes the structural half
    and fails the end-to-end one, which is CR-01's exact mechanism.

    **IN-02 rides along.** ``key = f.name[: -len(suffix)]`` degenerates to the
    empty key for *every* file when the suffix is empty, and the empty suffix was
    reachable only through the withdrawn attribute. With a module constant the
    slice length is decided at import time, so the degenerate case is
    unrepresentable rather than guarded.

    The negative controls and the base-class positive control are carried over
    from the D-26 test, for the reason that test gives: a change that repaired
    one path by breaking another would otherwise surface only as an aggregate
    suite-count change, which is not a diagnosis.
    """
    # -- The structural half: no second codec vocabulary can be introduced. --
    seeded_dir = tmp_path / "codec-cache"
    seeded_dir.mkdir()
    store = _store_with_cache_path(seeded_dir)

    for attr in WITHDRAWN_SUFFIX_ATTRS:
        assert not hasattr(DiskBackedStore, attr), (
            f"DiskBackedStore.{attr} resolves again; the artefact-suffix override point was "
            "withdrawn by D-27 precisely because discovery read it and retrieval did not, so a "
            "store could adopt keys it can never load (CR-01)"
        )
        assert not hasattr(store, attr), (
            f"an instance resolves {attr}; the class attribute is gone but something re-creates it "
            "per instance, which re-opens the same divergence (CR-01)"
        )

    # -- The end-to-end half: every adopted key is read, payload compared. --
    payloads = {
        "alpha": np.arange(4, dtype=np.float64),
        "beta": np.arange(10, 16, dtype=np.float64),
    }
    for key, payload in payloads.items():
        store.add_data_to_store(key, payload)
    store.offload(pickle_container=True)

    # Planted after the seeding so the writing store's own construction scans an
    # empty directory: the single WARNING asserted below is unambiguously the
    # reopen's.
    #
    # `foreign` carries the `.dat` memmap suffix on purpose. It is the surviving
    # second vocabulary this test's name is qualified against, and having it sit
    # here as a control is the cheapest possible reminder that the codec claim
    # does not reach it (STORE-08).
    foreign = seeded_dir / f"foreign{paths_mod.MEMMAP_SUFFIX}"
    refused = seeded_dir / f"CON{paths_mod.NPY_SUFFIX}"
    foreign.write_bytes(b"not-a-real-npy")
    refused.write_bytes(b"not-a-real-npy")

    caplog.clear()
    with caplog.at_level("WARNING"):
        reopened = _store_with_cache_path(seeded_dir)

    assert sorted(reopened.keys()) == sorted(payloads), (
        f"the reopened store adopted {sorted(reopened.keys())!r}, not {sorted(payloads)!r}"
    )
    for key, payload in payloads.items():
        entry = reopened[key]
        assert entry is not None, (
            f"the reopened store adopted {key!r} and handed back None for it; adopting a key the "
            "store cannot resolve is a silently short store (CR-01)"
        )
        np.testing.assert_array_equal(
            np.asarray(entry),
            payload,
            err_msg=(
                f"the payload round-tripped for {key!r} is not the one written; a rescan that "
                "adopts and returns an empty placeholder is the same defect as one that raises"
            ),
        )

    # -- The negative controls, carried over from the D-26 test. --
    assert "CON" not in reopened.keys(), "the rescan adopted a key the lexical rule refuses"
    assert foreign.stem not in reopened.keys(), (
        "the rescan adopted an artefact carrying the .dat memmap suffix, which the codec glob never sees"
    )

    messages = _warning_messages(caplog)
    assert len(messages) == 1, f"expected exactly one WARNING, for the refused artefact only; got {messages!r}"
    assert repr(refused.name) in messages[0], f"the single WARNING does not name the refused artefact: {messages[0]!r}"
    assert repr(foreign.name) not in messages[0], (
        "the foreign-suffix artefact produced a warning; the glob is supposed to never see it"
    )
    for label, artefact in {"foreign": foreign, "refused": refused}.items():
        assert artefact.exists(), f"the rescan deleted the {label} artefact instead of skipping it"

    # -- The base-class positive control, in its own directory. --
    #
    # A bare artefact with no sidecar, adopted with no warning: this pins that
    # the end-to-end half above did not tighten *adoption* into requiring a
    # complete codec pair. Adoption is a statement about the name; resolution is
    # a statement about the pair, and conflating them would be a different bug
    # wearing this one's clothes.
    caplog.clear()
    base_dir = tmp_path / "base-cache"
    base_dir.mkdir()
    (base_dir / f"ordinary{paths_mod.NPY_SUFFIX}").write_bytes(b"not-a-real-npy")
    with caplog.at_level("WARNING"):
        base_store = _store_with_cache_path(base_dir)

    assert base_store.keys() == ["ordinary"], (
        f"the base class adopted {base_store.keys()!r}; the deletion broke ordinary adoption"
    )
    assert _warning_messages(caplog) == [], "the base class now warns about an artefact it used to adopt"


def test_route_rescan_still_refuses_the_phantom_empty_key(tmp_cache_dir: Path) -> None:
    """Plan 14-14 / D-22 restated under D-26 / T-14-39.

    D-26 changes which builder the round-trip guard calls, so the property D-22
    bought is re-asserted directly rather than inferred from the guard still
    being present: the pre-fix artefact of the empty key — a file whose *whole*
    name is the array suffix — is adopted under neither the phantom key
    ``'.npy'`` (what ``Path.stem`` returns for it) nor the empty string (what the
    derivation returns).
    """
    (tmp_cache_dir / EMPTY_KEY_ARTEFACT_NAME).write_bytes(b"not-a-real-npy")

    store = _store_with_cache_path(tmp_cache_dir)

    assert store.keys() == [], f"the rescan adopted {store.keys()!r} for the empty-key artefact"
    assert paths_mod.NPY_SUFFIX not in store.keys(), "the phantom key came back"
    assert "" not in store.keys(), "the empty key was adopted despite not round-tripping"


# ---------------------------------------------------------------------------
# route_setstate (Plan 14-10 / WR-06 / D-21 / T-14-37)
#
# The containment base crosses the pickle boundary too, and it was the one
# thing crossing it unchecked.
# ---------------------------------------------------------------------------


def test_route_setstate_refuses_a_non_path_cache_dir_before_updating_the_instance(
    make_store: MakeStoreFn, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Plan 14-10 / WR-06 / D-21 / T-14-37.

    D-10 argues at length that unpickling is a trust boundary and that this
    module already treats it as one — then validated every incoming *key*
    against a containment base it installed **verbatim from the same untrusted
    state**. ``_cache_dir`` is the authorization boundary every builder resolves
    against, so key validation against an attacker-chosen base does not
    constrain where bytes land: every key passes, and every path lands wherever
    the base points.

    **The refusal must precede ``__dict__.update``, and the "left unmodified"
    half is what proves it did.** Asserting only that the call raised would be
    satisfied by a guard placed after the update, which is the silently-short —
    here silently-*redirected* — store D-10 exists to prevent. It is also why
    the guard sits before the per-key loop rather than after it: the refusal
    message for a bad key interpolates the incoming base, so a malformed base
    would otherwise be rendered into a message before anything noticed it was
    malformed.

    ↻ **CORRECTED by Plan 14-14 (§ WR-03).** This test used to end by asserting
    that an *absent* base and an *explicitly-``None``* base were both accepted,
    on the stated grounds that "a store with no configured cache path is a
    supported configuration and a guard that broke it would be a regression."
    The configuration is genuinely supported; it simply **does not produce
    either shape**. ``__init__`` assigns a ``Path`` on *both* of its branches —
    the no-cache-path arm falls back to ``Path(tempfile.mkdtemp())`` — so the
    carve-out protected nothing real while admitting a state the constructor
    cannot produce, after which the store accepted keys and died on the first
    path build with a bare ``TypeError``. Both shapes are refusals now.

    Worse, the ``absent_base`` half was **vacuous**: deleting the key from the
    state dictionary and then calling ``__dict__.update`` does not *remove* an
    attribute, so the victim kept the ``Path`` its own constructor assigned and
    the sub-case asserted nothing at all. That is why the absent-base case below
    asserts the state genuinely lacks the key before handing it over — the
    vacuity is checked for rather than assumed away.

    The over-tightness control did not disappear with the carve-out; it moved to
    the configuration that actually needs it. See
    ``test_route_setstate_ordinary_round_trips_still_work_under_every_configuration``
    and the closing assertion of
    ``test_route_setstate_refuses_a_cache_dir_shape_init_cannot_produce``.

    **Scope, recorded so it is not re-litigated.** No ``expected_cache_dir``
    parameter is added. The review offered one; the user declined the extra
    public surface, and that refusal is part of D-21 rather than an omission.
    """
    donor = make_store(tmp_cache_dir)
    victim = make_store(tmp_cache_dir)

    hostile: dict[str, Any] = dict(donor.__dict__)
    hostile["_cache_dir"] = str(tmp_path / "elsewhere")
    hostile["_store"] = {"legal_key": None}

    before = dict(victim.__dict__)
    with pytest.raises(StoreKeyError, match="cache directory"):
        victim.__setstate__(hostile)

    assert dict(victim.__dict__) == before, (
        "__setstate__ updated the instance before refusing the base — a partially restored store "
        "is the failure this guard exists to prevent"
    )
    assert isinstance(victim.cache_dir, Path), "a non-Path containment base was installed"
    assert "legal_key" not in victim.store, "the hostile state's keys were installed anyway"

    # ... and so are the two shapes that used to be carved out, because
    # ``__init__`` can produce neither of them.
    absent_base: dict[str, Any] = dict(donor.__dict__)
    del absent_base["_cache_dir"]
    absent_base["_store"] = {}
    assert "_cache_dir" not in absent_base, (
        "the absent-base case does not actually present an absent base; that is the vacuity WR-03 "
        "found in this test's predecessor"
    )
    with pytest.raises(StoreKeyError, match="cache directory"):
        make_store(tmp_cache_dir).__setstate__(absent_base)

    explicit_none: dict[str, Any] = dict(donor.__dict__)
    explicit_none["_cache_dir"] = None
    explicit_none["_store"] = {}
    with pytest.raises(StoreKeyError, match="cache directory"):
        make_store(tmp_cache_dir).__setstate__(explicit_none)


def test_route_setstate_refuses_a_cache_dir_shape_init_cannot_produce(
    make_store: MakeStoreFn, tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Plan 14-14 / WR-03 / D-21 as tightened / T-14-60.

    **The whole argument in one sentence:** ``__init__`` assigns a ``Path`` on
    *both* of its cache-directory branches — a configured one, or
    ``Path(tempfile.mkdtemp())`` when none was configured — so a pickled state
    presenting anything else is malformed or legacy **by construction**, and
    ``__setstate__`` refuses it rather than installing it.

    D-21's original carve-out let an absent or explicitly-``None`` base through
    on the grounds that a store with no configured cache path is supported. It
    is supported, and it produces a ``Path``. So the carve-out protected no real
    configuration while admitting a state the constructor forbids, after which
    the store **accepted keys, tracked them**, and died at the first path build
    with a bare ``TypeError``:

        ``TypeError: unsupported operand type(s) for /: 'NoneType' and 'str'``

    That is the untyped-crash class this phase replaced with typed refusals
    everywhere else, reached through the one route that could still produce it —
    and while it lasted, ``cache_dir`` returned ``None`` under a ``-> Path``
    annotation that mypy cannot see through a deserialization boundary.

    **The "left unmodified" half is not decoration.** Asserting only that the
    call raised would be satisfied by a guard placed *after*
    ``__dict__.update``, which is the partially-restored store — a silently
    short store by another name — that D-10 exists to prevent. Every refused
    shape is therefore checked against a full before/after snapshot of the
    target's attributes.

    The closing assertion is the positive statement of the same guarantee, and
    it is the over-tightness control the carve-out used to be: a store built
    with **no configured cache path** round-trips, and its restored
    cache-directory accessor returns a ``Path``.
    """
    donor = make_store(tmp_cache_dir)
    shapes: dict[str, dict[str, Any]] = {}

    a_string: dict[str, Any] = dict(donor.__dict__)
    a_string["_cache_dir"] = str(tmp_path / "elsewhere")
    a_string["_store"] = {}
    shapes["a string"] = a_string

    explicit_none: dict[str, Any] = dict(donor.__dict__)
    explicit_none["_cache_dir"] = None
    explicit_none["_store"] = {}
    shapes["explicitly None"] = explicit_none

    absent: dict[str, Any] = dict(donor.__dict__)
    del absent["_cache_dir"]
    absent["_store"] = {}
    assert "_cache_dir" not in absent, "the absent-base state still carries the key, so this case tests nothing"
    shapes["absent"] = absent

    for label, state in shapes.items():
        victim = make_store(tmp_cache_dir)
        before = dict(victim.__dict__)
        with pytest.raises(StoreKeyError, match="cache directory"):
            victim.__setstate__(state)
        assert dict(victim.__dict__) == before, (
            f"__setstate__ modified the instance before refusing a {label} containment base; a "
            "partially restored store is the failure this guard exists to prevent"
        )
        assert isinstance(victim.cache_dir, Path), (
            f"the {label} base was installed, so cache_dir now falsifies its own -> Path annotation"
        )

    # The over-tightness control, stated positively: the configuration the old
    # carve-out claimed to protect really is supported, and it really does carry
    # a Path across the boundary.
    fallback_store = _store_with_cache_path(None, fallback_root=tmp_path)
    revived: DiskBackedStore[DiskBackedNDArray] = pickle.loads(pickle.dumps(fallback_store))
    assert isinstance(revived.cache_dir, Path), (
        "a store built without a configured cache path no longer restores a Path — the guard is "
        "over-tight, which is a regression rather than a fix"
    )


def test_route_setstate_reports_the_base_before_the_keys(make_store: MakeStoreFn, tmp_cache_dir: Path) -> None:
    """Plan 14-14 / WR-03 / D-21 — the *placement*, not merely the presence.

    A pickled state carrying **both** a malformed containment base and an
    illegal key must report the **base**. That is what the guard's position
    ahead of the per-key loop buys, and it is a concrete requirement rather than
    tidiness: the refusal message for a bad key interpolates the incoming base,
    so a guard placed after the loop renders a malformed base into a message
    before anything has noticed it is malformed.

    Written up front rather than added only if a mutation happened to survive. A
    control that is authored in response to its own mutation is the deletable
    shape round-1 CR-01 was, and the phase has now met it twice.
    """
    donor = make_store(tmp_cache_dir)
    both_wrong: dict[str, Any] = dict(donor.__dict__)
    both_wrong["_cache_dir"] = None
    both_wrong["_store"] = {ILLEGAL_KEY: None}

    victim = make_store(tmp_cache_dir)
    with pytest.raises(StoreKeyError) as excinfo:
        victim.__setstate__(both_wrong)

    message = str(excinfo.value)
    assert "cache directory" in message, (
        f"the base was not what was reported; the per-key loop ran first and reported the key instead: {message!r}"
    )
    assert "Invalid store key" not in message, (
        f"the key was reported ahead of the malformed base, so the guard sits below the per-key loop: {message!r}"
    )


@pytest.mark.parametrize("enable_caching", [True, False], ids=["caching-on", "caching-off"])
def test_route_setstate_ordinary_round_trips_still_work_under_every_configuration(
    make_store: MakeStoreFn, tmp_cache_dir: Path, tmp_path: Path, enable_caching: bool
) -> None:
    """Plan 14-10 / D-21 — the over-tightness control.

    A type guard on a deserialization route is one line away from breaking every
    legitimate unpickle, and the two caching configurations take *different*
    paths through ``__setstate__`` (only the caching-on branch runs the reload
    loop). The no-cache-path store is the third shape: it never carried a
    configured cache directory at all, so it is the configuration a guard
    written as "the base must be present and be a ``Path``" would break.
    """
    expected = _small_array()
    store = make_store(tmp_cache_dir, enable_caching=enable_caching)
    store.add_data_to_store("k0", expected)

    revived: DiskBackedStore[DiskBackedNDArray] = pickle.loads(pickle.dumps(store))
    assert "k0" in revived.keys(), "an ordinary round-trip lost its entry"
    assert isinstance(revived.cache_dir, Path)

    fallback_store = _store_with_cache_path(None, fallback_root=tmp_path)
    fallback_store["legal_key"] = _entry()
    revived_fallback: DiskBackedStore[DiskBackedNDArray] = pickle.loads(pickle.dumps(fallback_store))
    assert "legal_key" in revived_fallback.keys(), "a store built without a configured cache path no longer round-trips"


def test_route_setstate_adds_no_expected_cache_dir_surface() -> None:
    """Plan 14-10 / D-21 — the refusal, pinned so it is not silently reversed.

    The review proposed an opt-in ``expected_cache_dir`` check alongside the type
    guard. It was considered and declined: the user did not want the extra public
    surface, and a published parameter in a package on production PyPI cannot be
    withdrawn without a break. Recorded as a test rather than only as prose,
    because a later reader meeting the guard without the context will read the
    absence as an oversight and helpfully add it.

    **Asserted over the AST rather than over the source text**, and the
    distinction is not pedantry: D-21 *requires* the name to appear in
    ``__setstate__``'s Notes block, because recording the refusal is half the
    decision. A substring assertion would therefore make the documentation of
    the refusal indistinguishable from its reversal — and, written first that
    way, it went red against the very sentence the decision asked for. The AST
    sees parameters, assignments, attributes and keywords; it does not see
    docstrings or comments, which is exactly the discrimination wanted.
    """
    package_root = Path(paths_mod.__file__).parent.parent
    modules = sorted(package_root.rglob("*.py"))
    assert modules, f"no source modules found under {package_root}"

    offenders: list[str] = []
    for module in modules:
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            named = (
                isinstance(node, ast.arg)
                and node.arg == DECLINED_SURFACE
                or isinstance(node, ast.Name)
                and node.id == DECLINED_SURFACE
                or isinstance(node, ast.Attribute)
                and node.attr == DECLINED_SURFACE
                or isinstance(node, ast.keyword)
                and node.arg == DECLINED_SURFACE
            )
            if named:
                offenders.append(f"{module.name}:{getattr(node, 'lineno', '?')}")

    assert not offenders, (
        f"the declined {DECLINED_SURFACE!r} surface reappeared as code at {offenders!r}; it was "
        "considered and refused, and the refusal is recorded in __setstate__'s Notes block"
    )
    assert DECLINED_SURFACE not in inspect.signature(DiskBackedStore.__setstate__).parameters

    # The refusal must still be *recorded* — a silently-dropped rationale is how
    # a declined option comes back as a helpful addition.
    notes = DiskBackedStore.__setstate__.__doc__ or ""
    assert DECLINED_SURFACE in notes, "the Notes block no longer records why this surface was declined"
