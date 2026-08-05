"""Route-policy matrix for the disk-backed store's key containment.

Covers STORE-01 (lexical key validation at every route that writes into
``_store``, extended by D-11 to the read route), and is extended later in the
phase with STORE-02 (resolved containment at every path builder) and STORE-03
(symlink tolerance). This file is created by Plan 14-03 with the six-route
policy matrix and the SC-1 worker-safety proof; Plan 14-06 extends it with the
builder-containment and symlink tests; Plan 14-07 extends it with the migration
snippet's round-trip.

The point of this file is the *differences* between the routes. Four distinct
policies apply to six routes, and each difference is a decision:

===================== ============================================== ======
Route                 Policy                                         Basis
===================== ============================================== ======
``__setitem__``       raise, with no filesystem syscall              SC-1
``add_data_to_store`` raise, unconditionally                         SC-1
``__getitem__``       raise, before ``_load_entry`` touches disk     D-11
``get``               return the default                            D-11a
``__init__`` rescan   warn and skip, leaving the file untouched      D-09
``__setstate__``      raise, never a silently short store            D-10
===================== ============================================== ======
"""

import os
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest

from GSEGUtils.lazy_disk_cache import StoreKeyError
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


def _store_with_cache_path(cache_path: Optional[Path]) -> DiskBackedStore[DiskBackedNDArray]:
    """Build an empty store over ``cache_path`` — or over none at all.

    ``cache_path=None`` is the second half of the ``route_add_data``
    parametrisation: the store then falls back to a ``mkdtemp`` directory, and
    ``add_data_to_store``'s ``if self._cache_dir`` conditional is exercised in
    the state its author intended it to gate.
    """
    cfg = LazyDiskCacheConfig(
        enable_caching=True,
        cache_path=cache_path,
        purge_disk_on_gc=False,
        automatic_offloading=False,
    )
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
    configured_cache_path: bool, tmp_cache_dir: Path
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
    store = _store_with_cache_path(tmp_cache_dir if configured_cache_path else None)

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
    """
    store = make_store(tmp_cache_dir)
    sentinel = object()

    assert store.get(ILLEGAL_KEY, sentinel) is sentinel, "get raised (or returned something else) for an illegal key"
    assert (ILLEGAL_KEY in store) is False, "__contains__ must stay an unguarded, dict-backed membership test"
    with pytest.raises(StoreKeyError, match="Invalid store key"):
        store[ILLEGAL_KEY]


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
    carry an illegal key: ``__getstate__`` snapshots a ``_store`` the parent
    already validated, so an illegal key arriving here means a legacy or a
    tampered pickle.

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
