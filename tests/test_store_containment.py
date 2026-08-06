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
import pickle
import re
import textwrap
from pathlib import Path
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
    """Build the path a parent-directory key would produce: ``<cache>/../victim.npy``."""
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

#: Two stems that used to work and are now refused, one per mechanism: a nested
#: key (the silent-leak bug fix) and a single filename carrying backslashes (a
#: perfectly legal POSIX filename, refused under the Windows interpretation).
ILLEGAL_STEMS = ("tile_03/range", "evil\\..\\x")


def _published_scan() -> Callable[[Path], list[str]]:
    """Return the ``refused_keys`` function exactly as the documentation publishes it.

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
    return cast(Callable[[Path], list[str]], namespace["refused_keys"])


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
    working — no more and no fewer. Two legal stems are seeded alongside the two
    illegal ones so the test can fail in *both* directions: a snippet that
    refused everything would pass a "finds the bad ones" assertion.

    The mypy-cache file is not decoration either. Its stem is ``..``, which the
    rule refuses, so a snippet that dropped the ``SKIP`` filter would report it
    and fail here.
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
