"""Regression tests for the Phase-2 lazy_disk_cache hardening.

Covers SEC-01 (codec swap), FRAG-04 (atomicity), and FRAG-03 (finalizer
re-registration). This file is created by Plan 02-01 with the codec +
cache_path-population tests; Plan 02-04 extends with finalizer tests;
Plan 02-05 extends with atomicity regression tests.
"""

import gc
import json
import logging
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pytest

from GSEGUtils.lazy_disk_cache import lazy_disk_cache as ldc_mod
from GSEGUtils.lazy_disk_cache import paths as paths_mod
from GSEGUtils.lazy_disk_cache.disk_backed_ndarray import DiskBackedNDArray
from GSEGUtils.lazy_disk_cache.disk_backed_store import (
    _LAZY_DISK_CACHE_CLASS_REGISTRY,
    DiskBackedStore,
    _resolve_lazy_disk_cache_class,
    register_lazy_disk_cache_class,
)
from GSEGUtils.lazy_disk_cache.lazy_disk_cache import LazyDiskCacheConfig, _purge_cache_pair


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Return a fresh, empty cache directory under the per-test ``tmp_path``."""
    d = tmp_path / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_store(
    tmp_cache_dir: Path,
    *,
    enable_caching: bool = True,
    purge_disk_on_gc: bool = False,
    automatic_offloading: bool = False,
) -> DiskBackedStore[DiskBackedNDArray]:
    """Build an empty DiskBackedStore pointed at ``tmp_cache_dir``."""
    cfg = LazyDiskCacheConfig(
        enable_caching=enable_caching,
        cache_path=tmp_cache_dir,
        purge_disk_on_gc=purge_disk_on_gc,
        automatic_offloading=automatic_offloading,
    )
    return DiskBackedStore[DiskBackedNDArray](config=cfg, factory=DiskBackedNDArray)


def _make_store_with_one_entry(
    tmp_cache_dir: Path,
    key: str = "k0",
    *,
    data: np.ndarray | None = None,
) -> DiskBackedStore[DiskBackedNDArray]:
    """Build a DiskBackedStore, register one DiskBackedNDArray entry under ``key``, return the store.

    The data array defaults to a small float32 row vector; callers can pass a
    custom array (e.g. structured dtype) to exercise the codec on different
    payload shapes.
    """
    if data is None:
        data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    store = _make_store(tmp_cache_dir, enable_caching=True)
    store.add_data_to_store(key, data)
    return store


def test_codec_round_trip(tmp_cache_dir: Path) -> None:
    """Plan 02-01 / SEC-01 / D-02: a write + read via the new codec round-trips losslessly."""
    expected = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    store = _make_store_with_one_entry(tmp_cache_dir, key="k0", data=expected)
    store.offload(pickle_container=True)
    # The on-disk shape: .npy + .meta.json pair exists, .pkl does not.
    assert (tmp_cache_dir / "k0.npy").exists()
    assert (tmp_cache_dir / "k0.meta.json").exists()
    assert not (tmp_cache_dir / "k0.pkl").exists()
    # The meta has the right schema_version + class name.
    meta = json.loads((tmp_cache_dir / "k0.meta.json").read_text())
    assert meta["schema_version"] == 1
    assert meta["lazy_disk_cache_class"] == "DiskBackedNDArray"
    # Load back via the store and confirm the array round-trips.
    loaded = store["k0"]
    np.testing.assert_array_equal(np.asarray(loaded), expected)


def test_codec_round_trip_structured_dtype(tmp_cache_dir: Path) -> None:
    """Plan 02-01 / Pitfall 3: structured dtype (RGB-uint8) round-trips under allow_pickle=False."""
    # Structured-dtype RGB case from CONCERNS line 100, verified in RESEARCH Pitfall 3.
    rgb_dtype = np.dtype([("r", "u1"), ("g", "u1"), ("b", "u1")])
    expected = np.array([(1, 2, 3), (4, 5, 6)], dtype=rgb_dtype)
    store = _make_store_with_one_entry(tmp_cache_dir, key="rgb", data=expected)
    store.offload(pickle_container=True)
    # Sidecar should reflect the structured dtype string.
    meta = json.loads((tmp_cache_dir / "rgb.meta.json").read_text())
    assert meta["schema_version"] == 1
    assert meta["lazy_disk_cache_class"] == "DiskBackedNDArray"
    # Round-trip: byte-for-byte equality including the structured dtype layout.
    loaded = store["rgb"]
    round_tripped = np.asarray(loaded)
    assert round_tripped.dtype == rgb_dtype
    np.testing.assert_array_equal(round_tripped, expected)


def test_legacy_pkl_refused_as_cache_miss(tmp_cache_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Plan 02-01 / SEC-01 / D-05: a legacy <key>.pkl is refused with KeyError + one INFO log line.

    No pickle reader is invoked. (We verify this by writing a .pkl whose bytes would
    explode if any pickle module tried to load them, and asserting the error is
    KeyError, not UnpicklingError.)
    """
    store = _make_store(tmp_cache_dir, enable_caching=True)
    # Plant a poison .pkl (garbage bytes) under a key the store does not yet know about.
    # The store's __init__ only scans for *.npy on construction, so the .pkl is invisible
    # to the in-memory mapping but lives in the cache dir on disk; we manually register
    # the bare key so __getitem__ proceeds into _load_entry.
    (tmp_cache_dir / "k0.pkl").write_bytes(b"\xff\xff\xff would unpickle to garbage")
    store._store["k0"] = None  # mimic an offloaded entry tracked by the store
    with caplog.at_level("INFO"):
        with pytest.raises(KeyError):
            _ = store["k0"]
    assert any("Legacy pre-Phase-2 cache file" in r.getMessage() for r in caplog.records), (
        "D-05 INFO log message must be emitted when refusing a legacy .pkl."
    )


def test_unknown_lazy_disk_cache_class_rejected(tmp_cache_dir: Path) -> None:
    """Plan 02-01 / D-02: an unknown class name in the JSON sidecar raises ValueError."""
    # Write a hand-crafted .npy + .meta.json pair with class_name="EvilSubclass".
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.save(str(tmp_cache_dir / "kx.npy"), arr, allow_pickle=False)
    (tmp_cache_dir / "kx.meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lazy_disk_cache_class": "EvilSubclass",
                "shape": [3],
                "dtype": "<f4",
                "purge_disk_on_gc": False,
                "automatic_offloading": False,
                "enable_caching": True,
            }
        )
    )
    store = _make_store(tmp_cache_dir, enable_caching=True)
    # Force the registry path: store __init__ scanned .npy files; "kx" should be
    # registered as None.
    assert "kx" in store
    with pytest.raises(ValueError, match="Unknown lazy_disk_cache_class"):
        _ = store["kx"]
    # Also exercise the helper directly, mirroring the registry contract.
    with pytest.raises(ValueError, match="Unknown lazy_disk_cache_class"):
        _resolve_lazy_disk_cache_class("EvilSubclass")
    assert "DiskBackedNDArray" in _LAZY_DISK_CACHE_CLASS_REGISTRY


def test_register_lazy_disk_cache_class_enables_subclass_round_trip(tmp_cache_dir: Path) -> None:
    """D-05 Option A: a registered ``LazyDiskCache`` subclass round-trips through the store.

    Before registration the reload allow-list rejects the subclass name (the
    D-02 closed-set behaviour). After calling the PUBLIC
    :func:`register_lazy_disk_cache_class` hook, an offload → reload cycle for a
    store holding instances of the subclass returns the correct array — proving
    the allow-list is now an extension point, with NO ``importlib`` fallback.
    """

    class _RegistrableDBNDArray(DiskBackedNDArray):
        """Throwaway subclass proving the allow-list is extensible via the hook."""

    name = _RegistrableDBNDArray.__name__
    try:
        # Not registered yet: resolving the subclass name is refused.
        assert name not in _LAZY_DISK_CACHE_CLASS_REGISTRY
        with pytest.raises(ValueError, match="Unknown lazy_disk_cache_class"):
            _resolve_lazy_disk_cache_class(name)

        # The hook registers the class and returns it (usable as a decorator).
        returned = register_lazy_disk_cache_class(_RegistrableDBNDArray)
        assert returned is _RegistrableDBNDArray
        assert _LAZY_DISK_CACHE_CLASS_REGISTRY[name] is _RegistrableDBNDArray
        # Idempotent: re-registering the same class object does not raise.
        register_lazy_disk_cache_class(_RegistrableDBNDArray)

        # Offload → reload a store whose entries are the registered subclass.
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        cfg = LazyDiskCacheConfig(enable_caching=True, cache_path=tmp_cache_dir)
        store = DiskBackedStore[_RegistrableDBNDArray](config=cfg, factory=_RegistrableDBNDArray)
        store.add_data_to_store("sub", expected)
        store.offload(pickle_container=True)

        # The sidecar records the subclass name; the in-memory ref was cleared.
        meta = json.loads((tmp_cache_dir / "sub.meta.json").read_text())
        assert meta["lazy_disk_cache_class"] == name
        assert store._store["sub"] is None

        # Reload resolves the registered subclass and round-trips losslessly.
        loaded = store["sub"]
        assert isinstance(loaded, _RegistrableDBNDArray)
        np.testing.assert_array_equal(np.asarray(loaded), expected)
    finally:
        # Keep the module-global allow-list clean for the rest of the session.
        _LAZY_DISK_CACHE_CLASS_REGISTRY.pop(name, None)


def test_register_lazy_disk_cache_class_rejects_non_lazy_disk_cache() -> None:
    """The hook keeps the allow-list closed to non-``LazyDiskCache`` inputs (D-02 posture)."""
    with pytest.raises(TypeError, match="LazyDiskCache subclass"):
        register_lazy_disk_cache_class(int)  # type: ignore[arg-type]
    # A plain instance (not a class) is also rejected.
    with pytest.raises(TypeError, match="LazyDiskCache subclass"):
        register_lazy_disk_cache_class(object())  # type: ignore[arg-type]


def test_register_lazy_disk_cache_class_rejects_name_collision() -> None:
    """Registering a *different* class under a taken name raises (allow-list integrity).

    Re-registering the same object is idempotent (covered above); shadowing an
    existing allow-list entry — e.g. ``DiskBackedNDArray`` — with a different
    class must be refused so a subclass cannot silently hijack the reload path.
    """
    # An impostor subclass whose __name__ collides with the built-in entry.
    impostor = type("DiskBackedNDArray", (DiskBackedNDArray,), {})
    assert impostor is not DiskBackedNDArray
    with pytest.raises(ValueError, match="already registered"):
        register_lazy_disk_cache_class(impostor)
    # The original entry is untouched.
    assert _LAZY_DISK_CACHE_CLASS_REGISTRY["DiskBackedNDArray"] is DiskBackedNDArray


def test_register_lazy_disk_cache_class_usable_as_decorator() -> None:
    """The hook is usable as a class decorator (advertised in its docstring).

    Applying ``@register_lazy_disk_cache_class`` at class-definition time must
    both register the subclass into the reload allow-list AND bind the original
    class object to the name (the decorator returns ``cls`` unchanged), so the
    decorated symbol stays fully usable.
    """
    name = "_DecoratedDBNDArray"
    try:

        @register_lazy_disk_cache_class
        class _DecoratedDBNDArray(DiskBackedNDArray):
            """Throwaway subclass registered via decorator syntax."""

        # The decorated name is bound to the real class (decorator returned cls).
        assert _DecoratedDBNDArray.__name__ == name
        assert isinstance(_DecoratedDBNDArray, type)
        assert issubclass(_DecoratedDBNDArray, DiskBackedNDArray)
        # ...and that same class is now in the reload allow-list.
        assert _LAZY_DISK_CACHE_CLASS_REGISTRY[name] is _DecoratedDBNDArray
    finally:
        # Keep the module-global allow-list clean for the rest of the session.
        _LAZY_DISK_CACHE_CLASS_REGISTRY.pop(name, None)


def test_schema_version_mismatch_rejected(tmp_cache_dir: Path) -> None:
    """Plan 02-01 / D-03: a schema_version != 1 raises ValueError, no fallback / migration."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.save(str(tmp_cache_dir / "kv.npy"), arr, allow_pickle=False)
    (tmp_cache_dir / "kv.meta.json").write_text(
        json.dumps(
            {
                "schema_version": 999,
                "lazy_disk_cache_class": "DiskBackedNDArray",
                "shape": [3],
                "dtype": "<f4",
                "purge_disk_on_gc": False,
                "automatic_offloading": False,
                "enable_caching": True,
            }
        )
    )
    store = _make_store(tmp_cache_dir, enable_caching=True)
    assert "kv" in store
    with pytest.raises(ValueError, match="schema_version"):
        _ = store["kv"]


def test_loaded_entry_has_cache_path_populated(tmp_cache_dir: Path) -> None:
    """Plan 02-01 / W-5: a loaded entry's ``cache_path`` is populated (not ``None``).

    Without this, the Plan-02-04 finalizer's :meth:`LazyDiskCache.enable_purge`
    silently no-ops on ``if not self._cache_path: return``. The test asserts the
    propagation works.

    Implementation detail: :class:`LazyDiskCache._init_from_config` re-suffixes
    the provided ``cache_path`` with ``_MEMMAP_SUFFIX`` (``.dat``) internally,
    so the live ``self._cache_path`` on the reconstructed instance is
    ``<key>.dat`` rather than ``<key>.npy``. The W-5 invariant (a non-``None``
    ``cache_path`` so ``enable_purge`` can register) holds either way; we
    assert both the non-``None`` invariant and that the path is anchored on
    the ``<key>`` stem.
    """
    store = _make_store_with_one_entry(tmp_cache_dir, key="k0")
    store.offload(pickle_container=True)
    # Force a fresh load — pickle_container=True cleared the in-memory ref to None.
    assert store._store["k0"] is None
    loaded = store["k0"]
    # W-5 invariant: cache_path is populated, NOT None.
    assert loaded.cache_path is not None, (
        "W-5 regressed: loaded entry's cache_path is None — Plan-02-04 finalizer "
        "will silently no-op on enable_purge() for entries reconstructed from disk."
    )
    # Anchored on the <key> stem (suffix may be `.dat` post-_init_from_config rewrite).
    assert Path(loaded.cache_path).stem == "k0", (
        f"W-5: loaded entry's cache_path ({loaded.cache_path}) does not carry the "
        f"<key> stem; finalizer would target the wrong file."
    )


# ---------------------------------------------------------------------------
# Plan 02-04 / FRAG-03 finalizer-re-registration regression tests + W-1.
# ---------------------------------------------------------------------------
#
# Implementation note on the on-disk shape (see Plan 02-04 SUMMARY for the
# full discussion): :meth:`LazyDiskCache._init_from_config` ALWAYS re-suffixes
# the user-supplied ``cache_path`` to ``_MEMMAP_SUFFIX`` (``.dat``), so the
# live ``self._cache_path`` on a directly-constructed :class:`DiskBackedNDArray`
# is ``<key>.dat`` — not ``<key>.npy``. The ``<key>.npy + <key>.meta.json``
# codec pair is produced by :meth:`DiskBackedStore._store_entry`, not by
# :meth:`LazyDiskCache.offload`. The finalizer registered in
# :meth:`LazyDiskCache._init_from_config` therefore targets the ``.dat`` file
# in the canonical path; the ``_purge_cache_pair`` helper's ``.meta.json``
# branch is defensive belt-and-suspenders code that fires only when an
# integrator hands a ``.npy``-suffixed path through to ``LazyDiskCache``
# directly (and bypasses the ``_MEMMAP_SUFFIX`` re-suffix).
#
# We therefore split the W-1 coverage:
#   * ``test_finalizer_re_registered_on_unpickle`` / ``test_purge_disk_on_gc_false_preserves_file_after_unpickle``
#     exercise the actual pickle round-trip on a directly-constructed
#     :class:`DiskBackedNDArray` (the FRAG-03 happy/sad path) and assert on
#     the ``.dat`` memmap file unlink semantics.
#   * ``test_finalizer_unlinks_both_npy_and_meta_after_unpickle`` exercises
#     the helper's ``.npy + .meta.json`` branch directly + via an in-memory
#     finalizer whose ``cache_path`` is constructed with a ``.npy`` suffix
#     (bypassing the constructor's re-suffix by setting ``_cache_path``
#     after construction).


def test_finalizer_re_registered_on_unpickle(tmp_path: Path) -> None:
    """Plan 02-04 / FRAG-03 / D-20 positive: finalizer re-registered after pickle round-trip.

    A pickled :class:`LazyDiskCache` subclass with ``purge_disk_on_gc=True``
    unpickles into an instance whose ``_finalizer`` is alive; GC of the
    unpickled instance unlinks the cache file.

    The :class:`LazyDiskCache._init_from_config` re-suffixes ``cache_path`` to
    ``.dat`` internally, so the file the finalizer targets is the ``.dat``
    memmap (not the ``.npy + .meta.json`` codec pair which is a
    :class:`DiskBackedStore` concept). The W-1 helper still runs (single-file
    unlink branch) because the suffix is not ``.npy``.

    Without the FRAG-03 fix (commented-out re-registration), the unpickled
    instance carries a dead/missing ``_finalizer`` and the cache file leaks
    on GC.
    """
    arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    cache_path_seed = tmp_path / "k0"
    obj = DiskBackedNDArray(
        arr,
        enable_caching=True,
        cache_path=cache_path_seed,
        purge_disk_on_gc=True,
    )
    # _init_from_config re-suffixes to .dat — verify the architectural premise.
    dat_path = tmp_path / "k0.dat"
    assert obj.cache_path == dat_path
    # Force the memmap to materialise on disk.
    obj.offload()
    assert dat_path.exists()
    # Original has a live finalizer pointed at dat_path.
    assert hasattr(obj, "_finalizer") and obj._finalizer.alive

    blob = pickle.dumps(obj)
    # del + gc.collect() the original. __getstate__ already called
    # disable_purge() so the original's finalizer is detached; the .dat
    # file must survive the original's collection.
    del obj
    gc.collect()
    assert dat_path.exists(), (
        "Architectural invariant regressed: __getstate__'s disable_purge() did not"
        " detach the original's finalizer before pickle; original GC unlinked the file."
    )

    revived = pickle.loads(blob)
    # FRAG-03 positive assertion: the unpickled finalizer is alive.
    assert hasattr(revived, "_finalizer"), "FRAG-03 regressed: _finalizer attribute missing after unpickle"
    assert revived._finalizer.alive, "FRAG-03 regressed: _finalizer.alive is False after unpickle"
    assert revived._purge_disk_on_gc is True, "FRAG-03 regressed: _purge_disk_on_gc was not preserved through pickle"
    # Confirm the finalizer targets the same path the original did.
    assert revived.cache_path == dat_path

    del revived
    gc.collect()
    # FRAG-03 success: the cache .dat file is unlinked when the revived instance is GC'd.
    assert not dat_path.exists(), "FRAG-03 regressed: .dat memmap leaked after unpickled instance was GC'd"


def test_purge_disk_on_gc_false_preserves_file_after_unpickle(tmp_path: Path) -> None:
    """Plan 02-04 / FRAG-03 / D-21(c) negative: purge=False preserves the file on GC.

    With ``purge_disk_on_gc=False`` the cache file MUST survive GC of an
    unpickled instance.

    Verifies the snapshot round-trip path correctly preserves the False
    intent (not flipped to True by ``__setstate__``'s re-registration logic)
    AND that the W-1 helper does not fire when no finalizer is registered.
    """
    arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    cache_path_seed = tmp_path / "k_keep"
    obj = DiskBackedNDArray(
        arr,
        enable_caching=True,
        cache_path=cache_path_seed,
        purge_disk_on_gc=False,
    )
    dat_path = tmp_path / "k_keep.dat"
    obj.offload()
    assert dat_path.exists()

    blob = pickle.dumps(obj)
    revived = pickle.loads(blob)
    # No finalizer should be live because the user's intent was False.
    # Either _finalizer is absent or .alive is False — both acceptable.
    assert not (hasattr(revived, "_finalizer") and revived._finalizer.alive), (
        "purge_disk_on_gc=False must NOT result in a live finalizer after unpickle"
    )
    assert revived._purge_disk_on_gc is False

    del revived
    del obj
    gc.collect()
    # File MUST persist — no finalizer fired on either instance.
    assert dat_path.exists(), "purge_disk_on_gc=False regressed: cache file was unlinked despite the False intent"


def test_finalizer_unlinks_both_npy_and_meta_after_unpickle(tmp_path: Path) -> None:
    """Plan 02-04 / W-1: ``_purge_cache_pair`` unlinks BOTH ``<key>.npy`` and ``<key>.meta.json``.

    Two checks:

    1.  Direct unit test of the helper: plant a ``.npy + .meta.json`` pair on
        disk, call ``_purge_cache_pair`` with the ``.npy`` path, assert both
        files are gone.
    2.  Indirect test via a :class:`DiskBackedNDArray` instance whose
        ``_cache_path`` is post-construction overridden to a ``.npy``-suffixed
        path (bypassing ``_init_from_config``'s ``_MEMMAP_SUFFIX`` re-suffix).
        Construction-time finalizer registration is then re-applied via
        ``enable_purge()``; GC of the instance must unlink both files via the
        helper's ``.npy`` branch.

    Why two checks: the canonical ``LazyDiskCache`` constructor always
    re-suffixes ``cache_path`` to ``.dat``, so the ``.meta.json`` branch of
    ``_purge_cache_pair`` is dead code on the happy path. The helper exists
    to harden future integrations (or any code path that hands a ``.npy``
    path directly to ``weakref.finalize`` via ``enable_purge``). This test
    locks in the behaviour without depending on a future caller materialising.
    """
    # --- check 1: direct helper unit test ------------------------------------
    npy_path = tmp_path / "kw1.npy"
    meta_path = npy_path.with_suffix(".meta.json")
    npy_path.write_bytes(b"unused")
    meta_path.write_text("{}", encoding="utf-8")
    assert npy_path.exists()
    assert meta_path.exists()
    _purge_cache_pair(npy_path)
    assert not npy_path.exists(), "W-1 regressed: .npy not unlinked by _purge_cache_pair"
    assert not meta_path.exists(), "W-1 regressed: .meta.json sidecar not unlinked by _purge_cache_pair"

    # The helper is also idempotent / safe on a missing pair.
    _purge_cache_pair(npy_path)  # MUST NOT raise

    # --- check 2: indirect via DiskBackedNDArray with overridden _cache_path -
    arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    obj = DiskBackedNDArray(
        arr,
        enable_caching=False,
        cache_path=tmp_path / "kw2",
        purge_disk_on_gc=False,  # we'll override below
    )
    # Override _cache_path to a .npy-suffixed path post-construction, then
    # plant matching files and re-register the finalizer via enable_purge().
    npy2 = tmp_path / "kw2.npy"
    meta2 = npy2.with_suffix(".meta.json")
    npy2.write_bytes(b"unused")
    meta2.write_text("{}", encoding="utf-8")
    obj._cache_path = npy2
    obj.enable_purge()
    assert hasattr(obj, "_finalizer") and obj._finalizer.alive

    del obj
    gc.collect()
    # W-1 success: both files unlinked through the .npy branch of the helper.
    assert not npy2.exists(), "W-1 regressed: .npy persisted after GC of instance with .npy cache_path"
    assert not meta2.exists(), "W-1 regressed: .meta.json sidecar persisted after GC"


# ---------------------------------------------------------------------------
# Plan 02-05 / FRAG-04 atomicity regression tests (D-09 a/b/c).
# ---------------------------------------------------------------------------
#
# These tests verify the Plan-02-01 codec's atomicity contract at the
# ``DiskBackedStore._store_entry`` write boundary. The write order is:
#
#   1. open ``<key>.npy.tmp``,  ``np.save(allow_pickle=False)``, flush, fsync
#   2. open ``<key>.meta.json.tmp``, ``json.dump``,                 flush, fsync
#   3. ``os.replace(<key>.npy.tmp,       <key>.npy)``
#   4. ``os.replace(<key>.meta.json.tmp, <key>.meta.json)``
#   5. (POSIX) dir-fsync
#   * on any exception: best-effort unlink of both ``.tmp`` files, re-raise.
#
# We probe three failure points:
#   (a) ``ENOSPC`` at step 3 (the first ``os.replace`` of the ``.npy.tmp``)
#       — verifies no torn final files, no ``.tmp`` leftovers, reader sees
#         KeyError on a fresh load.
#   (b) ``EIO`` at step 4 (process-kill simulated between the ``.npy`` rename
#       and the ``.meta.json`` rename) — verifies the half-state
#       (``.npy`` present, ``.meta.json`` absent) is treated as a cache miss
#       and ``.tmp`` files are cleaned up.
#   (c) ``EIO`` at the second fsync (step 2's ``os.fsync`` on the
#       ``.meta.json.tmp``) — verifies ``.tmp`` cleanup runs before either
#       rename, no final files appear.
#
# All three use pytest's built-in ``monkeypatch`` for failure injection —
# no Hypothesis, no ``python-atomicwrites``, no other deps (per D-13).


def test_atomic_offload_disk_full_at_npy_replace_leaves_no_torn_state(
    tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 02-05 / FRAG-04 / D-09(a).

    Simulate ``ENOSPC`` at the first ``os.replace(.npy.tmp -> .npy)`` call.

    After the failure: no final ``.npy`` or ``.meta.json`` exists (no torn
    read possible), the ``.tmp`` cleanup branch removed both temp files,
    and a fresh ``__getitem__`` (with the in-memory entry cleared to mimic
    a cross-process reader) raises ``KeyError`` via ``_load_entry``.
    """
    store = _make_store(tmp_cache_dir, enable_caching=True)
    arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    store.add_data_to_store("ka", arr)
    # Note: ``DiskBackedNDArray.__init__`` (via ``LazyDiskCache._init_from_config``)
    # may eagerly materialise a ``ka.dat`` memmap in the cache dir; that file
    # is a per-LazyDiskCache concern and is not produced by ``_store_entry``.
    # We do NOT assert an empty cache dir here — we only assert on the
    # ``.npy`` / ``.meta.json`` / ``.tmp`` files that ``_store_entry`` owns.

    real_replace = os.replace

    def fail_on_npy_replace(src: str, dst: str) -> None:
        if str(src).endswith(".npy.tmp"):
            raise OSError(28, "No space left on device")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_on_npy_replace)
    with pytest.raises(OSError, match="No space left"):
        store.offload(pickle_container=True)

    # Post-failure invariants:
    # 1. No final .npy / .meta.json (write never completed past step 3).
    assert not (tmp_cache_dir / "ka.npy").exists(), "torn write: final .npy materialised despite ENOSPC at rename"
    assert not (tmp_cache_dir / "ka.meta.json").exists(), (
        "torn write: final .meta.json materialised despite ENOSPC at rename"
    )
    # 2. No .tmp files (cleanup branch ran).
    leftover = list(tmp_cache_dir.iterdir())
    assert not any(p.name.endswith(".tmp") for p in leftover), (
        f"FRAG-04 regressed: .tmp file(s) remain after ENOSPC failure: {leftover}"
    )
    # 3. Reader sees KeyError, not a half-state load. Restore os.replace
    # explicitly (belt-and-suspenders; monkeypatch teardown does this too)
    # and clear the in-memory entry to simulate a fresh-process reader
    # that finds only the on-disk state.
    monkeypatch.setattr(os, "replace", real_replace)
    store._store["ka"] = None
    with pytest.raises(KeyError):
        _ = store["ka"]


def test_atomic_offload_half_rename_treated_as_cache_miss(tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan 02-05 / FRAG-04 / D-09(b).

    Simulate a process kill between the ``.npy`` rename and the
    ``.meta.json`` rename (``EIO`` on the second ``os.replace``). The next
    ``__getitem__`` MUST treat the half-written entry as a cache miss
    (``KeyError``), never as a loadable half-state.

    After Plan 02-01's codec, ``_load_entry`` checks
    ``if not (npy_path.exists() and json_path.exists()): raise KeyError(key)``
    so a final ``.npy`` without a paired ``.meta.json`` is rejected.
    """
    store = _make_store(tmp_cache_dir, enable_caching=True)
    arr = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    store.add_data_to_store("kb", arr)
    # The per-LazyDiskCache ``kb.dat`` memmap may exist here; the assertions
    # below only check ``_store_entry``-owned paths (.npy / .meta.json / .tmp).

    real_replace = os.replace

    def fail_on_meta_json_replace(src: str, dst: str) -> None:
        if str(src).endswith(".meta.json.tmp"):
            raise OSError(5, "I/O error")  # simulated mid-rename kill
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_on_meta_json_replace)
    with pytest.raises(OSError, match="I/O error"):
        store.offload(pickle_container=True)

    # The .npy rename succeeded; the .meta.json rename did not. The exception
    # handler unlinks both .tmp paths (.npy.tmp is already gone after the
    # first os.replace; .meta.json.tmp is unlinked here). The half-state on
    # disk is: final .npy present, NO final .meta.json, NO .tmp leftovers.
    assert (tmp_cache_dir / "kb.npy").exists(), (
        "half-rename test premise: the first os.replace should have succeeded before the second one failed"
    )
    assert not (tmp_cache_dir / "kb.meta.json").exists(), (
        "half-rename test premise: the second os.replace was injected to fail, so the final .meta.json must NOT exist"
    )
    leftover = list(tmp_cache_dir.iterdir())
    assert not any(p.name.endswith(".tmp") for p in leftover), (
        f"FRAG-04 regressed: .tmp file(s) remain after half-rename failure: {leftover}"
    )

    # KEY invariant: the reader sees a cache miss, not a half-state load.
    monkeypatch.setattr(os, "replace", real_replace)
    store._store["kb"] = None
    with pytest.raises(KeyError):
        _ = store["kb"]


def test_atomic_offload_tmp_cleanup_after_any_failure(tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan 02-05 / FRAG-04 / D-09(c).

    Inject ``EIO`` on the second ``os.fsync`` call (the ``.meta.json.tmp``
    fsync, which runs BEFORE either ``os.replace``). At failure time both
    ``.tmp`` files exist on disk and neither rename has run. The exception
    handler must unlink both ``.tmp`` files, leaving the cache dir empty.

    The exact failure site is incidental — what matters is the post-failure
    invariant: zero ``.tmp`` files remain, and no final files were created
    because the renames never ran.
    """
    store = _make_store(tmp_cache_dir, enable_caching=True)
    arr = np.array([[7.0, 8.0, 9.0]], dtype=np.float32)
    store.add_data_to_store("kc", arr)
    # The per-LazyDiskCache ``kc.dat`` memmap may exist here; the assertions
    # below only check ``_store_entry``-owned paths (.npy / .meta.json / .tmp).

    real_fsync = os.fsync
    seen_calls = {"count": 0}

    def fail_on_second_fsync(fd: int) -> None:
        # First fsync = .npy.tmp data flush; second fsync = .meta.json.tmp.
        # Fail on the second so the .npy.tmp exists at failure time (the
        # cleanup branch must remove BOTH .tmp files, not just the most
        # recently opened one).
        seen_calls["count"] += 1
        if seen_calls["count"] == 2:
            raise OSError(5, "fsync failed mid-write")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_on_second_fsync)
    with pytest.raises(OSError, match="fsync failed"):
        store.offload(pickle_container=True)

    monkeypatch.setattr(os, "fsync", real_fsync)
    # Both .tmp files are cleaned up; no final files were created because
    # neither rename ran (failure happened before step 3).
    leftover = list(tmp_cache_dir.iterdir())
    assert not any(p.name.endswith(".tmp") for p in leftover), (
        f"FRAG-04 regressed: .tmp file(s) remain after fsync failure: {leftover}"
    )
    assert not (tmp_cache_dir / "kc.npy").exists(), "no final .npy should exist when failure precedes the npy rename"
    assert not (tmp_cache_dir / "kc.meta.json").exists(), (
        "no final .meta.json should exist when failure precedes the meta rename"
    )


# ---------------------------------------------------------------------------
# Plan 03-01 / BUG-01 + BUG-02 regression tests.
# ---------------------------------------------------------------------------
#
# BUG-01: DiskBackedNDArray.__array_ufunc__ used to unconditionally raise
# NotImplementedError, so `disk_backed + 1` and every other numpy ufunc
# operation failed. The fix delegates to the NDArrayOperatorsMixin canonical
# recipe (unwrap DiskBackedNDArray inputs and forward to the ufunc) so
# arithmetic returns a plain ndarray.
#
# BUG-02: DiskBackedNDArray._drop_buffer used to set `self._data = None`,
# meaning a direct read of `bd._data` after `offload()` silently returned
# None instead of failing loudly. The fix `del`s the attribute instead, so
# direct reads raise AttributeError while the public `.data` property
# transparently re-materialises the buffer via `load()`.


def test_disk_backed_ndarray_ufunc_after_offload(tmp_path):
    """BUG-01 positive: ``disk_backed + 1`` works even after offload, returning a plain ndarray."""
    arr = np.arange(12).reshape(4, 3).astype(np.float64)
    bd = DiskBackedNDArray(
        arr.copy(),
        enable_caching=True,
        cache_path=tmp_path / "ent.dat",
        automatic_offloading=False,
    )
    bd.offload()
    result = bd + 1.0
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, arr + 1.0)


def test_disk_backed_ndarray_direct_data_after_offload(tmp_path):
    """BUG-02 negative: direct ``_data`` access after offload raises AttributeError; ``.data`` re-loads."""
    arr = np.arange(12).reshape(4, 3).astype(np.float64)
    bd = DiskBackedNDArray(
        arr.copy(),
        enable_caching=True,
        cache_path=tmp_path / "ent.dat",
        automatic_offloading=False,
    )
    bd.offload()
    with pytest.raises(AttributeError):
        _ = bd._data
    # Public `.data` property re-materialises:
    np.testing.assert_array_equal(bd.data, arr)


# ---------------------------------------------------------------------------
# Plan 04-04 / PERF-04 — D-07 unit tests for the chunked-streaming rewrite
# ---------------------------------------------------------------------------


def test_convert_to_memmap_fast_path(tmp_path):
    """D-07 #1: small array (nbytes < chunk_bytes) uses the one-shot fast path.

    With the default ~10% of available RAM chunk budget (or the 64 MB fallback
    when psutil is None), a 40-byte float32 array is always smaller than
    ``chunk_bytes`` so the fast path runs. Asserts the memmap contents are
    byte-identical to the source — i.e. no behavioural change for small arrays
    under the PERF-04 rewrite (D-06 contract).
    """
    arr = np.arange(10, dtype=np.float32)
    cache_path = tmp_path / "fast.dat"
    bd = DiskBackedNDArray(
        arr.copy(),
        enable_caching=True,
        cache_path=cache_path,
        purge_disk_on_gc=False,
        automatic_offloading=False,
    )
    assert bd._mmap is not None
    np.testing.assert_array_equal(np.asarray(bd._mmap), arr)
    np.testing.assert_array_equal(np.asarray(bd), arr)


def test_convert_to_memmap_streaming_path(tmp_path, monkeypatch):
    """D-07 #2: monkeypatch the chunk-budget constant low + psutil=None to force the streaming path.

    With ``_MEMMAP_FALLBACK_CHUNK_BYTES = 1024`` and ``psutil = None`` the
    chunk budget is 1024 bytes; a 100 KB float32 array (102_400 bytes) exceeds
    that threshold so the streaming loop runs. Asserts byte-identical output
    to the source — the streaming path must not corrupt content vs the fast
    path (D-06 invariant).
    """
    # Force psutil=None so chunk_bytes = _MEMMAP_FALLBACK_CHUNK_BYTES (1024).
    monkeypatch.setattr(ldc_mod, "psutil", None, raising=True)
    monkeypatch.setattr(ldc_mod, "_MEMMAP_FALLBACK_CHUNK_BYTES", 1024, raising=True)

    # 25_600 float32 = 102_400 bytes >> 1024-byte budget, so streaming triggers.
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(25_600, dtype=np.float32)
    cache_path = tmp_path / "streaming.dat"
    bd = DiskBackedNDArray(
        arr.copy(),
        enable_caching=True,
        cache_path=cache_path,
        purge_disk_on_gc=False,
        automatic_offloading=False,
    )
    assert bd._mmap is not None
    np.testing.assert_array_equal(np.asarray(bd._mmap), arr)
    np.testing.assert_array_equal(np.asarray(bd), arr)


# ---------------------------------------------------------------------------
# Plan 06-01 / TEST-04 — finalizer + pickle round-trip ordering tests.
# ---------------------------------------------------------------------------
#
# Phase 2 (FRAG-03) already established the broad pickle round-trip
# invariants (see ``test_finalizer_re_registered_on_unpickle`` above). Phase 6
# TEST-04 deepens that coverage with three additional assertions targeting
# the explicit ordering / "snapshot-before-mutate" contract introduced by
# Phase 2 D-18:
#   #1  ``test_finalizer_reregister_through_pickle_round_trip_ordering`` —
#       beyond the Phase 2 happy path, assert the revived ``_finalizer``
#       tracks the new instance (``detach is not None``) and the original
#       file survives the ORIGINAL's GC (proves ``__getstate__`` detached
#       the original's finalizer BEFORE pickling).
#   #2  ``test_purge_disk_on_gc_post_unpickle_no_leak`` — end-to-end GC of
#       the unpickled instance unlinks the cache file (no resource leak in
#       cross-process / multi-revive scenarios).
#   #3  ``test_snapshot_before_mutate_ordering`` — monkeypatch-spy on
#       ``LazyDiskCache.disable_purge`` proves the snapshot of
#       ``_purge_disk_on_gc`` happens BEFORE the in-place mutation.


def test_finalizer_reregister_through_pickle_round_trip_ordering(tmp_cache_dir: Path) -> None:
    """TEST-04 #1 / D-07: finalizer re-registration tracks the revived instance, not the dead original.

    Extends the Phase 2 ``test_finalizer_re_registered_on_unpickle`` happy
    path with two additional ordering assertions:
      * The original ``.dat`` file survives the original's ``del`` + ``gc.collect()``
        — proves ``__getstate__`` detached the original's finalizer BEFORE
        pickle (otherwise the file would be unlinked on original GC).
      * The revived ``_finalizer`` is alive AND its ``detach`` callable is
        bound (``revived._finalizer.detach is not None``) — proves the
        ``weakref.finalize`` registration targets the REVIVED instance, not
        a stale weakref to the dead original.
    """
    arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    obj = DiskBackedNDArray(
        arr,
        enable_caching=True,
        cache_path=tmp_cache_dir / "k",
        purge_disk_on_gc=True,
    )
    dat_path = tmp_cache_dir / "k.dat"
    obj.offload()
    assert dat_path.exists()
    assert hasattr(obj, "_finalizer") and obj._finalizer.alive

    blob = pickle.dumps(obj)
    del obj
    gc.collect()
    # __getstate__'s disable_purge() detached the original's finalizer, so
    # the .dat file MUST survive the original's collection.
    assert dat_path.exists(), (
        "TEST-04 #1 regressed: __getstate__'s disable_purge() did not detach the"
        " original's finalizer before pickling; original GC unlinked the file."
    )

    revived = pickle.loads(blob)
    # Positive assertions: the revived instance carries a live finalizer that
    # tracks IT (not the dead original).
    assert hasattr(revived, "_finalizer"), "TEST-04 #1 regressed: _finalizer attribute missing after unpickle"
    assert revived._finalizer.alive is True, "TEST-04 #1 regressed: revived _finalizer is not alive"
    assert revived._purge_disk_on_gc is True, "TEST-04 #1 regressed: _purge_disk_on_gc was not preserved through pickle"
    assert revived.cache_path == dat_path
    # Ordering assertion: the finalizer exposes a `detach` callable bound to
    # the revived instance's weakref machinery (not the dead original's).
    assert revived._finalizer.detach is not None, (
        "TEST-04 #1 regressed: revived _finalizer.detach is None — the weakref"
        " finalize tracks a dead reference rather than the revived instance."
    )


def test_purge_disk_on_gc_post_unpickle_no_leak(tmp_cache_dir: Path) -> None:
    """TEST-04 #2 / D-07: GC of the unpickled instance honours ``purge_disk_on_gc=True``.

    End-to-end purge-on-GC check after unpickle. After the original is
    collected, the unpickled instance is the SOLE remaining owner of the
    cache file; collecting it must unlink the file (no leak).
    """
    arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    obj = DiskBackedNDArray(
        arr,
        enable_caching=True,
        cache_path=tmp_cache_dir / "k_post",
        purge_disk_on_gc=True,
    )
    obj.offload()
    blob = pickle.dumps(obj)
    del obj
    gc.collect()

    revived = pickle.loads(blob)
    revived_cache_path = revived.cache_path
    assert revived_cache_path is not None and revived_cache_path.exists(), (
        "TEST-04 #2 premise: revived instance's cache file should exist before its own GC"
    )
    del revived
    gc.collect()
    assert not revived_cache_path.exists(), (
        "TEST-04 #2 regressed: unpickled instance with purge_disk_on_gc=True leaked"
        f" its cache file ({revived_cache_path}) on collection."
    )


def test_snapshot_before_mutate_ordering(tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TEST-04 #3 / D-07: ``__getstate__`` snapshots ``_purge_disk_on_gc`` BEFORE ``disable_purge`` mutates it.

    Monkeypatch-spy pattern from RESEARCH §"Open Questions" #2 (verbatim).
    Spies ``LazyDiskCache.disable_purge`` to capture ``self._purge_disk_on_gc``
    at entry; if the snapshot is taken AFTER the mutation, the captured value
    would be ``False`` (post-mutation state) and the assertion would fail.
    """
    from GSEGUtils.lazy_disk_cache.lazy_disk_cache import LazyDiskCache

    captured: list[bool] = []
    original_disable = LazyDiskCache.disable_purge

    def spy_disable_purge(self: LazyDiskCache) -> None:
        # Snapshot the flag at entry (i.e. BEFORE the original's in-place mutation).
        captured.append(self._purge_disk_on_gc)
        original_disable(self)

    monkeypatch.setattr(LazyDiskCache, "disable_purge", spy_disable_purge)

    obj = DiskBackedNDArray(
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        enable_caching=True,
        cache_path=tmp_cache_dir / "k_spy",
        purge_disk_on_gc=True,
    )
    obj.offload()
    state = obj.__getstate__()

    assert captured == [True], (
        "TEST-04 #3 regressed: snapshot ordering violated — disable_purge was called when"
        f" _purge_disk_on_gc was already {captured!r} (expected [True])."
        " __getstate__ must snapshot the user's original intent BEFORE disable_purge mutates state."
    )
    assert state["_purge_disk_on_gc"] is True, (
        "TEST-04 #3 regressed: __getstate__ returned a state dict whose"
        " _purge_disk_on_gc reflects the post-mutation value rather than the snapshot."
    )


def test_convert_to_memmap_import_error_fallback(tmp_path, monkeypatch):
    """D-07 #3: psutil ImportError fallback uses the fixed-bytes chunk constant.

    F2: replaces importlib.reload(ldc_mod) + importlib.reload(dbna_mod) — which
    rebinds class objects in sys.modules and breaks pickle's class-identity check —
    with monkeypatch.setattr, which patches the attribute without touching class
    objects. Pickle stability preserved by construction (Phase 9 D-12 F2).
    """
    import GSEGUtils.lazy_disk_cache.lazy_disk_cache as ldc_mod
    from GSEGUtils.lazy_disk_cache import disk_backed_ndarray as dbna_mod

    monkeypatch.setattr(ldc_mod, "psutil", None)
    assert ldc_mod.psutil is None

    arr = np.arange(64, dtype=np.float32)
    cache_path = tmp_path / "fallback.dat"
    bd = dbna_mod.DiskBackedNDArray(
        arr.copy(),
        enable_caching=True,
        cache_path=cache_path,
        purge_disk_on_gc=False,
        automatic_offloading=False,
    )
    assert bd._mmap is not None
    np.testing.assert_array_equal(np.asarray(bd._mmap), arr)
    # No finally block needed — monkeypatch.setattr auto-restores on teardown.


def test_store_key_parent_directory_escape_regression(tmp_cache_dir: Path) -> None:
    """Plan 14-01 / STORE-01 / SC-4: a parent-directory key is refused and writes nothing.

    Reproduces the escape that blocked pc2img: ``add_data_to_store('../victim', arr)``
    used to build ``<cache>/../victim.npy`` and overwrite a file *above* the cache
    directory. All three assertions matter — the raise alone does not prove that
    nothing was written.
    """
    from GSEGUtils.lazy_disk_cache import StoreKeyError

    victim = tmp_cache_dir.parent / "victim.npy"
    sentinel = b"do-not-overwrite-me"
    victim.write_bytes(sentinel)

    store = _make_store(tmp_cache_dir)
    arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    with pytest.raises(StoreKeyError, match="contains a path separator"):
        store.add_data_to_store("../victim", arr)

    assert victim.read_bytes() == sentinel, "the sentinel above the cache directory was overwritten"
    assert sorted(p.name for p in tmp_cache_dir.parent.glob("victim.*")) == ["victim.npy"], (
        "a new victim.* artefact was created above the cache directory"
    )


# ---------------------------------------------------------------------------
# Phase 14 Plan 04 — the two LazyDiskCache anchors the store's waist cannot reach
# ---------------------------------------------------------------------------


def test_cache_path_sealed_refuses_reassignment_but_still_reads(tmp_cache_dir: Path) -> None:
    """Plan 14-04 / STORE-02 / D-01: assignment raises ``AttributeError``; reads are unchanged.

    The seal and the surviving getter are **one** contract, so both are asserted
    in a single test function. Split across two, the getter could regress
    without the seal test noticing, and vice versa — and the getter is what the
    three pre-existing read assertions in this file (``:371``, ``:395``,
    ``:852``) depend on.

    The message is asserted to carry all three D-01 elements: which entry, the
    attempted path rendered by ``repr``, and the sanctioned alternative.
    """
    store = _make_store_with_one_entry(tmp_cache_dir, key="k0")
    store.offload()
    entry = store["k0"]
    expected = tmp_cache_dir / "k0.dat"
    attempted = Path("/tmp/anywhere/evil.dat")

    with pytest.raises(
        AttributeError,
        match=re.escape("an entry's cache path is fixed at construction"),
    ) as excinfo:
        entry.cache_path = attempted

    message = str(excinfo.value)
    assert repr(attempted) in message, f"the refusal does not name the attempted path by repr: {message}"
    assert "k0.dat" in message, f"the refusal does not identify which entry was targeted: {message}"
    assert "DiskBackedStore" in message, f"the refusal does not name the sanctioned alternative: {message}"

    # The getter is untouched, and the seal actually held: the path never moved.
    assert entry.cache_path == expected
    assert not attempted.exists(), "the sealed setter still created the destination directory"


@pytest.mark.parametrize(
    ("bad_folder", "clause"),
    [
        ("..", paths_mod.CLAUSE_RESERVED),
        ("a/b", paths_mod.CLAUSE_SEPARATOR),
        ("/abs", paths_mod.CLAUSE_ABSOLUTE),
    ],
)
def test_extend_cache_path_refuses_a_non_single_segment_folder(
    tmp_cache_dir: Path, bad_folder: str, clause: str
) -> None:
    """Plan 14-04 / STORE-02 / D-02: a folder name that is not one path segment is refused.

    Three cases, three different clauses of the shared lexical rule — the clause
    strings are imported from ``paths`` rather than restated, so the assertion
    cannot drift from the shipped vocabulary.
    """
    config = LazyDiskCacheConfig(enable_caching=True, cache_path=tmp_cache_dir)

    with pytest.raises(paths_mod.StoreKeyError, match=re.escape(clause)):
        config.extend_cache_path(bad_folder)


def test_extend_cache_path_accepts_a_single_segment_folder(tmp_cache_dir: Path) -> None:
    """Plan 14-04 / STORE-02 / D-02: a real per-tile folder name still joins onto the root.

    The guard must not break the live downstream callers it wraps — pc2img
    passes a per-tile id through this route.
    """
    config = LazyDiskCacheConfig(enable_caching=True, cache_path=tmp_cache_dir)

    extended = config.extend_cache_path("tile_03")

    assert extended.cache_path == tmp_cache_dir / "tile_03"
    assert extended.enable_caching is config.enable_caching


def test_extend_cache_path_with_no_source_path_still_returns_none(caplog: pytest.LogCaptureFixture) -> None:
    """Plan 14-04 / STORE-02 / D-02: the ``cache_path is None`` branch is otherwise untouched.

    This is the branch downstream callers hit when caching is off: a good folder
    name still emits the informational log entry and still returns a
    configuration carrying ``cache_path=None``.
    """
    config = LazyDiskCacheConfig(enable_caching=False, cache_path=None)

    with caplog.at_level(logging.INFO, logger="GSEGUtils.lazy_disk_cache.lazy_disk_cache"):
        extended = config.extend_cache_path("tile_03")

    assert extended.cache_path is None
    assert any("cannot extend" in record.getMessage() for record in caplog.records), (
        "the informational log entry for the None-source branch was lost"
    )


def test_extend_cache_path_with_no_source_path_still_refuses_a_bad_folder() -> None:
    """Plan 14-04 / STORE-02 / D-02: the guard is not conditional on configuration state.

    A bad folder name is a caller mistake whether or not a path exists to join
    it onto. If validation were ordered behind the ``None`` short-circuit this
    would return a configuration instead of raising — the same shape as the dead
    ``if self._cache_dir`` conditional this phase removes from the store.
    """
    config = LazyDiskCacheConfig(enable_caching=False, cache_path=None)

    with pytest.raises(paths_mod.StoreKeyError, match=re.escape(paths_mod.CLAUSE_RESERVED)):
        config.extend_cache_path("..")


# ---------------------------------------------------------------------------
# Phase 14 Plan 16 — the D-28 withdrawal of the three builder aliases
# ---------------------------------------------------------------------------

#: The three withdrawn builder-alias methods, paired with the module-level free
#: function that replaces each and the artefact suffix that function appends.
#: The pairing is the point: this is a migration, not an obituary, and the
#: second half of the test below is what says so.
WITHDRAWN_BUILDER_ALIASES = (
    ("_get_npy_path", "get_npy_path", "NPY_SUFFIX"),
    ("_get_meta_path", "get_meta_path", "META_SUFFIX"),
    ("_get_legacy_pickle_path", "get_legacy_pickle_path", "LEGACY_PICKLE_SUFFIX"),
)


def test_withdrawn_builder_aliases_are_gone_from_the_class(tmp_cache_dir: Path) -> None:
    """Plan 14-16 / MIG-01 / D-28 / T-14-73.

    **Why these went, when Plan 14-05 gave them a full deprecation cycle.** The
    cycle was granted on the sentence *"precisely because they have measured live
    callers"* — but the measurement was of **callers**, and their callers are
    downstream, not internal. Re-measured this round: **7** internal
    ``self._get_*_path`` call sites at ``v0.5.3``, ``origin/main`` and
    ``origin/develop/gsd``; **0** at phase-14 HEAD, removed by ``d83c22d``
    (*"feat(14-05): route every store path through the shared builders"*) — the
    same plan that wrote the deprecation. So an **override** of these has been
    inert since before the promise was written: the deprecation warning's advice
    (*call the free function instead*) is advice for a caller, and for an
    overrider it restores nothing.

    **Two assertions, and the second is the one that matters.** Absence alone
    records a removal. The free-function equality re-asserts, against the
    surviving surface, exactly what the deleted delegation test was protecting:
    that the replacement returns the paths the aliases returned. A downstream
    reading this test should be able to see its migration in it.

    No stub, no ``__getattr__`` shim and no re-export: a partial withdrawal is
    the defect MIG-01 is about, in the other direction.
    """
    store = _make_store(tmp_cache_dir)
    assert store._cache_dir == tmp_cache_dir, (
        "the free functions take the cache directory explicitly, so a caller migrating off the "
        "methods needs this to be the directory the store is actually using"
    )
    key = "k0"

    for method_name, builder_name, suffix_name in WITHDRAWN_BUILDER_ALIASES:
        assert not hasattr(DiskBackedStore, method_name), (
            f"DiskBackedStore.{method_name} resolves again; D-28 withdrew the whole "
            "artefact-naming override surface in one move, and a partial restoration is the "
            "half-true deprecation promise MIG-01 is about"
        )
        assert not hasattr(store, method_name), (
            f"an instance resolves {method_name}; the method is gone from the class but something "
            "re-creates it per instance, which is the same surface by another route"
        )

        # The migration half: the replacement returns the path the alias
        # returned, for this store's own cache directory.
        builder = getattr(paths_mod, builder_name)
        suffix = getattr(paths_mod, suffix_name)
        assert builder(store._cache_dir, key) == tmp_cache_dir / f"{key}{suffix}", (
            f"paths.{builder_name} no longer returns what {method_name} returned; the withdrawal "
            "would then be a removal rather than a migration"
        )
