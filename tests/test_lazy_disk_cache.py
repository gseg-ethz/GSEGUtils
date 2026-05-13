"""Regression tests for the Phase-2 lazy_disk_cache hardening.

Covers SEC-01 (codec swap), FRAG-04 (atomicity), and FRAG-03 (finalizer
re-registration). This file is created by Plan 02-01 with the codec +
cache_path-population tests; Plan 02-04 extends with finalizer tests;
Plan 02-05 extends with atomicity regression tests.
"""

import gc
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from GSEGUtils.lazy_disk_cache.disk_backed_ndarray import DiskBackedNDArray
from GSEGUtils.lazy_disk_cache.disk_backed_store import (
    _LAZY_DISK_CACHE_CLASS_REGISTRY,
    DiskBackedStore,
    _resolve_lazy_disk_cache_class,
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
