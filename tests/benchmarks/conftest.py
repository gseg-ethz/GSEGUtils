# GSEGUtils – General utility functions and classes for GSEG research/projects
#
# Copyright (c) 2025–2026 ETH Zurich
# Department of Civil, Environmental and Geomatic Engineering (D-BAUG)
# Institute of Geodesy and Photogrammetry
# Geosensors and Engineering Geodesy
#
# Authors:
#   Nicholas Meyer
#   Jon Allemand
#
# SPDX-License-Identifier: BSD-3-Clause

"""Session-scoped synthetic fixtures for Phase 4 GSEGUtils benchmarks.

Per Phase 4 D-33: shared session-scoped fixtures for the
``tests/benchmarks/`` directory. Plan 04-04 (PERF-04) consumes
``large_ndarray_500mb``; plan 04-05 (PERF-05) MAY refactor to consume
``singleton_class_fixture`` (it currently inlines its own
``_BenchSingleton`` — the bench file there documents the future refactor).

Fixtures are session-scoped (single construction per `pytest` invocation)
because building a 500 MB float32 ndarray costs ~1 s; per-test allocation
would dominate the bench numbers themselves.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def large_ndarray_500mb() -> np.ndarray:
    """Return a 500 MB float32 ndarray for the PERF-04 streaming bench.

    Shape = (131_072_000,) so ``nbytes == 524_288_000`` (exactly 500 * 1024**2,
    near-RAM-size on a small CI runner but tractable on a developer machine).
    Deterministic RNG (``np.random.default_rng(0)``) per D-33 fixture-seed
    convention so re-runs see byte-identical input — useful for spotting
    noise vs real signal in ``pytest-benchmark`` output.
    """
    rng = np.random.default_rng(0)
    return rng.standard_normal((131_072_000,), dtype=np.float32)


@pytest.fixture(scope="session")
def disk_backed_500mb(tmp_path_factory: pytest.TempPathFactory, large_ndarray_500mb: np.ndarray):
    """Return a ``DiskBackedNDArray`` wrapping a 500 MB float32 array (PERF-04 bench).

    Session-scoped so the underlying memmap is reused across bench runs in
    the same pytest invocation. The session-scoped ``tmp_path_factory`` keeps
    the on-disk cache out of per-test ``tmp_path`` dirs that pytest reaps.
    """
    from GSEGUtils.lazy_disk_cache.disk_backed_ndarray import DiskBackedNDArray

    cache_dir = tmp_path_factory.mktemp("bench_lazy_disk_cache")
    return DiskBackedNDArray(
        large_ndarray_500mb,
        enable_caching=True,
        cache_path=cache_dir / "bench_500mb",
        purge_disk_on_gc=True,
        automatic_offloading=False,
    )


@pytest.fixture(scope="session")
def singleton_class_fixture():
    """Return a fresh ``SingletonMeta``-using class for the PERF-05 bench.

    Future plans may refactor ``tests/benchmarks/test_singleton.py`` (created
    by plan 04-05 BEFORE this conftest) to consume this fixture instead of
    its inline ``_BenchSingleton``. The class registry is cleared on fixture
    setup so first-call slow-path costs are isolated from prior tests.
    """
    from GSEGUtils.singleton import SingletonMeta

    class _BenchSingleton(metaclass=SingletonMeta):
        """Dummy class used to time the metaclass ``__call__`` fast path."""

    SingletonMeta._instances.pop(_BenchSingleton, None)
    return _BenchSingleton
