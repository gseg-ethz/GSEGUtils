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

"""PERF-04 microbenchmark for ``LazyDiskCache._convert_to_memmap``.

Times the chunked-streaming rewrite on a ~500 MB float32 array. Decorated with
``@pytest.mark.benchmark`` so the CI default (``-m "not benchmark"`` per Phase 4
D-31) skips it; opt-in locally via ``pytest -m benchmark``. Additionally
gated by ``@pytest.mark.skipif(os.environ.get("CI"))`` per RESEARCH.md Open
Question #1 — a 500 MB synthetic ndarray exceeds GitHub-Actions runner RAM
headroom even when the streaming path keeps peak RSS bounded.

D-32 record: pre-fix peak RSS ~500 MB + ~5–10 s wall time;
post-fix streaming peak ~50 MB + ~5–10 s I/O-bound. Captured offline.
"""

import os

import numpy as np
import pytest

from GSEGUtils.lazy_disk_cache.disk_backed_ndarray import DiskBackedNDArray


@pytest.mark.benchmark
@pytest.mark.skipif(
    bool(os.environ.get("CI")),
    reason="Heavy 500 MB bench — local-only (RESEARCH.md Open Question #1)",
)
def test_convert_to_memmap_500mb(benchmark, large_ndarray_500mb, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Time ``_convert_to_memmap`` on a ~500 MB float32 array (streaming path).

    Uses the session-scoped ``large_ndarray_500mb`` fixture from
    ``tests/benchmarks/conftest.py`` so construction cost is paid once per
    pytest invocation. Per-bench ``tmp_path`` keeps the on-disk memmap
    isolated across benchmark rounds.

    Sanity-asserts the resulting memmap shape matches the source so a
    silent shape-mismatch regression in the chunked write would fail the
    bench, not just slow it.
    """

    def setup() -> tuple[tuple, dict]:
        cache_path = tmp_path / f"bench_500mb_{benchmark.name}.dat"
        # New DiskBackedNDArray per round so each timed call hits a cold mmap.
        dbna = DiskBackedNDArray(
            large_ndarray_500mb,
            enable_caching=False,
            cache_path=cache_path,
            purge_disk_on_gc=True,
            automatic_offloading=False,
        )
        return (dbna,), {}

    def run(dbna: DiskBackedNDArray) -> None:
        dbna._convert_to_memmap()

    benchmark.pedantic(run, setup=setup, rounds=3, iterations=1, warmup_rounds=0)

    # Sanity check: build one more cache and confirm shape round-trip post-bench.
    cache_path = tmp_path / "bench_500mb_assert.dat"
    dbna = DiskBackedNDArray(
        large_ndarray_500mb,
        enable_caching=False,
        cache_path=cache_path,
        purge_disk_on_gc=True,
        automatic_offloading=False,
    )
    dbna._convert_to_memmap()
    assert dbna._mmap is not None
    assert dbna._mmap.shape == large_ndarray_500mb.shape
    assert dbna._mmap.dtype == large_ndarray_500mb.dtype
    np.testing.assert_array_equal(np.asarray(dbna._mmap[:1024]), large_ndarray_500mb[:1024])
