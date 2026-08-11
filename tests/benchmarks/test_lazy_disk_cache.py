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

D-15 record (plan 15-05, 2026-08-11) — what the lost reopen optimisation costs
-----------------------------------------------------------------------------
15-CONTEXT D-15 makes this measurement **mandatory**: the atomic ``.dat`` route
writes the full array on every conversion, even when a valid ``<key>.dat`` is
already present, so the ``r+`` reopen optimisation is gone. Measured here on the
reopen path — convert once so a valid ``<key>.dat`` exists, then time a *second*
conversion of an equal-sized array on that same entry.

Pre-change numbers were taken against the baseline blob at
``$BASE = 0c26dd6`` (this plan's pre-execution branch tip; blob
``85c0988``), through the **out-of-tree** baseline procedure 15-02 established —
a second tree built under ``PYTHONPATH``, never ``git stash``, which captures
nothing once a task is committed and would silently measure the post-change code
twice. Pre and post runs were **interleaved** one for one, so a slow patch of
host time cannot land entirely on one side. Host: 16 cores, 89.7 GB available
RAM, ``/`` on ``/dev/sdg``. Wall time in seconds.

===================================== ==== ======== ============= ==============
cell                                  runs  median   min–max       branch taken
===================================== ==== ======== ============= ==============
50 MB float32, pre-change                7   0.0481  0.0412–0.0508 one-shot
50 MB float32, post-change               7   0.1106  0.1001–0.1286 one-shot
500 MB float32, pre-change               5   0.3970  0.3533–0.4136 one-shot
500 MB float32, post-change              5   0.8690  0.8554–1.0746 one-shot
500 MB forced-streaming, pre-change*     5   0.0641  0.0512–0.0702 chunked
500 MB forced-streaming, post-change*    5   0.6001  0.5489–0.6508 chunked
===================================== ==== ======== ============= ==============

Every individual run, in order taken::

    50 MB  pre  0.0481 0.0508 0.0497 0.0439 0.0412 0.0444 0.0503
    50 MB  post 0.1001 0.1155 0.1106 0.1114 0.1043 0.1024 0.1286
    500 MB pre  0.3670 0.3533 0.3970 0.4136 0.3975
    500 MB post 0.8554 0.8690 0.8625 1.0132 1.0746
    500 MB forced-streaming pre  0.0512 0.0527 0.0641 0.0648 0.0702
    500 MB forced-streaming post 0.5584 0.5489 0.6508 0.6001 0.6420

**On the two starred rows — a recorded finding, not a planned cell.** The plan's table labels the 500 MB
case "chunked streaming path". On this host it is not: ``psutil`` reports
89.7 GB available, so the D-04 chunk budget is ~9.2 GB and a 500 MB array takes
the *one-shot* branch. The same is true of ``test_convert_to_memmap_500mb``
above, which has therefore never measured the streaming path on a machine like
this one. The starred rows force the streaming branch (``psutil`` nulled,
``_MEMMAP_FALLBACK_CHUNK_BYTES`` pinned to 64 MB → 8 chunks) so the chunked route
is covered too. They are supplementary to the four cells D-15 requires, not a
substitute for them.

**The threshold, in its two-condition form.** The change is *material* if the
reopen path's median wall time regresses by more than **2x** **and** the pre- and
post-change min–max spreads do **not** overlap. Independently, a median
regression of more than **5 s absolute at 500 MB** is material regardless of
spread. If the medians regress but the spreads overlap, the verdict is
``inconclusive`` — a real outcome, not a run to repeat until it resolves.

**The threshold is advisory and gates nothing.** It selects which branch of the
plan-15-05 Task 4 checkpoint question the maintainer is asked; it does not fail
this benchmark, does not fail the suite, and does not itself authorise the
"rename only when content would change" fallback (rejected in D-15 for
complexity rather than for lack of merit). A virtualised runner having a bad
minute is not a reason to adopt a hot-path optimisation.

**Verdict: material.** 50 MB regresses 2.30x (+0.0625 s) and 500 MB regresses
2.19x (+0.4720 s); neither pair of spreads overlaps, so the first condition is
met at both required cells. The second condition is **not** met — the absolute
500 MB regression is +0.47 s, two orders of magnitude below the 5 s bar. The
forced-streaming rows are worse in ratio (9.36x) and comparable in absolute
terms (+0.5360 s), because the pre-change chunked route wrote into an
already-allocated file and performed no fsync at all, while the new route
allocates a fresh temporary and fsyncs it. The honest summary is that the cost
is a consistent, reproducible doubling of a sub-second operation, not a new
multi-second stall. **Not resolved here:** it goes back to the maintainer at
plan 15-05 Task 4, and no fallback is implemented under any verdict.
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


@pytest.fixture(scope="module")
def ndarray_50mb() -> np.ndarray:
    """Return a 50 MB float32 ndarray for the D-15 reopen bench.

    Module-scoped rather than session-scoped, and defined here rather than in
    ``tests/benchmarks/conftest.py``, because plan 15-05 Task 3's acceptance
    criteria require this task's diff to touch exactly one file.
    """
    n = (50 * 1024 * 1024) // 4
    return np.arange(n, dtype=np.float32)


def _time_reopen(benchmark, source: np.ndarray, tmp_path, rounds: int) -> None:  # type: ignore[no-untyped-def]
    """Time a SECOND ``_convert_to_memmap`` on an entry that already has a valid ``.dat``.

    This is the route D-15 changes. Before plan 15-05 the second conversion
    reopened the existing file ``r+`` and overwrote it in place; now it writes a
    fresh ``<key>.dat.tmp`` and renames over. The setup performs the first
    conversion **outside** the timed region, so the number is the reopen cost
    and not the cost of creating the file.
    """
    counter = {"i": 0}

    def setup() -> tuple[tuple, dict]:
        counter["i"] += 1
        cache = tmp_path / f"reopen_{counter['i']}"
        cache.mkdir(parents=True, exist_ok=True)
        entry = DiskBackedNDArray(
            source,
            enable_caching=False,
            cache_path=cache / "k",
            purge_disk_on_gc=True,
            automatic_offloading=False,
        )
        # Untimed: produce the valid <key>.dat the reopen path starts from.
        entry._convert_to_memmap()
        entry._data = source + 1.0
        return (entry,), {}

    def run(entry: DiskBackedNDArray) -> None:
        entry._convert_to_memmap()

    benchmark.pedantic(run, setup=setup, rounds=rounds, iterations=1, warmup_rounds=0)


@pytest.mark.benchmark
@pytest.mark.skipif(
    bool(os.environ.get("CI")),
    reason="Local-only bench (RESEARCH.md Open Question #1)",
)
def test_convert_to_memmap_reopen_path_50mb(benchmark, ndarray_50mb, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Time the D-15 reopen path on a 50 MB float32 array.

    The smaller of the two array sizes 15-CONTEXT D-15 requires. See the module
    docstring for the four measured medians, their spreads, the two-condition
    threshold, its advisory-and-non-gating standing, and the recorded verdict.
    """
    _time_reopen(benchmark, ndarray_50mb, tmp_path, rounds=5)

    # Sanity: the reopen actually produced the NEW array, so a bench that got
    # faster by not writing anything would fail rather than look like a win.
    cache = tmp_path / "reopen_assert"
    cache.mkdir(parents=True, exist_ok=True)
    entry = DiskBackedNDArray(
        ndarray_50mb,
        enable_caching=False,
        cache_path=cache / "k",
        purge_disk_on_gc=True,
        automatic_offloading=False,
    )
    entry._convert_to_memmap()
    updated = ndarray_50mb + 1.0
    entry._data = updated
    entry._convert_to_memmap()
    assert entry._mmap is not None
    np.testing.assert_array_equal(np.asarray(entry._mmap[:1024]), updated[:1024])


@pytest.mark.benchmark
@pytest.mark.skipif(
    bool(os.environ.get("CI")),
    reason="Heavy 500 MB bench — local-only (RESEARCH.md Open Question #1)",
)
def test_convert_to_memmap_reopen_path_500mb(benchmark, large_ndarray_500mb, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Time the D-15 reopen path on a ~500 MB float32 array.

    The larger of the two array sizes 15-CONTEXT D-15 requires. Note the module
    docstring's starred finding: on a host with plenty of free RAM this size
    takes the *one-shot* branch, not the chunked streaming one, because the D-04
    chunk budget scales with available memory.
    """
    _time_reopen(benchmark, large_ndarray_500mb, tmp_path, rounds=5)
