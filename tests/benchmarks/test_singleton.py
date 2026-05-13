"""PERF-05 microbenchmark for ``SingletonMeta.__call__``.

Times the post-init fast path of the lock-free metaclass call. Decorated with
``@pytest.mark.benchmark`` so the CI default (``-m "not benchmark"``, added by
plan 04-04 Task 2) skips it; opt-in locally via ``pytest -m benchmark``.

Plan 04-04 also lands the ``pytest-benchmark`` dev dep + the
``[tool.pytest.ini_options].markers`` registration, plus the shared
``singleton_class_fixture`` in ``tests/benchmarks/conftest.py``. Until those
land, this file is collectable but requires ``pytest-benchmark`` to actually
execute the ``benchmark`` fixture. By D-01 wave-1 order plan 04-05 ships
before plan 04-04, so an inline ``_BenchSingleton`` is used here; future plans
MAY refactor to consume the shared fixture once 04-04 lands.
"""

import pytest

from GSEGUtils.singleton import SingletonMeta


class _BenchSingleton(metaclass=SingletonMeta):
    """Minimal singleton class used to time the metaclass fast path."""


@pytest.mark.benchmark
def test_singleton_call_perf(benchmark) -> None:  # type: ignore[no-untyped-def]
    """Time the lock-free fast path of ``SingletonMeta.__call__``.

    The first call covers the slow-path one-off; ``pytest-benchmark`` then
    drives ``_BenchSingleton()`` repeatedly so the recorded numbers reflect the
    steady-state fast path (one ``cls._instances.get`` under the GIL, no lock
    acquire).
    """
    SingletonMeta._instances.pop(_BenchSingleton, None)
    _BenchSingleton()  # materialise via the slow path before timing the fast path.

    result = benchmark(_BenchSingleton)
    assert result is not None
