"""Opt-in microbenchmark suite for GSEGUtils (Phase 4).

Tests in this package are decorated ``@pytest.mark.benchmark`` and excluded
from CI via ``pytest -m "not benchmark"`` per Phase 4 D-31. Run locally with
``pytest -m benchmark``. Shared fixtures live in :mod:`tests.benchmarks.conftest`.
"""
