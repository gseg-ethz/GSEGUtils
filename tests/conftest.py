"""Shared fixtures for the GSEGUtils test suite.

Created by Plan 14-01 (Wave 0) as the fixture home the Phase-14 store-key and
containment tests block on: ``tmp_cache_dir``, ``symlinked_cache_dir`` and the
``make_store`` factory.

``test_lazy_disk_cache.py`` keeps its own file-local ``tmp_cache_dir`` and
``_make_store``; that duplication is deliberate, so the 24 pre-existing tests in
that module are shadowed by their own definitions and stay off this change's
blast radius.

``pytest-randomly`` shuffles test order on every run, so none of these fixtures
may carry cross-test state.
"""

from pathlib import Path
from typing import Protocol

import pytest

from GSEGUtils.lazy_disk_cache.disk_backed_ndarray import DiskBackedNDArray
from GSEGUtils.lazy_disk_cache.disk_backed_store import DiskBackedStore
from GSEGUtils.lazy_disk_cache.lazy_disk_cache import LazyDiskCacheConfig


class MakeStore(Protocol):
    """Callable protocol for the ``make_store`` factory fixture."""

    def __call__(
        self,
        cache_dir: Path,
        *,
        enable_caching: bool = True,
        purge_disk_on_gc: bool = False,
        automatic_offloading: bool = False,
    ) -> DiskBackedStore[DiskBackedNDArray]:
        """Build an empty ``DiskBackedStore`` pointed at ``cache_dir``."""
        ...


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Return a fresh, empty cache directory under the per-test ``tmp_path``."""
    d = tmp_path / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def symlinked_cache_dir(tmp_path: Path) -> Path:
    """Return a symlink pointing at a real, empty cache directory (STORE-03)."""
    real = tmp_path / "real"
    real.mkdir(parents=True, exist_ok=True)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    return link


@pytest.fixture
def make_store() -> MakeStore:
    """Return a factory building an empty ``DiskBackedStore`` on a given cache directory."""

    def _factory(
        cache_dir: Path,
        *,
        enable_caching: bool = True,
        purge_disk_on_gc: bool = False,
        automatic_offloading: bool = False,
    ) -> DiskBackedStore[DiskBackedNDArray]:
        cfg = LazyDiskCacheConfig(
            enable_caching=enable_caching,
            cache_path=cache_dir,
            purge_disk_on_gc=purge_disk_on_gc,
            automatic_offloading=automatic_offloading,
        )
        return DiskBackedStore[DiskBackedNDArray](config=cfg, factory=DiskBackedNDArray)

    return _factory
