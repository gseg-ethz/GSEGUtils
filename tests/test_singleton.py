"""Regression tests for the Phase-4 lock-free ``SingletonMeta.__call__``.

Covers PERF-05 (D-08 double-checked locking, D-10 test suite). This file is
created by Plan 04-05 with three primary assertions plus one threaded bonus:

- ``test_singleton_identity`` (D-10 #1) -- two consecutive calls return the same object.
- ``test_singleton_reset_returns_new_instance`` -- registry-clear semantics preserved.
- ``test_lock_acquired_once_after_first_init`` (D-10 #2) -- post-init fast path is
  lock-free; ``cls._lock`` is acquired exactly once across the first call + 1000
  follow-ups.
- ``test_singleton_thread_safe_first_init`` (recommended bonus) -- 100 threads
  barrier-synchronised on first construction yield a single live instance.

The D-10 #3 benchmark lives at ``tests/benchmarks/test_singleton.py``.
"""

import threading

import pytest

from GSEGUtils.singleton import SingletonMeta


class _DummySingleton(metaclass=SingletonMeta):
    """Minimal singleton class used to exercise the metaclass under test."""

    def __init__(self) -> None:
        self.value = 42


@pytest.fixture(autouse=True)
def _reset_singleton_registry():
    """Clear ``_DummySingleton`` from the metaclass registry around each test.

    The metaclass registry is process-global state (``SingletonMeta._instances``),
    so each test must start from a clean slate or earlier tests' instances leak
    into the assertions (notably the lock-count test, which depends on the next
    call taking the slow path).
    """
    SingletonMeta._instances.pop(_DummySingleton, None)
    yield
    SingletonMeta._instances.pop(_DummySingleton, None)


def test_singleton_identity() -> None:
    """Two consecutive calls return the same object (D-10 #1 equivalence)."""
    a = _DummySingleton()
    b = _DummySingleton()
    assert a is b
    assert a.value == 42


def test_singleton_reset_returns_new_instance() -> None:
    """After clearing the registry, the next call constructs a fresh instance."""
    a = _DummySingleton()
    SingletonMeta._instances.pop(_DummySingleton, None)
    b = _DummySingleton()
    assert a is not b


def test_lock_acquired_once_after_first_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """D-10 #2 -- post-init fast path is lock-free.

    Replaces ``SingletonMeta._lock`` with a counting wrapper that delegates
    ``__enter__`` / ``__exit__`` to the original ``RLock`` while incrementing a
    counter on every acquire. After one slow-path construction followed by 1000
    follow-up calls, the counter must still read 1 -- proving the follow-ups
    took the lock-free fast path.
    """
    original_lock = SingletonMeta._lock
    counter = {"acquires": 0}

    class _CountingLock:
        def __enter__(self) -> object:
            counter["acquires"] += 1
            return original_lock.__enter__()

        def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> object:
            return original_lock.__exit__(exc_type, exc_val, exc_tb)

    monkeypatch.setattr(SingletonMeta, "_lock", _CountingLock())

    _ = _DummySingleton()  # first call: slow path acquires the lock once.
    first_count = counter["acquires"]
    for _ in range(1000):
        _ = _DummySingleton()  # fast path: no lock acquire expected.
    final_count = counter["acquires"]

    assert first_count == 1
    assert final_count == 1


def test_singleton_thread_safe_first_init() -> None:
    """100 threads racing on first construction yield a single live instance.

    Bonus coverage for the slow-path race window: a ``threading.Barrier``
    releases 100 worker threads simultaneously into ``_DummySingleton()``;
    the re-check inside ``with cls._lock:`` must guarantee that at most one
    thread runs ``super().__call__`` and every result is the same object.
    """
    results: list[object] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(100)

    def _call() -> None:
        barrier.wait()
        instance = _DummySingleton()
        with results_lock:
            results.append(instance)

    threads = [threading.Thread(target=_call) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 100
    assert all(r is results[0] for r in results)
