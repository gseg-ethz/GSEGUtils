"""Plan 15-05 / STORE-08 / D-15 — atomicity of the ``<key>.dat`` write route.

Phase 14 gave the memmap artefact containment; this module holds the other
half of STORE-08: that no file-level write route in
:meth:`LazyDiskCache._convert_to_memmap` can tear a ``<key>.dat`` that was
previously good. The route writes ``<key>.dat.tmp``, flushes, fsyncs, renames
onto the final name and fsyncs the directory — the same sequence
``DiskBackedStore._store_entry`` already uses for the codec pair.

**Two invariants, deliberately tested apart (review finding F1 / D-15a).**
The *filesystem* invariant is that the previously-valid ``<key>.dat`` keeps its
bytes. The *object* invariant is that the live entry keeps reading them. They
are not the same claim, and the second is the one a byte-level assertion cannot
see: if the temporary mapping were adopted before the rename committed, the
final file's bytes would survive while the entry read through a temporary the
cleanup handler had just unlinked. Group B asserts the first; Group B2 asserts
the second.

**Which tests are POSIX-marked, and why the split is not uniform (D-15b / F2).**
The atomic route is POSIX-only by maintainer decision: renaming over an open,
mapped file is not permitted on Windows, so off POSIX the conversion falls back
to a direct write on the final name and is *not* torn-write-safe. Tests that
assert a property only the atomic route produces — Groups A, B, B2, C and F —
therefore carry :data:`POSIX_ONLY`. Groups D and E carry **no** marker on
purpose: containment on the temporary name is checked *before* the platform
branch, and a round trip must hold on every platform, so marking those would
hide a real regression off POSIX rather than describe a platform limit. A
future port inherits the marked tests as a to-do list rather than having to
rediscover them.

Assertion messages name STORE-08, or the decision, or the defect they defend,
so a red test reads as a finding rather than as a diff.
"""

import os
import stat
from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np
import pytest
from numpy.typing import DTypeLike

from GSEGUtils.lazy_disk_cache import lazy_disk_cache as ldc_module
from GSEGUtils.lazy_disk_cache import paths
from GSEGUtils.lazy_disk_cache.disk_backed_ndarray import DiskBackedNDArray
from GSEGUtils.lazy_disk_cache.paths import StoreContainmentError

#: One platform decision, one spelling — mirroring the single
#: ``os.name == "posix"`` predicate on the source side. Applied by name so the
#: predicate is never retyped, and so a reader can see at a glance which tests
#: describe a platform limit and which describe a universal property.
POSIX_ONLY = pytest.mark.skipif(
    os.name != "posix",
    reason="atomic memmap replace is POSIX-only (15-05 D-15b)",
)

_TMP_NAME_SUFFIX = paths.MEMMAP_SUFFIX + paths.TMP_SUFFIX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SourcedNDArray(DiskBackedNDArray):
    """A :class:`DiskBackedNDArray` whose conversion *source* can be swapped.

    Setting :attr:`_source` makes the next ``_convert_to_memmap`` write that
    array instead of the live buffer, **without touching the live buffer
    itself**. That separation is what makes Group B2 possible: assigning
    ``entry._data = other`` directly would destroy the very state the test is
    trying to observe, so a failed conversion would appear to have changed the
    object when in fact the test did.

    The seam is the one the base class already defines. ``_describe_buffer``
    returns ``(shape, dtype, source)`` and ``_convert_to_memmap`` consumes
    exactly that tuple, so overriding it needs no monkeypatching of numpy and
    no knowledge of the conversion's internals.
    """

    _source: Optional[Union[np.ndarray, "_ExplodingSource"]] = None

    def _describe_buffer(self) -> tuple[tuple[int, ...], DTypeLike, np.ndarray]:
        """Return the swapped-in source when one is set, else the real buffer."""
        src = self._source
        if src is None:
            return super()._describe_buffer()
        # `cast` rather than a real ndarray: the base class only ever reads
        # `shape`/`dtype`/`ndim`/`itemsize`/`nbytes` off this value and then
        # either coerces it with `np.array(..., copy=True)` or slices it, so
        # any array-like satisfies the contract. Group C depends on that.
        return src.shape, src.dtype, cast(np.ndarray, src)


class _ExplodingSource:
    """Array-like that delegates to a real ndarray and raises on a nominated access.

    The deterministic fault-injection seam review finding **F3** asked for, and
    the reason the two obvious instruments are not used anywhere in this file:

    * Patching ``np.memmap.flush`` injects at the **commit**, not mid-populate —
      the flush fires after the populate block has already written every byte,
      so a test built on it proves nothing about an interrupted populate.
    * Patching the source array's ``__getitem__`` cannot reach the one-shot
      fast path at all, because ``np.array(array, dtype=dtype, copy=True)``
      coerces through ``__array__`` and never indexes.

    This wrapper injects at whichever access the path under test actually
    performs, and counts both, so a test that stops driving its intended branch
    fails loudly instead of silently passing as a duplicate of its sibling.
    """

    def __init__(
        self,
        arr: np.ndarray,
        *,
        raise_on_array: bool = False,
        raise_after_chunks: Optional[int] = None,
    ) -> None:
        self._arr = arr
        self._raise_on_array = raise_on_array
        self._raise_after_chunks = raise_after_chunks
        self.array_calls = 0
        self.getitem_calls = 0

    # -- ndarray-shaped metadata the conversion reads -----------------------
    @property
    def shape(self) -> tuple[int, ...]:
        """Delegate the source array's shape."""
        return self._arr.shape

    @property
    def dtype(self) -> np.dtype:
        """Delegate the source array's dtype."""
        return self._arr.dtype

    @property
    def ndim(self) -> int:
        """Delegate the source array's rank."""
        return self._arr.ndim

    @property
    def itemsize(self) -> int:
        """Delegate the source array's item size."""
        return self._arr.itemsize

    @property
    def nbytes(self) -> int:
        """Delegate the source array's byte count."""
        return self._arr.nbytes

    # -- the two injection points ------------------------------------------
    def __array__(self, dtype: DTypeLike = None, copy: Optional[bool] = None) -> np.ndarray:
        """Coercion hook — the access the one-shot fast path takes."""
        self.array_calls += 1
        if self._raise_on_array:
            raise _InjectedPopulateError("injected failure during the one-shot coercion")
        return self._arr if dtype is None else self._arr.astype(dtype, copy=True)

    def __getitem__(self, key: Any) -> np.ndarray:
        """Slice hook — the access the chunked streaming path takes."""
        self.getitem_calls += 1
        if self._raise_after_chunks is not None and self.getitem_calls > self._raise_after_chunks:
            raise _InjectedPopulateError(
                f"injected failure on chunk {self.getitem_calls} — bytes for "
                f"{self._raise_after_chunks} earlier chunk(s) are already in the temporary"
            )
        return cast(np.ndarray, self._arr[key])


class _InjectedPopulateError(RuntimeError):
    """Distinctive exception type so a test can never mistake a real bug for its injection."""


def _make_entry(
    cache: Path,
    array: np.ndarray,
    name: str = "k",
) -> _SourcedNDArray:
    """Build a *non-converting* entry on the configured branch.

    ``enable_caching=False`` routes ``_init_from_config`` to
    ``_convert_to_ndarray``, which returns immediately while ``_mmap`` is
    ``None``. That leaves the first ``_convert_to_memmap`` under the test's
    control, which is what a failure-injection test needs — the conversion has
    to happen *after* the patch is installed, not during construction.
    """
    return _SourcedNDArray(
        array,
        cache_path=cache / name,
        enable_caching=False,
        purge_disk_on_gc=False,
        automatic_offloading=False,
    )


def _temporaries_in(directory: Path) -> list[str]:
    """Return every name in ``directory`` ending in the temporary suffix."""
    return sorted(p.name for p in directory.iterdir() if p.name.endswith(paths.TMP_SUFFIX))


def _mode_of(path: Path) -> int:
    """Return the permission bits of ``path`` (Plan 15-07 / G-3).

    ``S_IMODE`` rather than a raw ``st_mode & 0o777`` so the spelling matches
    the one the source side uses to *capture* the mode, and so a failure prints
    the same quantity the fix reads.
    """
    return stat.S_IMODE(os.stat(path).st_mode)


def _fail_replace_on_temporaries(monkeypatch: pytest.MonkeyPatch, errno: int = 28) -> None:
    """Make ``os.replace`` raise ``ENOSPC`` for renames whose source is a temporary.

    Predicate-gated rather than unconditional so pytest's own machinery — and
    any unrelated rename inside the call — is untouched, which is the shape the
    Plan 02-05 injection template in ``test_lazy_disk_cache.py`` established.
    """
    real_replace = os.replace

    def _boom(src: Any, dst: Any, **kwargs: Any) -> None:
        if str(src).endswith(paths.TMP_SUFFIX):
            raise OSError(errno, "No space left on device")
        real_replace(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", _boom)


def _force_one_shot(monkeypatch: pytest.MonkeyPatch, budget: int = 1 << 30) -> None:
    """Pin the chunk budget high so the one-shot fast path is taken deterministically."""
    monkeypatch.setattr(ldc_module, "psutil", None)
    monkeypatch.setattr(ldc_module, "_MEMMAP_FALLBACK_CHUNK_BYTES", budget)


def _force_chunked(monkeypatch: pytest.MonkeyPatch, budget: int) -> None:
    """Pin the chunk budget low so the chunked streaming path is taken deterministically."""
    monkeypatch.setattr(ldc_module, "psutil", None)
    monkeypatch.setattr(ldc_module, "_MEMMAP_FALLBACK_CHUNK_BYTES", budget)


# ---------------------------------------------------------------------------
# Group A — torn write with no prior file
# ---------------------------------------------------------------------------


@POSIX_ONLY
def test_rename_failure_with_no_prior_dat_leaves_no_final_file_and_no_temporary(
    tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 15-05 / STORE-08 / D-15.

    A conversion interrupted at the rename must not materialise a partial final
    file. Where no ``<key>.dat`` existed, none exists afterwards, and the
    best-effort cleanup takes the temporary with it so the directory is left as
    it was found.
    """
    entry = _make_entry(tmp_cache_dir, np.arange(64, dtype=np.float32))
    _fail_replace_on_temporaries(monkeypatch)

    with pytest.raises(OSError, match="No space left on device"):
        entry._convert_to_memmap()

    assert not (tmp_cache_dir / "k.dat").exists(), (
        "STORE-08 regressed: a failed rename materialised a final <key>.dat. Content must reach "
        "the final name only by a rename that succeeded"
    )
    assert _temporaries_in(tmp_cache_dir) == [], (
        "T-15-20 regressed: the failed conversion left its <key>.dat.tmp behind. The exception "
        "path must unlink the temporary best-effort before re-raising"
    )


@POSIX_ONLY
def test_a_stale_temporary_is_never_adopted_as_data_by_the_next_conversion(
    tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 15-05 / STORE-08 / T-15-21.

    A temporary left by an interrupted run — planted here directly, since the
    cleanup handler removes the real one — must never be read as data. The next
    conversion opens it ``w+``, which truncates it, so the resulting
    ``<key>.dat`` holds the new array and nothing from the interrupted attempt.
    """
    stale = tmp_cache_dir / f"k{_TMP_NAME_SUFFIX}"
    stale.write_bytes(b"\xff" * 4096)

    fresh = np.arange(100, 164, dtype=np.float32)
    entry = _make_entry(tmp_cache_dir, fresh.copy())
    entry._convert_to_memmap()

    assert not stale.exists(), "the stale temporary survived the conversion that should have consumed its name"
    on_disk = np.fromfile(tmp_cache_dir / "k.dat", dtype=np.float32)
    assert np.array_equal(on_disk, fresh), (
        "T-15-21 regressed: the <key>.dat produced after a stale temporary was present does not hold "
        "the new array — residue from the interrupted attempt was adopted as data"
    )


# ---------------------------------------------------------------------------
# Group B — torn write over a valid prior file (the case D-15 exists for)
# ---------------------------------------------------------------------------


@POSIX_ONLY
def test_rename_failure_over_a_valid_dat_leaves_its_bytes_untouched(
    tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 15-05 / STORE-08 / D-15 — the filesystem invariant.

    The pre-change ``r+`` reopen route wrote straight into an existing, valid
    ``<key>.dat``, so a crash part-way through tore a file that was previously
    good. Write-fresh-then-replace cannot: the partial bytes live in the
    temporary, which never gets the final name.
    """
    first = np.arange(64, dtype=np.float32)
    second = np.arange(1000, 1064, dtype=np.float32)

    entry = _make_entry(tmp_cache_dir, first.copy())
    entry._convert_to_memmap()
    dat = tmp_cache_dir / "k.dat"
    good_bytes = dat.read_bytes()
    assert good_bytes, "precondition: the first conversion must have produced a non-empty <key>.dat"

    entry._source = second
    _fail_replace_on_temporaries(monkeypatch)
    with pytest.raises(OSError, match="No space left on device"):
        entry._convert_to_memmap()

    assert dat.read_bytes() == good_bytes, (
        "STORE-08 regressed: an interrupted conversion changed the bytes of a previously-valid "
        "<key>.dat. This is the in-place overwrite D-15 exists to remove"
    )
    assert _temporaries_in(tmp_cache_dir) == [], "the failed conversion left a temporary behind"


@POSIX_ONLY
def test_rename_failure_leaves_the_live_entry_reading_the_old_array(
    tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 15-05 / STORE-08 / D-15a — the *object* invariant (review finding F1).

    Group B asserts on file bytes and would stay green against an
    implementation that assigned the temporary mapping to ``self._mmap`` before
    the rename: the final file's bytes would survive while the entry read
    through a temporary the cleanup handler had just unlinked. This test asserts
    on the entry instead, which is the only instrument that can see it.

    The ``entry._mmap.filename`` check in step 3 is an attribute check standing
    in for an identity check — the pre-call mapping object is not held by the
    test — so it is stated as such rather than dressed up as more than it is.
    """
    first = np.arange(64, dtype=np.float32)
    second = np.arange(1000, 1064, dtype=np.float32)
    third = np.arange(2000, 2064, dtype=np.float32)
    assert not np.array_equal(first, second), (
        "precondition: the two arrays must differ, or step 2's equality assertion passes vacuously"
    )

    entry = _make_entry(tmp_cache_dir, first.copy())
    entry._convert_to_memmap()

    entry._source = second
    _fail_replace_on_temporaries(monkeypatch)

    # 1. the injected failure propagates
    with pytest.raises(OSError, match="No space left on device"):
        entry._convert_to_memmap()

    # 2. the entry still returns the OLD array — asserted in both directions,
    #    because equality alone would pass if the fixtures happened to match
    read_back = np.asarray(entry)
    assert np.array_equal(read_back, first), (
        "D-15a violated: after a failed rename the entry no longer returns the array it was "
        "returning before the call. The file survived; the live object did not (review finding F1)"
    )
    assert not np.array_equal(read_back, second), (
        "D-15a violated: the entry returns the array the FAILED conversion was writing. Observable "
        "state was repointed before the write committed"
    )

    # 3. `self._mmap` is not the temporary mapping
    assert entry._mmap is not None, "the failed conversion dropped the live mapping entirely"
    assert not str(entry._mmap.filename).endswith(paths.TMP_SUFFIX), (
        f"D-15a violated: the live mapping is the temporary ({entry._mmap.filename!r}). The "
        "hand-off must happen only after os.replace and the directory fsync both succeed"
    )
    assert str(entry._mmap.filename).endswith(paths.MEMMAP_SUFFIX), (
        f"the live mapping does not name a memmap artefact: {entry._mmap.filename!r}"
    )

    # 4. the temporary was cleaned up WITHOUT taking the object's buffer with it
    assert _temporaries_in(tmp_cache_dir) == [], (
        "the failed conversion left a temporary behind — cleanup and buffer preservation are both "
        "required, and the ordering (drop the local reference, then unlink) delivers both"
    )

    # 5. the entry is still usable — a healthy object, not one that happens to
    #    answer a single read correctly
    monkeypatch.undo()
    entry._source = third
    entry._convert_to_memmap()
    assert np.array_equal(np.asarray(entry), third), (
        "D-15a violated: the entry survived the failed conversion as an unusable object — the next "
        "successful conversion did not take effect"
    )


# ---------------------------------------------------------------------------
# Group C — crash mid-populate, through the deterministic seam (F3)
# ---------------------------------------------------------------------------


@POSIX_ONLY
def test_one_shot_populate_failure_leaves_file_and_object_intact(
    tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 15-05 / STORE-08 / D-15 — the one-shot fast path (review finding F3, C1).

    What this proves and what it does not: the one-shot assignment
    ``target[:] = array_copy`` has no observable mid-point, so "mid-populate"
    on this path means *before any byte reaches the temporary*. The injection
    fires from ``__array__``, which is the access
    ``np.array(array, dtype=dtype, copy=True)`` actually performs — patching
    ``__getitem__`` here would never fire at all, which is exactly the trap F3
    named.

    The chunk budget is forced high rather than assumed, so the branch under
    test is the branch taken.
    """
    first = np.arange(64, dtype=np.float32)
    entry = _make_entry(tmp_cache_dir, first.copy())
    entry._convert_to_memmap()
    dat = tmp_cache_dir / "k.dat"
    good_bytes = dat.read_bytes()

    _force_one_shot(monkeypatch)
    source = _ExplodingSource(np.arange(1000, 1064, dtype=np.float32), raise_on_array=True)
    entry._source = source

    with pytest.raises(_InjectedPopulateError, match="one-shot coercion"):
        entry._convert_to_memmap()

    assert source.array_calls == 1, (
        f"the one-shot fast path was not the branch taken (__array__ called {source.array_calls} "
        "times) — this test is not exercising what its name claims"
    )
    assert source.getitem_calls == 0, "the chunked loop ran: this is C2's branch, not C1's"
    assert dat.read_bytes() == good_bytes, "STORE-08 regressed: a failed populate tore the previously-valid <key>.dat"
    assert _temporaries_in(tmp_cache_dir) == [], "the failed populate left a temporary behind"
    assert np.array_equal(np.asarray(entry), first), (
        "D-15a violated: a failed populate changed what the live entry returns"
    )


@POSIX_ONLY
def test_chunked_populate_failure_mid_stream_leaves_file_and_object_intact(
    tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 15-05 / STORE-08 / D-15 — the chunked streaming path (review finding F3, C2).

    Genuinely mid-populate: the wrapper delegates the first chunk and raises on
    the second, so bytes are already in the temporary when the failure lands.
    The streaming branch is *forced* — ``psutil`` is nulled and the fallback
    chunk budget pinned below the array's size — rather than approached by
    allocating a real 500 MB array.

    The call-counter assertion is load-bearing rather than decorative: if the
    budget heuristic later changes and the loop degenerates into a single pass,
    this test silently becomes a second copy of C1, and the counter is what
    catches that.
    """
    first = np.zeros((64, 4), dtype=np.float32)
    entry = _make_entry(tmp_cache_dir, first.copy())
    entry._convert_to_memmap()
    dat = tmp_cache_dir / "k.dat"
    good_bytes = dat.read_bytes()

    # row bytes = 4 columns * 4 bytes = 16; budget 64 -> chunk_rows = 4, so a
    # 64-row array needs 16 chunks. nbytes (1024) >= budget (64) selects the
    # streaming branch.
    _force_chunked(monkeypatch, budget=64)
    payload = np.arange(64 * 4, dtype=np.float32).reshape(64, 4) + 1000.0
    source = _ExplodingSource(payload, raise_after_chunks=1)
    entry._source = source

    with pytest.raises(_InjectedPopulateError, match="injected failure on chunk"):
        entry._convert_to_memmap()

    assert source.getitem_calls > 1, (
        f"the chunked loop ran only {source.getitem_calls} time(s): the streaming branch was not "
        "taken and this test has silently become a duplicate of C1"
    )
    assert source.array_calls == 0, "the one-shot coercion ran: this is C1's branch, not C2's"
    assert dat.read_bytes() == good_bytes, (
        "STORE-08 regressed: a failure mid-stream tore the previously-valid <key>.dat. The partial "
        "bytes belong in the temporary, which never gets the final name"
    )
    assert _temporaries_in(tmp_cache_dir) == [], "the interrupted stream left a temporary behind"
    assert np.array_equal(np.asarray(entry), first), (
        "D-15a violated: an interrupted stream changed what the live entry returns"
    )

    # The interrupted attempt leaves nothing that a later conversion can adopt.
    monkeypatch.undo()
    entry._source = None
    entry._data = payload
    entry._shape = payload.shape
    entry._convert_to_memmap()
    assert np.array_equal(np.fromfile(dat, dtype=np.float32).reshape(64, 4), payload), (
        "T-15-21 regressed: the conversion after an interrupted stream did not produce the new array"
    )


@POSIX_ONLY
def test_chunked_populate_failure_with_no_prior_dat_materialises_nothing(
    tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 15-05 / STORE-08 / D-15 — C2's no-prior-file half.

    Same interruption as C2, from the state Group A tests: no ``<key>.dat``
    exists. None must exist afterwards, and no temporary may survive.
    """
    entry = _make_entry(tmp_cache_dir, np.zeros((64, 4), dtype=np.float32))
    _force_chunked(monkeypatch, budget=64)
    payload = np.arange(64 * 4, dtype=np.float32).reshape(64, 4)
    source = _ExplodingSource(payload, raise_after_chunks=1)
    entry._source = source

    with pytest.raises(_InjectedPopulateError):
        entry._convert_to_memmap()

    assert source.getitem_calls > 1, "the streaming branch was not taken"
    assert not (tmp_cache_dir / "k.dat").exists(), (
        "STORE-08 regressed: an interrupted stream materialised a final <key>.dat holding partial data"
    )
    assert _temporaries_in(tmp_cache_dir) == [], "the interrupted stream left a temporary behind"


# ---------------------------------------------------------------------------
# Group D — containment on the temporary (NOT POSIX-marked: the check runs
# ahead of the platform branch and must hold everywhere)
# ---------------------------------------------------------------------------


def test_a_symlink_planted_at_the_temporary_name_is_refused_and_its_target_untouched(
    tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Plan 15-05 / STORE-08 / T-15-18.

    The STORE-08 threat in its named form, one artefact over from the one Phase
    14 closed. ``<key>.dat.tmp`` is a second, differently-named truncating write
    target, and the containment check on the *final* name does not cover it: an
    attacker with write access to the cache directory can occupy the temporary
    name independently.

    Asserts on the sentinel's bytes **and** its length, not on its existence —
    a truncating open leaves the file present and empty, which an existence
    check would call a pass.

    Not POSIX-marked: this check runs before the platform branch, so refusing
    the plant is a property of every platform.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "victim.bin"
    sentinel_bytes = b"do-not-overwrite-me" * 8
    sentinel.write_bytes(sentinel_bytes)
    (tmp_cache_dir / f"k{_TMP_NAME_SUFFIX}").symlink_to(sentinel)

    entry = _make_entry(tmp_cache_dir, np.arange(64, dtype=np.float32))
    with pytest.raises(StoreContainmentError):
        entry._convert_to_memmap()

    assert sentinel.read_bytes() == sentinel_bytes, (
        "T-15-18 regressed: the sentinel outside the cache directory was modified through a symlink "
        "planted at the <key>.dat.tmp name. The temporary needs its own containment check"
    )
    assert sentinel.stat().st_size == len(sentinel_bytes), "the sentinel was truncated despite the refusal"


def test_a_symlink_at_the_final_name_resolving_outside_is_refused_and_its_target_untouched(
    tmp_cache_dir: Path, tmp_path: Path
) -> None:
    """Plan 15-05 / STORE-02 / D-31 — the outward escape at the **final** name.

    The negative half of the pair this module was missing. The two neighbouring
    tests each cover a different quadrant and neither covers this one: the
    ``T-15-18`` test above plants its link at the **temporary** name, and the
    adopted-entry test below points its link at a payload **inside** the cache
    directory. This is a link at the *final* name resolving *outward* — the case
    that must be refused.

    ``test_lazy_disk_cache.py`` carries a Phase-14 sibling of this test, and the
    duplication is deliberate rather than an oversight. That one was written
    against the pre-15-05 route and pins the ``w+``/``r+`` opens that no longer
    exist. What is asserted here and cannot be asserted there is the third
    assertion below: the refusal happens **before the temporary is created**, so
    the guard sits ahead of the platform branch rather than inside the atomic
    route. A future edit that moved the containment check into the POSIX branch
    would still pass the Phase-14 test and fail this one.

    Not POSIX-marked: refusing an outward escape is a property of every
    platform, and marking it would hide a real regression off POSIX.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "victim.bin"
    sentinel_bytes = b"do-not-overwrite-me" * 8
    sentinel.write_bytes(sentinel_bytes)
    (tmp_cache_dir / "k.dat").symlink_to(sentinel)

    entry = _make_entry(tmp_cache_dir, np.arange(64, dtype=np.float32))
    with pytest.raises(StoreContainmentError):
        entry._convert_to_memmap()

    assert sentinel.read_bytes() == sentinel_bytes, (
        "STORE-02 regressed: the sentinel outside the cache directory was modified through a "
        "symlink planted at the <key>.dat name. A truncating write escaped the cache directory"
    )
    assert sentinel.stat().st_size == len(sentinel_bytes), "the sentinel was truncated despite the refusal"
    assert _temporaries_in(tmp_cache_dir) == [], (
        "the refusal came too late: a <key>.dat.tmp was created before the final name was "
        "containment-checked. The guard belongs ahead of the platform branch, not inside it"
    )


def test_an_adopted_entry_symlink_still_receives_the_data_through_the_atomic_route(
    tmp_cache_dir: Path,
) -> None:
    """Plan 15-05 / STORE-03 / D-17 — the positive symlink case, under the new route.

    ``os.replace`` never follows a symlink; it replaces the link. So the naive
    atomic rewrite — rename onto ``self._cache_path`` — silently converts a
    legitimately *adopted* entry into a regular file and orphans the payload the
    caller adopted, with no error anywhere. The route renames onto the
    **resolved** path for exactly this reason.

    Not POSIX-marked: the property holds on both routes, by different
    mechanisms (rename-onto-resolved here, open-through-the-link off POSIX).
    """
    payload = tmp_cache_dir / "adopted_payload.bin"
    payload.write_bytes(b"\x00" * 256)
    (tmp_cache_dir / "k.dat").symlink_to(payload)

    values = np.arange(64, dtype=np.float32)
    entry = _make_entry(tmp_cache_dir, values.copy())
    entry._convert_to_memmap()

    assert (tmp_cache_dir / "k.dat").is_symlink(), (
        "STORE-03 regressed: the atomic route replaced the adopted entry's symlink with a regular "
        "file. The caller's adopted payload silently stops receiving data"
    )
    assert np.array_equal(np.fromfile(payload, dtype=np.float32), values), (
        "STORE-03 regressed: the write did not reach the adopted target the link resolves to"
    )
    assert _temporaries_in(tmp_cache_dir) == [], "the adopted-entry conversion left a temporary behind"


# ---------------------------------------------------------------------------
# Group E — round trip, success shape and the empty case (NOT POSIX-marked:
# these must hold on every platform)
# ---------------------------------------------------------------------------


def test_a_successful_conversion_leaves_a_dat_and_no_temporary(tmp_cache_dir: Path) -> None:
    """Plan 15-05 / STORE-08 / D-15.

    The success-path shape: the artefact set after a fresh write is exactly the
    final name. A temporary that survives a *successful* conversion is a
    D-14/D-09 artefact-count defect as well as a STORE-08 one.
    """
    entry = _make_entry(tmp_cache_dir, np.arange(64, dtype=np.float32))
    entry._convert_to_memmap()

    assert (tmp_cache_dir / "k.dat").exists(), "the conversion produced no <key>.dat"
    assert _temporaries_in(tmp_cache_dir) == [], (
        "a successful conversion left a <key>.dat.tmp behind — the rename must consume the temporary"
    )


def test_float32_round_trips_byte_identically_through_the_atomic_route(tmp_cache_dir: Path) -> None:
    """Plan 15-05 / STORE-05 — precision, discharged here.

    The temporary-file detour is the only thing in this phase that could
    introduce a coercion, a truncation or a precision change on the ``.dat``
    payload, so the precision question is answered against this route.
    """
    values = (np.arange(64, dtype=np.float32) / 3.0).astype(np.float32)
    entry = _make_entry(tmp_cache_dir, values.copy())
    entry._convert_to_memmap()

    read_back = np.asarray(entry)
    assert np.array_equal(read_back, values), "STORE-05 regressed: float32 did not round-trip byte-identically"
    assert read_back.dtype == values.dtype, f"dtype changed across the route: {read_back.dtype} != {values.dtype}"
    assert read_back.shape == values.shape, f"shape changed across the route: {read_back.shape} != {values.shape}"
    assert (tmp_cache_dir / "k.dat").read_bytes() == values.tobytes(), (
        "STORE-05 regressed: the bytes on disk are not the source array's bytes"
    )


def test_float64_round_trips_byte_identically_through_the_atomic_route(tmp_cache_dir: Path) -> None:
    """Plan 15-05 / STORE-05 — the wider dtype, so the check is not float32-shaped.

    A route that quietly narrowed to float32 would pass the float32 test and
    fail this one, which is why both dtypes are asserted rather than one.
    """
    values = np.arange(64, dtype=np.float64) / 3.0
    entry = _make_entry(tmp_cache_dir, values.copy())
    entry._convert_to_memmap()

    read_back = np.asarray(entry)
    assert np.array_equal(read_back, values), "STORE-05 regressed: float64 did not round-trip byte-identically"
    assert read_back.dtype == np.dtype(np.float64), f"float64 was narrowed to {read_back.dtype} across the route"
    assert (tmp_cache_dir / "k.dat").read_bytes() == values.tobytes(), (
        "STORE-05 regressed: the bytes on disk are not the source array's bytes"
    )


def test_zero_length_array_resolves_exactly_as_it_did_before_the_change(tmp_cache_dir: Path) -> None:
    """Plan 15-05 / STORE-08 — the empty case, measured rather than assumed.

    The pre-change behaviour was **measured** on the baseline blob at
    ``$BASE = 0c26dd6`` (this plan's pre-execution branch tip) before Task 1
    edited anything, not guessed from reading::

        OUTCOME: raised ValueError
          message: cannot mmap an empty file
          files: ['empty.dat']
          any .tmp survivor: []

    So the assertion is that it *still* raises ``ValueError`` with that message,
    and that no temporary survives either way. One thing did change and is
    recorded rather than asserted away: pre-change the failed ``w+`` open left a
    zero-byte ``<key>.dat`` behind, and post-change the failure happens on the
    temporary, which the cleanup handler unlinks — so nothing survives at all.
    The *resolution* is identical; the residue is strictly cleaner, and no route
    now creates a final file it cannot fill.
    """
    entry = _make_entry(tmp_cache_dir, np.zeros((0,), dtype=np.float32), name="empty")

    with pytest.raises(ValueError, match="cannot mmap an empty file"):
        entry._convert_to_memmap()

    assert _temporaries_in(tmp_cache_dir) == [], (
        "the zero-length conversion left a temporary behind — the empty case must clean up like every other failure"
    )


# ---------------------------------------------------------------------------
# Group F — write order
# ---------------------------------------------------------------------------


@POSIX_ONLY
def test_write_order_is_flush_then_fsync_then_replace_then_dir_fsync(
    tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 15-05 / STORE-08 / D-15 — the sequence, asserted rather than assumed.

    The order is the contract ``DiskBackedStore._store_entry`` already
    establishes for the codec pair, ported to this artefact rather than
    reinvented beside it: populate, flush, fsync the data, rename, fsync the
    directory. Each step is pointless without its predecessor — renaming before
    the fsync makes the final name durable while its contents are not.

    The two ``os.fsync`` calls are distinguished by ``fstat`` rather than by
    position, so the assertion is about *which* descriptor was synced and not
    merely about how many were. Asserted on relative position, not on an exact
    list, so an added step does not fail a test about ordering.

    Note this group patches ``numpy.memmap.flush`` to **delegate** and record.
    That is the opposite of Group C's rejected instrument, which was *raising*
    from ``flush`` to simulate a mid-populate crash — an injection that fires
    after the populate block has already finished.
    """
    events: list[str] = []
    real_flush = np.memmap.flush
    real_fsync = os.fsync
    real_replace = os.replace

    def _flush(self: np.memmap) -> None:
        events.append("flush")
        real_flush(self)

    def _fsync(fd: int) -> None:
        kind = "dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
        events.append(f"fsync:{kind}")
        real_fsync(fd)

    def _replace(src: Any, dst: Any, **kwargs: Any) -> None:
        events.append("replace")
        real_replace(src, dst, **kwargs)

    monkeypatch.setattr(np.memmap, "flush", _flush)
    monkeypatch.setattr(os, "fsync", _fsync)
    monkeypatch.setattr(os, "replace", _replace)

    entry = _make_entry(tmp_cache_dir, np.arange(64, dtype=np.float32))
    entry._convert_to_memmap()
    monkeypatch.undo()

    for expected in ("flush", "fsync:file", "replace", "fsync:dir"):
        assert expected in events, f"D-15 regressed: {expected!r} never happened. Recorded sequence: {events}"

    i_flush = events.index("flush")
    i_data_fsync = events.index("fsync:file")
    i_replace = events.index("replace")
    i_dir_fsync = events.index("fsync:dir")

    assert i_flush < i_data_fsync, (
        f"D-15 regressed: the data fsync preceded the flush, so it synced an incomplete file. "
        f"Recorded sequence: {events}"
    )
    assert i_data_fsync < i_replace, (
        f"D-15 regressed: the rename preceded the data fsync, so the final name could become "
        f"durable before its contents. Recorded sequence: {events}"
    )
    assert i_replace < i_dir_fsync, (
        f"D-15 regressed: the directory fsync preceded the rename it exists to make durable. "
        f"Recorded sequence: {events}"
    )


@POSIX_ONLY
def test_the_final_dat_name_is_never_opened_for_writing(tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan 15-05 / STORE-08 / SC-6 / D-15 — content arrives only by rename.

    SC-6's headline clause is *"the final name is never opened for writing on the
    POSIX route — content arrives only by rename"*, and until now it was measured
    only by the verifier's throwaway ``sc6_repro.py``. Nothing in the suite
    asserted it, so the claim rested on a script that no longer exists.

    **This is not the write-order test wearing a second name.**
    ``test_write_order_is_flush_then_fsync_then_replace_then_dir_fsync`` pins the
    *sequence* of syscalls; it says nothing about the **target** of the write. An
    implementation that populated ``<key>.dat`` directly and then performed a
    perfectly ordered flush/fsync/replace/fsync dance over a decoy temporary
    would satisfy every assertion in that test while having already torn the file
    the whole route exists to protect. The two tests fail on disjoint mutations,
    which is the reason both are kept.

    The instrument records the **filename and mode of every** ``np.memmap``
    **construction**, then asserts a negative over the final name. Recording and
    delegating rather than raising: a spy that intercepted the call would change
    the behaviour it is measuring, and the conversion must actually succeed for
    the absence of a write-open to mean anything — hence the round-trip assertion
    at the end, which stops this from going green against a route that wrote
    nothing at all.

    ``mode`` is read from ``kwargs`` first and then positionally, because
    ``numpy.memmap(filename, dtype, mode, ...)`` accepts it either way and a
    kwargs-only reading would silently record ``None`` for a positional caller
    and pass vacuously.
    """
    opens: list[tuple[Path, str]] = []
    real_memmap = np.memmap

    def _spy(filename: Any, *args: Any, **kwargs: Any) -> Any:
        mode = kwargs.get("mode")
        if mode is None and len(args) >= 2:
            mode = args[1]
        opens.append((Path(str(filename)), str(mode)))
        return real_memmap(filename, *args, **kwargs)

    monkeypatch.setattr(ldc_module.np, "memmap", _spy)

    payload = np.arange(64, dtype=np.float32)
    entry = _make_entry(tmp_cache_dir, payload.copy())
    entry._convert_to_memmap()
    monkeypatch.undo()

    final = (tmp_cache_dir / "k.dat").resolve()
    writing = [(p, m) for p, m in opens if m.startswith(("w", "r+", "a"))]

    assert writing, (
        f"the instrument recorded no writing open at all, so the negative below would pass "
        f"vacuously against a route that had stopped writing. Recorded: {opens}"
    )
    assert all(p.name.endswith(paths.TMP_SUFFIX) for p, _ in writing), (
        f"STORE-08 / SC-6 regressed: a writing open landed on something other than a "
        f"{paths.TMP_SUFFIX} temporary. Recorded writing opens: {writing}"
    )

    to_final = [(p, m) for p, m in writing if p.resolve() == final]
    assert to_final == [], (
        f"STORE-08 / SC-6 regressed: the FINAL <key>.dat name was opened for writing "
        f"({to_final}). Content must arrive at the final name only by os.replace — a route "
        f"that writes it directly can tear a previously-valid file no matter how well the "
        f"surrounding flush/fsync/rename sequence is ordered"
    )
    assert np.array_equal(np.fromfile(tmp_cache_dir / "k.dat", dtype=np.float32), payload), (
        "the conversion did not actually write the array, which would make every assertion above vacuous"
    )


# ---------------------------------------------------------------------------
# Group G — the destination's permissions survive the atomic rename
# (POSIX-marked: both `os.replace`'s inode semantics and the mode bits
# themselves are POSIX-scoped, exactly like Groups A/B/C/F)
# ---------------------------------------------------------------------------


@POSIX_ONLY
def test_the_default_unconfigured_branch_keeps_mkstemps_0600_across_the_atomic_rename() -> None:
    """Plan 15-07 / G-3 / CR-02 / CWE-732 — the phase-introduced permission regression.

    ``os.replace`` moves the temporary's **inode** onto the final name, carrying
    that inode's mode and discarding the destination's. The victim is the
    default, unconfigured branch: ``_init_from_config`` creates the ``.dat``
    with ``tempfile.mkstemp``, whose entire purpose is ``0o600``, and the first
    conversion threw that away for ``0o666 & ~umask``.

    Measured, not read (same script against both trees)::

        9c80f64 (phase base)  mkstemp created: 0o600   after convert: 0o600
        0956838 (phase HEAD)  mkstemp created: 0o600   after convert: umask default

    The HEAD reading is named by its *mechanism* rather than by the octal digits
    it happened to produce here, because the digits are host state: this host
    runs ``umask 022`` so ``0o666 & ~umask`` read as world-readable, and a
    developer on ``umask 077`` would reproduce the same defect with different
    digits. The literal transcript is in ``15-VERIFICATION.md`` and
    ``15-REVIEW.md`` § CR-02.

    On a shared ``/scratch`` or a multi-user ``/tmp`` that is the whole payload
    of every cache entry built without an explicit ``cache_path``, readable by
    any local user, with nothing logged and nothing raised.

    ``enable_caching=True`` so the *construction* itself takes the conversion —
    with the ``enable_caching=False`` default no write happens at build time and
    the first reading would be ``mkstemp``'s own mode, which asserts nothing
    about the route under test.
    """
    entry = DiskBackedNDArray(
        np.arange(64, dtype=np.float32),
        enable_caching=True,
        purge_disk_on_gc=False,
        automatic_offloading=False,
    )
    dat = Path(entry.cache_path)
    try:
        after_construction = _mode_of(dat)
        entry._convert_to_memmap()
        after_second = _mode_of(dat)
    finally:
        dat.unlink(missing_ok=True)

    assert after_construction == 0o600, (
        f"STORE-08 / CWE-732: the unconfigured branch's <key>.dat is {oct(after_construction)} after "
        f"construction, not tempfile.mkstemp's 0o600. Measured 0o600 at phase base 9c80f64 and the "
        f"umask default at 0956838 — os.replace carried the temporary's mode onto the final name"
    )
    assert after_second == 0o600, (
        f"STORE-08 / CWE-732: a second conversion widened the unconfigured branch's <key>.dat to "
        f"{oct(after_second)}. The destination's mode must be carried across the rename on every "
        f"conversion, not merely on the first. Measured 0o600 at 9c80f64, umask default at 0956838"
    )


@POSIX_ONLY
def test_a_tightened_configured_dat_keeps_its_mode_across_a_reconversion(tmp_cache_dir: Path) -> None:
    """Plan 15-07 / G-3 / STORE-08 — the destination-exists side of the boundary.

    The operator's side of the same defect. An administrator who tightens a
    cache artefact by hand has made a decision the library must not quietly
    revert; before the fix the next conversion reset it to the umask default,
    with nothing logged.

    The array is asserted *as well as* the mode, on purpose: a mode assertion
    that passed because the second write never landed would be worthless, and
    would go green against an implementation that had stopped writing at all.
    """
    first = np.arange(64, dtype=np.float32)
    second = np.arange(1000, 1064, dtype=np.float32)

    entry = _make_entry(tmp_cache_dir, first.copy())
    entry._convert_to_memmap()
    dat = tmp_cache_dir / "k.dat"
    os.chmod(dat, 0o640)
    assert _mode_of(dat) == 0o640, "precondition: the tightening chmod did not take"

    entry._source = second
    entry._convert_to_memmap()

    assert _mode_of(dat) == 0o640, (
        f"STORE-08 / CWE-732: a reconversion widened an operator-tightened <key>.dat from 0o640 to "
        f"{oct(_mode_of(dat))}. os.replace carries the temporary's inode and its mode; the "
        f"destination's mode has to be applied to the temporary before the rename"
    )
    assert np.array_equal(np.fromfile(dat, dtype=np.float32), second), (
        "the reconversion did not write the second array — the mode assertion above would be "
        "vacuous against a write that silently did nothing"
    )


@POSIX_ONLY
def test_a_configured_dat_created_fresh_gets_the_umask_default_unchanged(tmp_cache_dir: Path) -> None:
    """Plan 15-07 / G-3 / T-15-27 — the destination-absent side, asserting *no change*.

    One step over the boundary, and its whole value is that it asserts nothing
    moved. When no destination exists there is nothing whose mode could be
    preserved, so the mode-carrying step must stand aside and let the operator's
    umask decide the first write, exactly as the pre-change code did.

    This is the guard against someone later "finishing" the fix with
    ``15-REVIEW.md`` § CR-02's ``desired = 0o600`` default. That literal is a
    *tightening*, not a preservation: it would silently change the observable
    permissions of every configured-cache-dir first write, which is a policy
    change requiring its own justification rather than part of closing a
    regression.

    The expected value is computed from the live umask with the read-then-restore
    idiom, never hard-coded. The digits are host state — this host runs
    ``umask 022`` — and a developer running ``umask 077`` must not inherit a red
    suite from a number baked in on someone else's machine.
    """
    old_umask = os.umask(0)
    os.umask(old_umask)
    expected = 0o666 & ~old_umask

    dat = tmp_cache_dir / "fresh.dat"
    assert not dat.exists(), "precondition: this branch requires the destination to be absent"

    entry = _make_entry(tmp_cache_dir, np.arange(64, dtype=np.float32), name="fresh")
    entry._convert_to_memmap()

    assert _mode_of(dat) == expected, (
        f"T-15-27: the first write of a configured <key>.dat produced {oct(_mode_of(dat))} where the "
        f"pre-change code produced {oct(expected)} (0o666 & ~umask, umask={oct(old_umask)}). The "
        f"absent-destination branch must perform no chmod — imposing a library default here tightens "
        f"the operator's artefact behind their back"
    )


@POSIX_ONLY
@pytest.mark.parametrize("configured", [False, True], ids=["mkstemp", "configured"])
def test_converting_twice_leaves_the_mode_unchanged_on_both_branches(tmp_cache_dir: Path, configured: bool) -> None:
    """Plan 15-07 / G-3 / STORE-08 — idempotency, so the step cannot ratchet.

    A mode-carrying step that read the *temporary's* mode instead of the
    destination's would still look correct once and then drift on every
    subsequent conversion, in whichever direction the umask pushed it. Reading
    the same file after conversion *n* and *n+1* is the instrument that sees
    that; a single-conversion test cannot.

    The ``mkstemp`` parameter additionally asserts the absolute value ``0o600``
    so the parameterisation cannot degenerate into two copies of the weaker
    "unchanged" check — an implementation that had abandoned the fix entirely
    would satisfy equality on both branches.
    """
    entry: DiskBackedNDArray
    if configured:
        entry = _make_entry(tmp_cache_dir, np.arange(64, dtype=np.float32), name="idem")
        entry._convert_to_memmap()
        dat = tmp_cache_dir / "idem.dat"
    else:
        entry = DiskBackedNDArray(
            np.arange(64, dtype=np.float32),
            enable_caching=True,
            purge_disk_on_gc=False,
            automatic_offloading=False,
        )
        dat = Path(entry.cache_path)

    try:
        mode_n = _mode_of(dat)
        entry._convert_to_memmap()
        mode_n1 = _mode_of(dat)
        entry._convert_to_memmap()
        mode_n2 = _mode_of(dat)
    finally:
        if not configured:
            dat.unlink(missing_ok=True)

    assert mode_n == mode_n1 == mode_n2, (
        f"STORE-08: the mode-carrying step is not idempotent on the {'configured' if configured else 'mkstemp'} "
        f"branch — successive conversions read {oct(mode_n)}, {oct(mode_n1)}, {oct(mode_n2)}. The step must "
        f"carry the DESTINATION's mode; reading the temporary's would ratchet toward the umask default"
    )
    if not configured:
        assert mode_n == 0o600, (
            f"STORE-08 / CWE-732: the mkstemp branch settled at {oct(mode_n)} rather than tempfile.mkstemp's "
            f"0o600. Equality above is satisfied by any stable wrong value, which is why this branch also "
            f"asserts the absolute one"
        )


@POSIX_ONLY
def test_a_failed_chmod_leaves_the_previous_dat_and_its_mode_intact(
    tmp_cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 15-07 / G-3 / D-15a — the new step inherits the two-phase-commit property.

    The mode-carrying step is placed inside the commit ``try`` deliberately, and
    this test is what makes that placement a property rather than a comment. A
    ``chmod`` that failed outside the handler would leave the temporary on disk
    and the entry in a half-committed permission state; inside it, the handler
    drops the local mapping, unlinks the temporary and re-raises with the
    previously-valid ``<key>.dat`` unchanged in **both** bytes and mode, and the
    live entry still reading the old array.

    All four are asserted because they are four different claims — Group B
    established the byte and object halves for a failed *rename*, and a new
    failure point ahead of the rename must not weaken either.
    """
    first = np.arange(64, dtype=np.float32)
    second = np.arange(1000, 1064, dtype=np.float32)

    entry = _make_entry(tmp_cache_dir, first.copy())
    entry._convert_to_memmap()
    dat = tmp_cache_dir / "k.dat"
    good_bytes = dat.read_bytes()
    good_mode = _mode_of(dat)
    assert good_bytes, "precondition: the first conversion must have produced a non-empty <key>.dat"

    real_chmod = os.chmod

    def _boom(path: Any, mode: int, **kwargs: Any) -> None:
        # Predicate-gated on the temporary suffix, matching
        # `_fail_replace_on_temporaries`: pytest's own machinery and any
        # unrelated chmod inside the call stay real.
        if str(path).endswith(paths.TMP_SUFFIX):
            raise PermissionError(1, "Operation not permitted")
        real_chmod(path, mode, **kwargs)

    monkeypatch.setattr(os, "chmod", _boom)

    entry._source = second
    with pytest.raises(PermissionError):
        entry._convert_to_memmap()
    monkeypatch.undo()

    assert dat.read_bytes() == good_bytes, (
        "D-15a regressed: a failed chmod changed the bytes of a previously-valid <key>.dat. The "
        "mode-carrying step must sit inside the commit try, ahead of the rename"
    )
    assert _mode_of(dat) == good_mode, (
        f"D-15a regressed: a failed chmod left the previously-valid <key>.dat at {oct(_mode_of(dat))} "
        f"instead of {oct(good_mode)} — a half-committed permission state"
    )
    assert _temporaries_in(tmp_cache_dir) == [], (
        "the failed chmod left a temporary behind — the existing except-handler must have unlinked it"
    )
    assert np.array_equal(np.asarray(entry), first), (
        "D-15a regressed: after a failed chmod the live entry no longer reads the array it read before "
        "the call. Nothing observable may be repointed before the commit"
    )
