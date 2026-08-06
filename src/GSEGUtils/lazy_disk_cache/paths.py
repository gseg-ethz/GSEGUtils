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

"""Store-key rules and on-disk path construction for the disk-backed cache.

Owns the lexical rule that decides whether a string may become a filename inside
a cache directory. Exposes :func:`is_valid_store_key` as the public predicate
(D-07), the internal raising validator :func:`validate_store_key`, and the two
published exception types :exc:`StoreKeyError` and :exc:`StoreContainmentError`
(D-12).

It also owns the artefact suffix vocabulary and **every** path the store builds
(D-14): :func:`get_npy_path`, :func:`get_meta_path`,
:func:`get_legacy_pickle_path`, :func:`get_npy_tmp_path` and
:func:`get_meta_tmp_path`, each of which validates its key lexically and then
verifies that the path it is about to return resolves *inside* the cache
directory. ``STORE_PATH_BUILDERS`` enumerates them so "every builder is guarded"
is a property that can be iterated rather than a claim that must be re-asserted
each time a builder is added.

This module deliberately imports **nothing** from the rest of the package
(D-14). ``disk_backed_store.py`` already imports from ``lazy_disk_cache.py``, so
a validator defined in either of them would be unreachable from the other; a
third, dependency-free module makes the cycle structurally impossible rather
than merely avoided.
"""

__all__ = [
    "StoreKeyError",
    "StoreContainmentError",
    "validate_store_key",
    "is_valid_store_key",
    "get_npy_path",
    "get_meta_path",
    "get_legacy_pickle_path",
    "get_npy_tmp_path",
    "get_meta_tmp_path",
]

from collections.abc import Callable
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final, NoReturn, Optional

# ---------------------------------------------------------------------------
# Phase-14 exception hierarchy (D-12)
# ---------------------------------------------------------------------------


class StoreKeyError(ValueError):
    """Raised when a store key violates the lexical single-segment key rule."""


class StoreContainmentError(StoreKeyError):
    """Raised when a resolved store path would land outside its cache directory."""


# ---------------------------------------------------------------------------
# Phase-14 lexical key rule (D-04 / D-05 / D-06 / D-13)
# ---------------------------------------------------------------------------

#: The clause vocabulary. Tests ``match=`` on these strings — keep them stable.
CLAUSE_RESERVED: Final[str] = "is empty or a reserved path name"
CLAUSE_CONTROL: Final[str] = "contains a control character"
CLAUSE_ABSOLUTE: Final[str] = "is an absolute or drive-relative path"
CLAUSE_SEPARATOR: Final[str] = "contains a path separator"

_RESERVED_KEYS: Final[frozenset[str]] = frozenset({"", ".", ".."})
_SEPARATORS: Final[tuple[str, str]] = ("/", "\\")
_DEL_CHAR: Final[str] = "\x7f"


def _refuse(key: str, cache_dir: Optional[Path], clause: str) -> NoReturn:
    """Raise :exc:`StoreKeyError` naming the key, the cache directory and the clause.

    Parameters
    ----------
    key : str
        The refused key, rendered with ``repr``. The ``repr`` is load-bearing
        rather than cosmetic (D-13): an embedded newline is one of the refused
        cases, and interpolating such a key raw would split the log record —
        which would make this containment fix's own refusal message a
        log-injection vector. ``repr`` also renders NUL and other control
        characters visible, which is exactly when the reader needs to see them.
    cache_dir : Path, optional
        The cache directory the key was destined for. ``None`` renders as an
        explicit unset marker rather than silently disappearing from the message.
    clause : str
        Which clause of the rule was violated; one of the ``CLAUSE_*`` constants.

    Raises
    ------
    StoreKeyError
        Always.
    """
    location = repr(str(cache_dir)) if cache_dir is not None else "<unset>"
    raise StoreKeyError(
        f"Invalid store key {key!r} for cache directory {location}: the key {clause}; "
        "a store key must be a single path segment — no '/' or '\\', not an absolute or "
        "drive-relative path, not '', '.' or '..', and free of control characters."
    )


def validate_store_key(key: str, cache_dir: Optional[Path] = None) -> None:
    r"""Validate that ``key`` may be used as a single-segment store key.

    Implements D-04 (a key is one path segment) and D-05 (a *property denylist*,
    never an allowlist charset). The charset stays open on purpose: pc2img
    treats feature names as simultaneously public API and cache key, and real
    keys look like ``rrim_pack_(range,r16,d8,z1e-05)``, so an allowlist drawn
    around today's grammar would make the next legitimate feature name a
    cross-repo break.

    Clauses are evaluated in a **fixed order**, so a key violating several
    always reports the same one:

    1. ``CLAUSE_RESERVED`` — the key is ``''``, ``'.'`` or ``'..'``.
    2. ``CLAUSE_CONTROL`` — the key contains a character below ``\x20``, or
       ``\x7f``. This covers newline and NUL.
    3. ``CLAUSE_ABSOLUTE`` — the key has a non-empty ``anchor`` or ``drive``
       under *either* :class:`~pathlib.PurePosixPath` or
       :class:`~pathlib.PureWindowsPath`. This refuses ``/etc/passwd``,
       ``C:evil``, ``C:/abs`` and ``\\server\share\x``.
    4. ``CLAUSE_SEPARATOR`` — one clause with two tests, either of which refuses:

       * a **raw-character scan** for ``/`` or ``\`` at any position. This is
         what refuses a *trailing* separator, and it cannot be replaced by a
         ``parts`` test: ``pathlib`` normalises trailing separators away, so
         ``PurePosixPath('a/').parts`` is ``('a',)`` — length 1 — and a length
         test alone therefore **accepts** ``'a/'``. D-05 names a trailing
         separator among the required refusals.
       * a **dual-interpretation structural test**: ``parts`` of length other
         than 1 under either flavour. It refuses anything ``pathlib`` reads as
         multi-segment even where no ASCII separator appears, and the dual
         interpretation is what makes ``..\..\x`` fail on Linux structurally
         as well as by character — ``PurePosixPath`` reads it as one harmless
         segment while ``PureWindowsPath`` reads it as three.

    **Dots are legal except the exact ``.`` and ``..``, including leading dots**
    (D-06): ``.hidden``, ``foo.bar``, ``z1.2345678``, ``foo.`` and ``a.npy`` all
    pass. The reason matters, because the obvious reading is wrong: **the
    separator is the entire escape mechanism, and dots carry no escape risk.**
    The path builders never join the bare key — they concatenate an extension
    first — so ``Path('/cache') / '...npy'`` (key ``'..'``) is a literal file
    *inside* the cache directory and does not escape, while
    ``Path('/cache') / '../victim.npy'`` does, and the separator is what did it.
    **Refusing ``.`` and ``..`` is therefore defence-in-depth against a future
    bare join, not the closing of a live hole** — no path builder does a bare
    join today.

    The empty key has a separate and concrete reason: key ``''`` builds
    ``<cache>/.npy``, whose rescan stem is ``'.npy'`` — a *different,
    legal-looking* key. ``''`` is refused because it does not round-trip, not
    because it is untidy.

    No normalisation of any kind is applied before validation (D-05). Validating
    one string and building the path from another is the classic bypass: a
    fullwidth ``'．．/victim'`` NFKC-folds to ``'../victim'``. The exact
    characters the caller supplied are the characters validated *and* the
    characters used to build the path.

    There is deliberately **no** ``@validate_call`` / ``validate_variables``
    decorator here: that decorator checks *types*, and a malicious key is a
    perfectly well-typed :class:`str`, so adding it would read as the rule while
    enforcing nothing.

    That reasoning stands unchanged. The explicit non-``str`` guard below is not
    a reversal of it — it is its **replacement for the type axis specifically**.
    Leaving the decorator off answered the value axis correctly and left the
    type axis with no answer at all, which is a defect on a *published*
    predicate: :func:`is_valid_store_key` documents a :class:`bool` return, and
    a non-``str`` used to escape it as a bare :exc:`TypeError` from the
    control-character scan. One explicit guard answers the type axis at the one
    place the predicate can honour its documented return type, without importing
    a decorator that would overstate what is being checked.

    The guard is the **first** statement of this function, ahead of the
    reserved-name membership test, and the position is load-bearing rather than
    stylistic: ``key in _RESERVED_KEYS`` raises ``TypeError: unhashable type``
    for an unhashable argument, so a guard placed after it would leave one input
    class still escaping as exactly the bare :exc:`TypeError` this fixes.

    Parameters
    ----------
    key : str
        The candidate store key, exactly as the caller supplied it.
    cache_dir : Path, optional
        The cache directory the key is destined for. Used only to make the
        refusal message actionable (D-13); it is never read from disk, so this
        function performs no I/O and is pure.

    Returns
    -------
    None
        Returns ``None`` when ``key`` is legal.

    Raises
    ------
    StoreKeyError
        If ``key`` is not a :class:`str` at all, or if it violates any clause of
        the rule. ``StoreKeyError`` subclasses :class:`ValueError` (D-12), so
        every existing ``except ValueError`` still catches it, and it is
        deliberately *not* a :class:`KeyError` —
        :meth:`DiskBackedStore.add_data_to_store` already raises that one for
        "key exists". The non-``str`` refusal names the received type and
        carries a ``repr`` of the value, and deliberately reports **no**
        ``CLAUSE_*``: the clause vocabulary describes how a key is *shaped*, and
        a value that is not a ``str`` is not a key at all.
    """
    # Widened to ``object`` before the test so the type checker does not discard
    # the branch as statically unreachable. It *is* statically unreachable and
    # entirely reachable at runtime — a caller who ignores the annotation is the
    # case this guard exists for, and a path-typed identifier fed straight into
    # the published pre-check is the real reported shape.
    supplied: object = key
    if not isinstance(supplied, str):
        raise StoreKeyError(
            f"Invalid store key {supplied!r}: a store key must be a str, but got "
            f"{type(supplied).__name__}. Convert it explicitly — a path-typed identifier is not a "
            "key, and this function will not guess which of its components was meant."
        )

    if key in _RESERVED_KEYS:
        _refuse(key, cache_dir, CLAUSE_RESERVED)

    if any(ord(char) < 32 or char == _DEL_CHAR for char in key):
        _refuse(key, cache_dir, CLAUSE_CONTROL)

    posix = PurePosixPath(key)
    windows = PureWindowsPath(key)
    if posix.anchor or posix.drive or windows.anchor or windows.drive:
        _refuse(key, cache_dir, CLAUSE_ABSOLUTE)

    if any(sep in key for sep in _SEPARATORS) or len(posix.parts) != 1 or len(windows.parts) != 1:
        _refuse(key, cache_dir, CLAUSE_SEPARATOR)


def is_valid_store_key(key: str) -> bool:
    """Return whether ``key`` is a legal single-segment store key.

    The public form of the rule (D-07). It is implemented as a ``try``/``except``
    around :func:`validate_store_key` so the predicate and the validator can
    never disagree — the migration note's cache-directory scan snippet *imports*
    this predicate rather than restating the rule as a regex, which is what makes
    drift between the snippet and the shipped rule impossible.

    Downstreams can also use it to pre-check a composed feature name *before* it
    becomes a key, or to decide per-tile without catching an exception.

    **The predicate is total: every argument produces a** :class:`bool`. A
    non-``str`` is refused by :func:`validate_store_key`'s own type guard as a
    :exc:`StoreKeyError`, which this function already catches, so nothing
    escapes as an exception — there is deliberately no second guard here. That
    totality is what lets a downstream pre-check a path-typed identifier without
    wrapping the call, which is the shape the "decide per-tile without catching
    an exception" guidance above was written for.

    Parameters
    ----------
    key : str
        The candidate store key, exactly as the caller supplied it. Annotated
        ``str`` because that is the supported call, but a caller who ignores the
        annotation gets ``False`` rather than a :exc:`TypeError`.

    Returns
    -------
    bool
        ``True`` if :func:`validate_store_key` accepts ``key``, ``False`` if it
        refuses it. The call is pure: repeated calls on the same key return the
        same verdict, touch no filesystem and mutate no state.
    """
    try:
        validate_store_key(key)
    except StoreKeyError:
        return False
    return True


# ---------------------------------------------------------------------------
# Phase-14 artefact suffix vocabulary (D-14)
# ---------------------------------------------------------------------------

# The vocabulary used to live in three places — ``DiskBackedStore``'s three
# class attributes, ``LazyDiskCache._MEMMAP_SUFFIX`` on the ABC, and bare
# literals inside ``_store_entry`` — because ``disk_backed_store.py`` imports
# *from* ``lazy_disk_cache.py`` and neither could reach the other's constants.
# That split has already produced one shipped defect (``_purge_cache_pair``
# hardcodes ".npy"/".meta.json" and its FRAG-03 intent has never held). The
# constants live here so there is one vocabulary and one seam.
#
# BOUNDARY: moving the vocabulary is this phase's job; actually fixing the
# ``_purge_cache_pair`` dead branch is STORE-06 and belongs to Phase 15. The
# move is what makes that fix a one-liner rather than a refactor.

NPY_SUFFIX: Final[str] = ".npy"
META_SUFFIX: Final[str] = ".meta.json"
LEGACY_PICKLE_SUFFIX: Final[str] = ".pkl"
MEMMAP_SUFFIX: Final[str] = ".dat"
TMP_SUFFIX: Final[str] = ".tmp"


# ---------------------------------------------------------------------------
# Phase-14 resolved containment (D-03 / D-17 / STORE-02)
# ---------------------------------------------------------------------------

#: The containment clause. Kept beside the lexical ``CLAUSE_*`` vocabulary so a
#: test can import the wording rather than restate it.
CLAUSE_ESCAPES: Final[str] = "resolves outside its cache directory"


def _assert_contained(cache_dir: Path, candidate: Path) -> None:
    """Verify that ``candidate`` resolves inside ``cache_dir``, or raise.

    The second of the two layers. The lexical rule
    (:func:`validate_store_key`) runs first and refuses an untrusted *key*;
    this layer refuses a *path* that would leave the cache directory anyway,
    which — with the lexical layer in place at every route — can now only
    happen because of the environment: a symlinked or replaced directory
    component inside the cache directory.

    Three things below look like mistakes without their reasoning, so all
    three are recorded here.

    **1. Why the parent chain is resolved but the final component is not.**
    The candidate is re-formed as ``candidate.parent.resolve() /
    candidate.name`` (D-17). A full ``candidate.resolve()`` would refuse the
    legitimate *adopted entry* — a real codec pair living outside the cache
    with a final-component symlink inside the cache pointing at it — and it
    would additionally refuse **every** key whenever the cache directory is
    itself a symlink, which is the default under ``mkdtemp`` on macOS
    (``/var`` → ``/private/var``) and normal on ETH ``/scratch``. Parent-only
    resolution catches every escape full resolution catches: the reproduced
    matrix has four rows and parent-only is correct on all four. Note the
    base *is* fully resolved, so a symlinked cache directory compares equal to
    itself.

    **2. The accepted residual.** Parent-only resolution permits a
    *pre-existing* symlink placed inside the cache directory to write through
    to wherever it points. This is accepted, deliberately (D-17, threat
    T-14-18), because it is a different threat model from the one being
    fixed: to plant that symlink an attacker must already hold write access
    to the cache directory, whereas the threat being closed here is an
    untrusted **key** escaping a directory the attacker cannot write to. It
    is an accepted residual, not an oversight, and it is recorded in
    ``DESIGN-DECISIONS.md`` as well as here.

    **3. Why none of the obvious primitives is used.** Each of the three is a
    *complete* bypass, not a partial one, and all three fail in the direction
    of accepting an escape:

    * A **string prefix test** (``str(candidate).startswith(str(cache_dir))``)
      accepts ``/cache-evil/x.npy`` against ``/cache``: containment is a path
      relationship, and a sibling that merely shares the directory's name
      prefix is not inside it.
    * A **bare** :meth:`~pathlib.PurePath.is_relative_to` on an *unresolved*
      path accepts ``/cache/../victim.npy``. ``is_relative_to`` is purely
      lexical and does not collapse the ``..`` segment.
    * A **bare common-path computation** (``os.path.commonpath``) has the same
      lexical blind spot for the same reason.

    So the resolution is what does the work, and removing it silently removes
    the control. A future simplification meets this comment before it meets a
    reviewer.

    The base is ``cache_dir.resolve()``, computed **inside this function on
    every call** (D-03). It must never become an instance attribute, a
    module-level cache or a default argument:
    :meth:`DiskBackedStore.__getstate__` is ``self.__dict__.copy()``, so a
    stored resolved base would be pickled verbatim and travel into a worker
    that may be on a different mount namespace, where it would be compared
    against paths that genuinely differ. Re-deriving removes the failure mode
    structurally instead of defending against it with symmetric state hooks
    that must stay in sync forever.

    Both ``resolve()`` calls are non-strict (the :mod:`pathlib` default). The
    builders run *before* the file exists, so strict resolution would raise
    :exc:`FileNotFoundError` on every ordinary write.

    Parameters
    ----------
    cache_dir : Path
        The configured cache directory; the authorization boundary.
    candidate : Path
        The path a builder is about to return.

    Returns
    -------
    None
        Returns ``None`` when ``candidate`` is contained.

    Raises
    ------
    StoreContainmentError
        If ``candidate`` resolves outside ``cache_dir``. It subclasses
        :exc:`StoreKeyError` (and therefore :exc:`ValueError`), so a broad
        handler catches both layers, while the distinct type says *which*
        layer fired: a bare :exc:`StoreKeyError` means the caller passed a bad
        key, a :exc:`StoreContainmentError` means the environment is wrong.
    """
    base = cache_dir.resolve()
    resolved = candidate.parent.resolve() / candidate.name
    if not resolved.is_relative_to(base):
        raise StoreContainmentError(
            f"Store path {str(candidate)!r} for cache directory {str(cache_dir)!r}: the path "
            f"{CLAUSE_ESCAPES} — it resolves to {str(resolved)!r}, which is not inside the "
            f"resolved cache directory {str(base)!r}; this is evidence about the environment "
            "rather than about the key (most likely a symlinked or replaced directory component "
            "inside the cache directory), so inspect the cache directory before retrying."
        )


# ---------------------------------------------------------------------------
# Phase-14 path builders and the builder registry (D-14 / D-15 / STORE-02)
# ---------------------------------------------------------------------------


def _build(cache_dir: Path, key: str, suffix: str) -> Path:
    """Validate ``key``, join ``suffix``, verify containment and return the path.

    The single seam every builder goes through, so the two layers cannot run
    in different orders at different builders and the suffix cannot be joined
    by two different shapes.

    The **order is load-bearing**: lexical refusal first, resolved containment
    second. A key such as ``'../victim'`` violates *both* layers, and running
    the lexical check first is what makes it raise the base
    :exc:`StoreKeyError` rather than the :exc:`StoreContainmentError`
    subclass. That keeps the exception type a stable signal of which layer
    fired — a bad key from the caller, versus a symlink planted in the cache
    directory. Inverted, every bad key would look like an attack.

    Parameters
    ----------
    cache_dir : Path
        The configured cache directory.
    key : str
        The candidate store key, exactly as the caller supplied it.
    suffix : str
        The full artefact suffix, joined onto ``key`` as a string. The suffix
        is never applied with :meth:`~pathlib.PurePath.with_suffix` on another
        builder's result — one concept, one construction shape.

    Returns
    -------
    Path
        ``cache_dir / f"{key}{suffix}"``.

    Raises
    ------
    StoreKeyError
        If ``key`` is not a legal single-segment store key.
    StoreContainmentError
        If the joined path resolves outside ``cache_dir``.
    """
    validate_store_key(key, cache_dir)
    candidate = cache_dir / f"{key}{suffix}"
    _assert_contained(cache_dir, candidate)
    return candidate


def get_npy_path(cache_dir: Path, key: str) -> Path:
    """Return the on-disk ``<key>.npy`` array path inside ``cache_dir``.

    Parameters
    ----------
    cache_dir : Path
        The configured cache directory.
    key : str
        The store key.

    Returns
    -------
    Path
        The validated, contained ``.npy`` path.

    Raises
    ------
    StoreKeyError
        If ``key`` is not a legal single-segment store key.
    StoreContainmentError
        If the path would resolve outside ``cache_dir``.
    """
    return _build(cache_dir, key, NPY_SUFFIX)


def get_meta_path(cache_dir: Path, key: str) -> Path:
    """Return the on-disk ``<key>.meta.json`` sidecar path inside ``cache_dir``.

    Parameters
    ----------
    cache_dir : Path
        The configured cache directory.
    key : str
        The store key.

    Returns
    -------
    Path
        The validated, contained sidecar path.

    Raises
    ------
    StoreKeyError
        If ``key`` is not a legal single-segment store key.
    StoreContainmentError
        If the path would resolve outside ``cache_dir``.
    """
    return _build(cache_dir, key, META_SUFFIX)


def get_legacy_pickle_path(cache_dir: Path, key: str) -> Path:
    """Return the legacy pre-Phase-2 ``<key>.pkl`` path inside ``cache_dir``.

    The artefact is refused on read; the path is still built (and therefore
    still guarded) because the reader probes for its existence.

    Parameters
    ----------
    cache_dir : Path
        The configured cache directory.
    key : str
        The store key.

    Returns
    -------
    Path
        The validated, contained legacy-pickle path.

    Raises
    ------
    StoreKeyError
        If ``key`` is not a legal single-segment store key.
    StoreContainmentError
        If the path would resolve outside ``cache_dir``.
    """
    return _build(cache_dir, key, LEGACY_PICKLE_SUFFIX)


def get_npy_tmp_path(cache_dir: Path, key: str) -> Path:
    """Return the temporary ``<key>.npy.tmp`` write path inside ``cache_dir``.

    Internal construction detail with no downstream caller, and deliberately
    absent from the package's published surface.

    Parameters
    ----------
    cache_dir : Path
        The configured cache directory.
    key : str
        The store key.

    Returns
    -------
    Path
        The validated, contained temporary array path.

    Raises
    ------
    StoreKeyError
        If ``key`` is not a legal single-segment store key.
    StoreContainmentError
        If the path would resolve outside ``cache_dir``.
    """
    return _build(cache_dir, key, f"{NPY_SUFFIX}{TMP_SUFFIX}")


def get_meta_tmp_path(cache_dir: Path, key: str) -> Path:
    """Return the temporary ``<key>.meta.json.tmp`` write path inside ``cache_dir``.

    Built by exactly the same shape as every other artefact name — the suffix
    is joined onto the key here, not derived from another builder's result.
    Before this seam existed the store constructed this one name inline, and
    it was safe only *incidentally*, because the real ``.meta.json`` builder
    ran three lines earlier and would already have raised. That is a filter;
    this is the invariant.

    Parameters
    ----------
    cache_dir : Path
        The configured cache directory.
    key : str
        The store key.

    Returns
    -------
    Path
        The validated, contained temporary sidecar path.

    Raises
    ------
    StoreKeyError
        If ``key`` is not a legal single-segment store key.
    StoreContainmentError
        If the path would resolve outside ``cache_dir``.
    """
    return _build(cache_dir, key, f"{META_SUFFIX}{TMP_SUFFIX}")


#: Every builder, by name. Deliberately **not** in ``__all__``: it is a
#: test-facing seam rather than published surface. The guarded-builder contract
#: test iterates this registry instead of hand-listing cases, so a sixth builder
#: added later without joining it fails that test by omission rather than
#: escaping it silently.
STORE_PATH_BUILDERS: dict[str, Callable[[Path, str], Path]] = {
    "get_npy_path": get_npy_path,
    "get_meta_path": get_meta_path,
    "get_legacy_pickle_path": get_legacy_pickle_path,
    "get_npy_tmp_path": get_npy_tmp_path,
    "get_meta_tmp_path": get_meta_tmp_path,
}
