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
]

import logging
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final, NoReturn, Optional

logger = logging.getLogger(__name__)


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
        If ``key`` violates any clause of the rule. ``StoreKeyError`` subclasses
        :class:`ValueError` (D-12), so every existing ``except ValueError``
        still catches it, and it is deliberately *not* a :class:`KeyError` —
        :meth:`DiskBackedStore.add_data_to_store` already raises that one for
        "key exists".
    """
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

    Parameters
    ----------
    key : str
        The candidate store key, exactly as the caller supplied it.

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
