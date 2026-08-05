"""Lexical store-key rule matrix for Phase 14 / STORE-01.

Covers D-04 (a store key is a single path segment), D-05 (a *property
denylist*, never an allowlist charset, evaluated over the exact characters the
caller supplied), D-06 (dots are legal except the exact ``.`` and ``..``) and
D-07 (the published predicate agrees with the internal raising validator).
This file is created by Plan 14-02; Plan 14-01 shipped the rule it pins and
that rule is frozen for the duration of Wave 2.

Pure lexical by construction: this module constructs no store, touches no
filesystem and takes no fixture. That is what makes it the phase's cheapest
and most frequently-run feedback signal.

``pytest.mark.parametrize`` has no prior occurrence anywhere in this suite, so
the convention established here is: one ``@pytest.mark.parametrize("key", ...)``
decorator per group, ``ids=`` omitted (pytest's generated ids render the
escapes readably), and the case lists hoisted to module-level constants so a
later group can reuse them rather than restate them.
"""

import re
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from GSEGUtils.lazy_disk_cache.paths import (
    CLAUSE_ABSOLUTE,
    CLAUSE_CONTROL,
    CLAUSE_RESERVED,
    CLAUSE_SEPARATOR,
    StoreKeyError,
    is_valid_store_key,
    validate_store_key,
)

# A plain ``Path`` value used only so the refusal message has a cache directory
# to name (D-13). It is never created, opened, resolved or stat-ed — nothing in
# this module touches the filesystem.
_CACHE_DIR = Path("/tmp/cache")


# ---------------------------------------------------------------------------
# The case lists (shared by every group, including Task 2's agreement test)
# ---------------------------------------------------------------------------

#: D-04's core: nested keys are refused. ``'a/'`` is here as well as in the
#: degenerate set because a trailing separator is a separator.
SEPARATOR_KEYS: tuple[str, ...] = ("../victim", "a/b", "tile_03/range", "a/")

#: The Windows-semantics escape. Refused on Linux; see the dedicated test,
#: which pins the *mechanism* rather than only the outcome.
WINDOWS_TRAVERSAL_KEY: str = "..\\..\\x"

#: Absolute, drive-relative and UNC. The last two are here deliberately: a
#: hand-rolled character scan for ``/`` and ``\`` alone misses ``C:evil``.
ABSOLUTE_KEYS: tuple[str, ...] = ("/etc/passwd", "C:evil", "C:/abs", "\\\\server\\share\\x")

#: The degenerate set from D-05. Every member is also present in
#: :data:`REFUSED_KEYS` with the clause it must report.
DEGENERATE_KEYS: tuple[str, ...] = ("", ".", "..", "a/", "a\\", "x\n", "x\x00y")

#: Two fullwidth U+FF0E FULL STOP characters. They NFKC-fold to the refused
#: ``'..'``, so accepting them unchanged is the positive evidence that the rule
#: applies no normalisation of any kind (D-05).
FULLWIDTH_DOTS_KEY: str = "\uff0e\uff0e"

#: The same fullwidth dots carrying a real ASCII separator. Refused for its
#: separator, not for its dots.
FULLWIDTH_SEPARATOR_KEY: str = "\uff0e\uff0e/victim"

#: Every key the rule must refuse, paired with the clause it must report under
#: the validator's fixed evaluation order. One list serves the refusal groups,
#: Task 2's predicate-agreement group and Task 2's message assertions — which
#: is the mechanism that keeps the two functions from drifting apart.
REFUSED_KEYS: list[tuple[str, str]] = [
    *[(key, CLAUSE_SEPARATOR) for key in SEPARATOR_KEYS],
    (WINDOWS_TRAVERSAL_KEY, CLAUSE_SEPARATOR),
    *[(key, CLAUSE_ABSOLUTE) for key in ABSOLUTE_KEYS],
    ("", CLAUSE_RESERVED),
    (".", CLAUSE_RESERVED),
    ("..", CLAUSE_RESERVED),
    ("a\\", CLAUSE_SEPARATOR),
    ("x\n", CLAUSE_CONTROL),
    ("x\x00y", CLAUSE_CONTROL),
    (FULLWIDTH_SEPARATOR_KEY, CLAUSE_SEPARATOR),
]

#: Every key the rule must accept. This is the over-tight-allowlist regression
#: guard and it protects a cross-repo contract: pc2img treats feature names as
#: simultaneously public API and cache key, so an allowlist drawn around
#: today's grammar makes the next legitimate feature name a break.
ACCEPTED_KEYS: tuple[str, ...] = (
    ".hidden",
    "foo.bar",
    "z1.2345678",
    "foo.",
    "a.npy",
    "rrim_pack_(range,r16,d8,z1e-05)",
    "norm_(range,2,98)",
    FULLWIDTH_DOTS_KEY,
)

#: Lookup from a refused key to the clause it must report.
EXPECTED_CLAUSE: dict[str, str] = dict(REFUSED_KEYS)


def test_refused_key_groups_are_all_present_in_the_master_list() -> None:
    """Plan 14-02: every group list is a subset of ``REFUSED_KEYS``.

    ``REFUSED_KEYS`` is what Task 2's agreement and message groups iterate, so
    a key exercised by a group here but missing from the master list would be
    silently exempt from the predicate-agreement proof.
    """
    master = set(EXPECTED_CLAUSE)
    for group_name, group in (
        ("SEPARATOR_KEYS", SEPARATOR_KEYS),
        ("ABSOLUTE_KEYS", ABSOLUTE_KEYS),
        ("DEGENERATE_KEYS", DEGENERATE_KEYS),
        ("WINDOWS_TRAVERSAL_KEY", (WINDOWS_TRAVERSAL_KEY,)),
        ("FULLWIDTH_SEPARATOR_KEY", (FULLWIDTH_SEPARATOR_KEY,)),
    ):
        missing = sorted(set(group) - master)
        assert not missing, f"{group_name} carries keys absent from REFUSED_KEYS: {missing!r}"
    assert len(REFUSED_KEYS) == len(master), "REFUSED_KEYS contains a duplicate key"


# ---------------------------------------------------------------------------
# Group 1 — separators (D-04)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", SEPARATOR_KEYS)
def test_separator_bearing_key_is_refused(key: str) -> None:
    """Plan 14-02 / STORE-01 / D-04: a key carrying a path separator is refused.

    Naming why this is a bug fix rather than a preference: ``DiskBackedStore``'s
    ``__init__`` rescan globs ``*.npy`` **non-recursively**, so a nested key
    like ``tile_03/range`` inserts and offloads perfectly well and then vanishes
    on store reopen while its file leaks forever. The separator is also the
    entire escape mechanism (D-06) — ``<cache>/../victim.npy`` lands outside the
    cache directory while ``<cache>/...npy`` (key ``'..'``) does not.
    """
    with pytest.raises(StoreKeyError, match=re.escape(CLAUSE_SEPARATOR)):
        validate_store_key(key, _CACHE_DIR)
    assert not is_valid_store_key(key), f"the predicate accepted the separator-bearing key {key!r}"


# ---------------------------------------------------------------------------
# Group 2 — the Windows-semantics escape (D-05, T-14-06)
# ---------------------------------------------------------------------------


def test_windows_interpretation_refuses_backslash_traversal_on_linux() -> None:
    r"""Plan 14-02 / STORE-01 / D-05: ``..\..\x`` is refused on a POSIX host.

    This test pins the **mechanism**, not just the outcome. Under POSIX path
    semantics the key is a *single* harmless segment; only under Windows
    semantics is it three segments, the middle two of which walk upward. The
    segment-count assertions are therefore load-bearing: without them the test
    would still pass if someone later reduced the rule to a bare character
    scan, and the dual interpretation is the only thing that catches this class
    of key on Linux structurally rather than incidentally.
    """
    posix_parts = PurePosixPath(WINDOWS_TRAVERSAL_KEY).parts
    windows_parts = PureWindowsPath(WINDOWS_TRAVERSAL_KEY).parts
    assert len(posix_parts) == 1, (
        f"POSIX semantics no longer read {WINDOWS_TRAVERSAL_KEY!r} as one segment ({posix_parts!r}); "
        "the premise of the dual-interpretation rule has changed"
    )
    assert len(windows_parts) == 3, (
        f"Windows semantics no longer read {WINDOWS_TRAVERSAL_KEY!r} as three segments ({windows_parts!r}); "
        "the dual-interpretation rule can no longer catch this escape structurally"
    )

    with pytest.raises(StoreKeyError, match=re.escape(CLAUSE_SEPARATOR)):
        validate_store_key(WINDOWS_TRAVERSAL_KEY, _CACHE_DIR)
    assert not is_valid_store_key(WINDOWS_TRAVERSAL_KEY), (
        "the predicate accepted a key that walks upward under Windows path semantics"
    )


# ---------------------------------------------------------------------------
# Group 3 — absolute, drive-relative and UNC (D-05)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ABSOLUTE_KEYS)
def test_absolute_or_drive_relative_key_is_refused(key: str) -> None:
    r"""Plan 14-02 / STORE-01 / D-05: absolute, drive-relative and UNC keys are refused.

    ``C:evil`` and ``\\server\share\x`` are in this group deliberately.
    ``C:evil`` carries no separator at all, so a hand-rolled scan for ``/`` and
    ``\`` misses it entirely; both are caught by testing ``anchor`` and
    ``drive`` under *both* path flavours. Note the clause is the absolute one
    even for the UNC key, whose Windows ``parts`` is also multi-segment — the
    validator's fixed evaluation order is what makes that deterministic.
    """
    with pytest.raises(StoreKeyError, match=re.escape(CLAUSE_ABSOLUTE)):
        validate_store_key(key, _CACHE_DIR)
    assert not is_valid_store_key(key), f"the predicate accepted the absolute/drive-relative key {key!r}"


# ---------------------------------------------------------------------------
# Group 4 — the degenerate set (D-05)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", DEGENERATE_KEYS)
def test_degenerate_key_is_refused(key: str) -> None:
    r"""Plan 14-02 / STORE-01 / D-05: the degenerate set is refused, each with its own clause.

    Three members look arbitrary and are not:

    ``''`` — the empty key builds ``<cache>/.npy``, whose rescan stem is
    ``'.npy'``: a *different*, legal-looking key. It is refused because it does
    not round-trip, not because it is untidy.

    ``'x\x00y'`` — a NUL-bearing key **builds a path without raising** and only
    fails later at ``write_bytes``, i.e. after state may already have been
    mutated. Refusing it lexically moves the failure to the one place where
    nothing has happened yet.

    ``'a/'`` and ``'a\'`` — these are *not* redundant with the separator group.
    ``pathlib`` normalises a trailing separator away, so
    ``PurePosixPath('a/').parts`` and ``PureWindowsPath('a\').parts`` are both
    ``('a',)`` — length 1 — and a structural ``parts``-length test therefore
    **accepts** them. They are refused by Clause 4's raw-character sub-clause,
    and this pair is the only thing standing between that sub-clause and a
    future "simplification" back to a length test. ``'a/'`` also has a concrete
    consequence: joined with a suffix it builds ``<cache>/a/.npy`` — a
    *directory* ``a`` holding a hidden file, which the non-recursive rescan
    cannot see at all.
    """
    expected_clause = EXPECTED_CLAUSE[key]
    with pytest.raises(StoreKeyError, match=re.escape(expected_clause)):
        validate_store_key(key, _CACHE_DIR)
    assert not is_valid_store_key(key), f"the predicate accepted the degenerate key {key!r}"


def test_degenerate_trailing_separators_survive_pathlib_normalisation() -> None:
    r"""Plan 14-02 / STORE-01 / D-05: ``pathlib`` folds a trailing separator away, so the rule cannot rely on it.

    This is the measured premise behind the two trailing-separator cases above,
    asserted rather than assumed. If a future ``pathlib`` stopped normalising,
    this test would flag that the raw-character sub-clause is no longer the
    only thing refusing ``'a/'``.
    """
    assert PurePosixPath("a/").parts == ("a",), "PurePosixPath no longer folds a trailing '/' away"
    assert PureWindowsPath("a\\").parts == ("a",), "PureWindowsPath no longer folds a trailing '\\' away"


# ---------------------------------------------------------------------------
# Group 5 — the accepted set (D-06, T-14-08)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ACCEPTED_KEYS)
def test_accepts_dotted_and_composed_feature_names(key: str) -> None:
    """Plan 14-02 / STORE-01 / D-06: dots are legal except the exact ``.`` and ``..``.

    The rule is a *denylist over properties*, not an allowlist over a charset.
    This group is the regression guard for that: ``rrim_pack_(range,r16,d8,z1e-05)``
    and ``norm_(range,2,98)`` are real pc2img feature names, which are
    simultaneously public API and cache key over there. An allowlist drawn
    around today's grammar would make the next legitimate feature name a
    cross-repo break, and this test is what turns that from a review comment
    into a failing build.
    """
    validate_store_key(key, _CACHE_DIR)  # must not raise
    assert is_valid_store_key(key), f"the predicate refused the legal key {key!r}"


# ---------------------------------------------------------------------------
# Group 6 — no normalisation (D-05, T-14-07)
# ---------------------------------------------------------------------------


def test_no_normalisation_accepts_a_fullwidth_dots_key_unchanged() -> None:
    """Plan 14-02 / STORE-01 / D-05: a fullwidth-dots key is accepted, proving no NFKC folding.

    ``'\uff0e\uff0e'`` folds to the refused ``'..'`` under NFKC. Accepting it
    unchanged is therefore the *positive* evidence that the rule normalises
    nothing — a negative ("we do not call ``unicodedata``") is not observable
    from outside the module, while this is.
    """
    validate_store_key(FULLWIDTH_DOTS_KEY, _CACHE_DIR)  # must not raise
    assert is_valid_store_key(FULLWIDTH_DOTS_KEY), (
        "a fullwidth-dots key was refused, which means the rule folded it to '..' before validating"
    )


def test_no_normalisation_refuses_a_fullwidth_key_for_its_separator_clause() -> None:
    """Plan 14-02 / STORE-01 / D-05: a fullwidth key with a real separator is refused *for the separator*.

    The pair with the test above is the point: the same fullwidth characters
    are accepted alone and refused when an ASCII separator joins them, and the
    reported clause is the separator one rather than the reserved-name one.
    Validating one string and building the path from another is the classic
    bypass; this pins that the characters validated are the characters
    supplied.
    """
    with pytest.raises(StoreKeyError, match=re.escape(CLAUSE_SEPARATOR)):
        validate_store_key(FULLWIDTH_SEPARATOR_KEY, _CACHE_DIR)
    assert not is_valid_store_key(FULLWIDTH_SEPARATOR_KEY), (
        f"the predicate accepted {FULLWIDTH_SEPARATOR_KEY!r}, which carries a real ASCII separator"
    )
