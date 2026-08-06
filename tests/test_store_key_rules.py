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
from typing import cast

import pytest

from GSEGUtils.lazy_disk_cache.paths import (
    CLAUSE_ABSOLUTE,
    CLAUSE_CONTROL,
    CLAUSE_RESERVED,
    CLAUSE_SEPARATOR,
    CLAUSE_TRAILING,
    StoreContainmentError,
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

#: The six keys § WR-04 **measured as accepted** on this host against the Plan
#: 14-09 device set, closed by Plan 14-13 / D-23. Kept as its own tuple so the
#: dedicated group below reads as the closure of a reproduced finding rather than
#: as a generic list of extra names.
WR04_ACCEPTED_DEVICE_KEYS: tuple[str, ...] = ("COM0", "LPT0", "CONIN$", "CONOUT$", "com¹", "CON .txt")

#: The fullwidth spelling of a device name — U+FF23 U+FF2F U+FF2E, ``ＣＯＮ``
#: (Plan 14-13 / D-23). It is **accepted**, and that acceptance is the standing
#: proof that a rule which now compares against a *name list* still folds
#: nothing: ``'ＣＯＮ'.upper()`` is ``'ＣＯＮ'``, which stays outside the ASCII
#: set. On Win32 it is an ordinary filename and not the console device, so
#: accepting it is correct rather than an oversight. Same role for the device
#: clause that :data:`FULLWIDTH_DOTS_KEY` plays for the dot clauses.
FULLWIDTH_DEVICE_KEY: str = "ＣＯＮ"

#: Reserved Win32 device names (Plan 14-09 / D-20 / WR-05, widened by Plan 14-13
#: / D-23). Matched on the **pre-dot stem**, upper-cased, so an
#: extension-bearing device name is caught too — on Win32 ``con.npy`` is still
#: the console with a suffix attached, and a write to it is discarded while a
#: read comes back empty.
#:
#: ``lpt9`` is lower-case and ``con.npy`` mixed with an extension on purpose:
#: between them they pin that the match is case-insensitive *and* stem-based,
#: which a set-membership test over the raw key would fail.
#:
#: D-23's additions follow: the six § WR-04 keys, plus ``COM²`` and ``lpt³`` so
#: the superscript forms are covered in both cases. The superscript entries are
#: **exact characters, not a case fold** — ``'¹'.upper()`` is ``'¹'`` — which is
#: what keeps the widening inside D-05's no-normalisation rule.
WIN_DEVICE_KEYS: tuple[str, ...] = (
    "CON",
    "NUL",
    "PRN",
    "AUX",
    "lpt9",
    "con.npy",
    "COM1.dat",
    *WR04_ACCEPTED_DEVICE_KEYS,
    "COM²",
    "lpt³",
)

#: Keys whose trailing run is spaces or dots (Plan 14-09 / D-20 / WR-05). Win32
#: strips both, so ``'a'``, ``'a '`` and ``'a.'`` become **one file** — two
#: distinct store keys silently overwriting one artefact.
#:
#: ``'...'`` and ``' '`` are not among the keys D-20 enumerates and are here
#: because they are a *wider consequence* of the rule as written: stripping the
#: trailing run empties them entirely. Discovering that from a downstream rather
#: than from this list is exactly the outcome the list exists to prevent.
TRAILING_KEYS: tuple[str, ...] = ("foo.", "a ", "x.", "...", " ")

#: An NTFS alternate-data-stream name (Plan 14-09 / D-20 / WR-05). It needs its
#: own case because :class:`~pathlib.PureWindowsPath` detects only *single-letter*
#: drives: ``PureWindowsPath('ab:cd').drive`` is ``''``, so the pre-existing
#: ``anchor``/``drive`` test does not catch it and no amount of tightening that
#: test would. Reported under the absolute clause, which already covers
#: drive-relative constructs.
COLON_KEY: str = "ab:cd"

#: The D-20 collision additions with their clauses. Kept as its own list so the
#: widening is legible as one decision rather than scattered through
#: :data:`REFUSED_KEYS`, and so the group-subset guard can name it.
WIN_COLLISION_KEYS: list[tuple[str, str]] = [
    *[(key, CLAUSE_RESERVED) for key in WIN_DEVICE_KEYS],
    *[(key, CLAUSE_TRAILING) for key in TRAILING_KEYS],
    (COLON_KEY, CLAUSE_ABSOLUTE),
]

#: The two families whose **opposite pulls** fix the shape of the device clause's
#: stem strip (Plan 14-13 / D-23 / D-24). Neither was in :data:`REFUSED_KEYS`
#: before, which is exactly why an unconditional strip would have repointed three
#: of them with nothing going red.
#:
#: * ``'com1 '``, ``'con '``, ``'nul  '`` — a device stem followed by the key's
#:   **own** trailing run. They report the *trailing* clause and must keep doing
#:   so. An **unconditional** ``rstrip(' ')`` on the stem turns each of them into
#:   a device stem and repoints them onto the reserved clause. The strip is
#:   therefore conditional on the key containing a ``'.'``: interior spaces are
#:   the device clause's business, trailing spaces are clause 5's.
#: * ``'con.'``, ``'com1. '`` — a device stem that *also* ends in a trailing run.
#:   They report the *reserved* clause and must keep doing so, which is what
#:   forbids the naive "just put the device test last" placement.
DEVICE_STEM_BOUNDARY_KEYS: list[tuple[str, str]] = [
    ("com1 ", CLAUSE_TRAILING),
    ("con ", CLAUSE_TRAILING),
    ("nul  ", CLAUSE_TRAILING),
    ("con.", CLAUSE_RESERVED),
    ("com1. ", CLAUSE_RESERVED),
]

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
    *WIN_COLLISION_KEYS,
    *DEVICE_STEM_BOUNDARY_KEYS,
]

#: Every key the rule must accept. This is the over-tight-allowlist regression
#: guard and it protects a cross-repo contract: pc2img treats feature names as
#: simultaneously public API and cache key, so an allowlist drawn around
#: today's grammar makes the next legitimate feature name a break.
#:
#: ↻ AMENDED by Plan 14-09 (D-20, which amends locked D-06): ``'foo.'`` used to
#: be a member of this tuple and is now in :data:`TRAILING_KEYS` instead. That
#: move is the *point* of the widening and it makes Phase 14 unambiguously
#: breaking — **do not restore it**. Leading dots are unaffected: ``.hidden`` is
#: still here, because the rule refuses a trailing dot, not a dot.
ACCEPTED_KEYS: tuple[str, ...] = (
    ".hidden",
    "foo.bar",
    "z1.2345678",
    "a.npy",
    "rrim_pack_(range,r16,d8,z1e-05)",
    "norm_(range,2,98)",
    FULLWIDTH_DOTS_KEY,
    # Plan 14-13 / D-23: the widening's *positive* boundary, pinned in the same
    # place as its negative one. Do not remove it when the device list is next
    # extended — it is what proves the list is compared, not folded into.
    FULLWIDTH_DEVICE_KEY,
)

#: Lookup from a refused key to the clause it must report.
EXPECTED_CLAUSE: dict[str, str] = dict(REFUSED_KEYS)

#: The non-``str`` inputs (Plan 14-09 / WR-02). The first four are the review's
#: verbatim reproduction, each of which escaped as a bare :exc:`TypeError` from
#: the control-character scan on a predicate documented to return ``bool``.
#:
#: The fifth is **not** from that reproduction and is here for a different
#: reason: a :class:`bytearray` is *unhashable*, so ``key in _RESERVED_KEYS``
#: raises ``TypeError: unhashable type: 'bytearray'`` before any character is
#: examined. It is what makes the guard's *position* — first statement, ahead of
#: the reserved-name membership test — provable rather than stylistic: move the
#: guard after that test and this case alone goes red.
#:
#: Deliberately **not** folded into :data:`REFUSED_KEYS`, whose
#: ``list[tuple[str, str]]`` typing feeds :data:`EXPECTED_CLAUSE` and the
#: group-subset guard. There is also no ``CLAUSE_*`` constant for this case: the
#: clause vocabulary describes *how a key is shaped*, and these values are not
#: keys at all.
NON_STR_KEYS: list[object] = [
    None,
    Path("../victim"),
    b"../victim",
    5,
    bytearray(b"../victim"),
]


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
        ("WIN_DEVICE_KEYS", WIN_DEVICE_KEYS),
        ("WR04_ACCEPTED_DEVICE_KEYS", WR04_ACCEPTED_DEVICE_KEYS),
        ("TRAILING_KEYS", TRAILING_KEYS),
        ("COLON_KEY", (COLON_KEY,)),
        ("DEVICE_STEM_BOUNDARY_KEYS", tuple(key for key, _ in DEVICE_STEM_BOUNDARY_KEYS)),
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
# Group 4b — Win32 / APFS collision shapes (Plan 14-09 / D-20 / WR-05,
#            T-14-31, T-14-32)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("key", "clause"), WIN_COLLISION_KEYS)
def test_win32_collision_shaped_key_is_refused(key: str, clause: str) -> None:
    r"""Plan 14-09 / STORE-01 / D-20: collision shapes are refused, not only escape shapes.

    The rule already imported :class:`~pathlib.PureWindowsPath` to refuse
    ``..\..\x``, ``C:evil`` and UNC *on Linux*, so Win32 was inside the threat
    model by construction — but it stopped at **escape**. These keys are
    **collisions**, a different threat: two distinct keys collapsing onto one
    artefact, which is the sanitise-rather-than-reject failure this phase set
    out to avoid.

    * A **trailing space or dot** is stripped by Win32, so ``'a'``, ``'a '`` and
      ``'a.'`` become one file and one silently overwrites the other's data.
    * A **device name** routes the write to a character device: writes are
      discarded and reads come back empty. It is still the device with a suffix
      attached, so ``con.npy`` is refused too.
    * A **colon** opens an NTFS alternate data stream, hiding bytes outside the
      visible artefact.

    None of these is an escape, and the containment layer cannot see any of them
    — they are all lexically inside the cache directory. The lexical rule is the
    only layer that can refuse them.
    """
    with pytest.raises(StoreKeyError, match=re.escape(clause)):
        validate_store_key(key, _CACHE_DIR)
    assert not is_valid_store_key(key), f"the predicate accepted the collision-shaped key {key!r}"


@pytest.mark.parametrize("key", WR04_ACCEPTED_DEVICE_KEYS)
def test_widened_device_list_refuses_every_reserved_name_the_contract_page_claims(key: str) -> None:
    """Plan 14-13 / STORE-01 / D-23 / WR-04: the denylist covers the list the documentation appeals to.

    D-20 widened the rule to Win32 collision semantics and both the contract page
    and ``validate_store_key``'s residual paragraph presented the device axis as
    closed. § WR-04 measured six keys **accepted** against that set. Each is here
    with its reason:

    * ``COM0`` and ``LPT0`` — Microsoft's *Naming Files, Paths, and Namespaces*
      reserved list, the list the code's own comment appeals to, runs the digit
      from ``0``; the shipped comprehensions started at ``1``.
    * ``CONIN$`` and ``CONOUT$`` — reserved console names on the same list.
    * ``com¹`` — the superscript ``COM``/``LPT`` forms are on that list too, and
      they are matched as **exact characters**, so the lower-case spelling here
      pins that the existing ``.upper()`` reaches them.
    * ``CON .txt`` — Win32 strips trailing spaces from the name component
      *before* device resolution, so it resolves to the console while passing
      both the old device test (stem ``'CON '`` upper-cases outside the set) and
      the trailing test (the key ends in ``t``). It is what the conditional stem
      strip exists for.

    **What is and is not confirmed here**, per § WR-04's own note: the *acceptance
    measurements* were reproduced on this Linux host and this test asserts their
    closure. The Win32 *resolution mechanisms* quoted above cannot be confirmed on
    a Linux host and are not what this test measures.
    """
    with pytest.raises(StoreKeyError, match=re.escape(CLAUSE_RESERVED)):
        validate_store_key(key, _CACHE_DIR)
    assert not is_valid_store_key(key), f"the predicate accepted the reserved device name {key!r}"


@pytest.mark.parametrize(("key", "clause"), DEVICE_STEM_BOUNDARY_KEYS)
def test_the_device_stem_strip_stays_out_of_the_trailing_clause_s_territory(key: str, clause: str) -> None:
    """Plan 14-13 / STORE-01 / D-23: the two boundary families keep the clauses they have always reported.

    This is a **control**, and it is deliberately green both before and after
    D-23's widening: every one of these five keys reported the clause asserted
    here before the device half existed at all. It exists to make the *conditional*
    in :func:`~GSEGUtils.lazy_disk_cache.paths._device_stem` load-bearing, which
    no other test in this file does.

    Make the stem strip **unconditional** — right-strip whether or not the key
    contains a ``'.'`` — and the first three rows go red: ``'com1 '``, ``'con '``
    and ``'nul  '`` become device stems and repoint from the trailing clause onto
    the reserved one. That is a repoint of pre-existing refusals introduced by the
    very widening D-24 exists to keep from repointing anything, and without this
    group it happens with nothing going red, because no key of that shape was in
    :data:`REFUSED_KEYS` before Plan 14-13 put it there.

    The last two rows pull the other way and constrain the *placement* rather than
    the strip: ``'con.'`` and ``'com1. '`` are device stems that also end in a
    trailing run, and they must keep reporting the reserved clause. That is what
    forbids the naive "just put the device test last" reading of D-24.
    """
    with pytest.raises(StoreKeyError, match=re.escape(clause)) as excinfo:
        validate_store_key(key, _CACHE_DIR)
    assert not is_valid_store_key(key), f"the predicate accepted the boundary-family key {key!r}"

    other = CLAUSE_RESERVED if clause == CLAUSE_TRAILING else CLAUSE_TRAILING
    assert other not in str(excinfo.value), (
        f"{key!r} is now reported under {other!r} as well as {clause!r}; the device clause and the "
        "trailing clause have stopped dividing the space cleanly"
    )


def test_a_device_name_in_a_fullwidth_spelling_is_still_accepted() -> None:
    """Plan 14-13 / STORE-01 / D-23 / D-05: the device widening folds nothing.

    ``'ＣＯＮ'`` (U+FF23 U+FF2F U+FF2E) NFKC-folds to the refused ``'CON'``.
    Accepting it unchanged is the *positive* evidence that a rule which now
    compares against a **name list** still applies no normalisation — the same
    role :func:`test_no_normalisation_accepts_a_fullwidth_dots_key_unchanged`
    plays for the dot clauses, and the reason D-23's superscript entries are
    exact characters rather than a fold.

    The ``.upper()`` assertion is load-bearing rather than decorative: it is the
    mechanism claim behind the acceptance, and without it the test would still
    pass if a future edit refused fullwidth keys for some unrelated reason and
    then someone "fixed" the acceptance by deleting the key.

    Confirmed before the case was written: the key trips no other clause either —
    no control character, no separator, no colon, no anchor or drive, and no
    trailing space or dot — so its acceptance is attributable to the device
    clause and not incidental.
    """
    assert FULLWIDTH_DEVICE_KEY.upper() == FULLWIDTH_DEVICE_KEY, (
        "the fullwidth spelling now upper-cases to something else; the no-fold argument has changed"
    )
    assert FULLWIDTH_DEVICE_KEY.upper() != "CON", (
        "the fullwidth spelling upper-cased into the ASCII device set, which means str.upper() started folding"
    )

    validate_store_key(FULLWIDTH_DEVICE_KEY, _CACHE_DIR)  # must not raise
    assert is_valid_store_key(FULLWIDTH_DEVICE_KEY), (
        "a fullwidth device name was refused, which means the widened rule folded it to 'CON' before validating"
    )


def test_widening_refuses_a_trailing_dot_but_not_a_leading_or_interior_one() -> None:
    """Plan 14-09 / STORE-01 / D-20 (amending D-06): the rule refuses trailing dots, not dots.

    The obvious reading of "dots are now restricted" is wrong, and this test is
    what makes the boundary observable rather than a docstring claim. D-06 made
    **all** dots legal except the exact ``.`` and ``..``; D-20 removes exactly
    one case from that — the trailing run — and leaves leading and interior dots
    untouched.

    The paired assertions are the point: the same stem is legal with the dot in
    front and refused with the dot behind. A future "simplification" that
    refused dots generally would fail here rather than in a downstream's feature
    name.
    """
    for legal in (".hidden", ".x", "foo.bar", "z1.2345678", "a.npy"):
        validate_store_key(legal, _CACHE_DIR)  # must not raise
        assert is_valid_store_key(legal), f"the widening refused the legal dotted key {legal!r}"

    for refused in ("hidden.", "x.", "foo.bar.", "a.npy."):
        with pytest.raises(StoreKeyError, match=re.escape(CLAUSE_TRAILING)):
            validate_store_key(refused, _CACHE_DIR)
        assert not is_valid_store_key(refused), f"the predicate accepted the trailing-dot key {refused!r}"


def test_widening_does_not_change_the_clause_reported_by_an_already_refused_key() -> None:
    r"""Plan 14-09 / STORE-01 / D-20: evaluation order is preserved for every pre-existing refusal.

    The trailing test is placed **last**, after the separator test, and that
    position is load-bearing rather than tidy. ``'a/'`` and ``'a\'`` are already
    refused by the separator clause and existing tests assert *that* clause; a
    trailing test placed earlier would silently repoint them. ``'.'`` and
    ``'..'`` end in dots and would likewise be re-reported under the new clause
    from any position ahead of the reserved test.

    This test pins the three keys that would move if the order were changed,
    independently of the parametrised groups that would also catch it — because
    a reordering is exactly the kind of edit that comes with "and update the
    expected clauses" attached.
    """
    for key, clause in (
        ("a/", CLAUSE_SEPARATOR),
        ("a\\", CLAUSE_SEPARATOR),
        ("..", CLAUSE_RESERVED),
        (".", CLAUSE_RESERVED),
    ):
        with pytest.raises(StoreKeyError, match=re.escape(clause)) as excinfo:
            validate_store_key(key, _CACHE_DIR)
        assert CLAUSE_TRAILING not in str(excinfo.value), (
            f"{key!r} is now reported under the trailing clause; the new test ran ahead of "
            f"the {clause!r} test and changed a pre-existing refusal"
        )


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


# ---------------------------------------------------------------------------
# Group 7 — predicate/validator agreement (D-07, T-14-09)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", [*(key for key, _ in REFUSED_KEYS), *ACCEPTED_KEYS])
def test_predicate_agrees_with_the_raising_validator(key: str) -> None:
    """Plan 14-02 / STORE-01 / D-07: ``is_valid_store_key`` is exactly the negation of "the validator raised".

    Parametrised over the **union** of :data:`REFUSED_KEYS` and
    :data:`ACCEPTED_KEYS` — the whole matrix, not a hand-picked subset, because
    drift between the two functions is precisely what a subset would miss.

    This group exists for a concrete reason. The D-18 migration note ships a
    "scan your cache directories" snippet that **imports** the predicate rather
    than restating the rule as a regex. If the predicate and the validator
    could disagree, that published snippet would tell consumers a different
    story than the library enforces — exactly the doc/code drift class this
    milestone keeps finding. One list drives both assertions, so the snippet
    cannot drift from the rule it imports.
    """
    try:
        validate_store_key(key, _CACHE_DIR)
    except StoreKeyError:
        raised = True
    else:
        raised = False

    assert is_valid_store_key(key) is (not raised), (
        f"is_valid_store_key({key!r}) disagrees with the raising validator "
        f"(validator raised: {raised}); the published predicate and the enforced rule have drifted"
    )


# ---------------------------------------------------------------------------
# Group 8 — refusal message content (D-13, T-14-05)
# ---------------------------------------------------------------------------

#: One representative refused key per clause, so the message assertions cover
#: the whole clause vocabulary without re-running the full matrix.
CLAUSE_REPRESENTATIVES: list[tuple[str, str]] = [
    ("..", CLAUSE_RESERVED),
    ("x\n", CLAUSE_CONTROL),
    ("/etc/passwd", CLAUSE_ABSOLUTE),
    ("../victim", CLAUSE_SEPARATOR),
    # Plan 14-09 / D-20: added so the D-13 three-element assertion covers the
    # whole clause vocabulary rather than the pre-widening subset.
    ("foo.", CLAUSE_TRAILING),
]


@pytest.mark.parametrize(("key", "clause"), CLAUSE_REPRESENTATIVES)
def test_message_contains_the_three_required_elements(key: str, clause: str) -> None:
    """Plan 14-02 / STORE-01 / D-13: every refusal names the key, the cache directory and the clause.

    All three are required content, not decoration: the key so the reader knows
    *what* was refused, the cache directory so a multi-store process knows
    *where*, and the clause so the reader knows *which* part of the rule fired
    without reading the source.
    """
    with pytest.raises(StoreKeyError) as excinfo:
        validate_store_key(key, _CACHE_DIR)
    message = str(excinfo.value)

    assert repr(key) in message, f"the refusal message does not carry repr({key!r}): {message!r}"
    assert str(_CACHE_DIR) in message, f"the refusal message does not name the cache directory: {message!r}"
    assert clause in message, f"the refusal message does not name the expected clause {clause!r}: {message!r}"


def test_message_contains_no_newline_for_a_newline_bearing_key() -> None:
    r"""Plan 14-02 / STORE-01 / D-13 / T-14-05: a newline in the key cannot split the log record.

    This is the load-bearing case in the group rather than a nicety. A newline
    is one of the characters the rule refuses, and interpolating such a key raw
    into the message would let the refused key forge a second log line — which
    would make this containment fix's own refusal message a log-injection
    vector. ``repr()`` is what prevents it, and the single-line assertion is
    what proves ``repr()`` is still there.
    """
    key = "x\n"
    with pytest.raises(StoreKeyError, match=re.escape(CLAUSE_CONTROL)) as excinfo:
        validate_store_key(key, _CACHE_DIR)
    message = str(excinfo.value)

    assert "\n" not in message, f"a newline-bearing key split the refusal message across lines: {message!r}"
    assert "\\n" in message, f"the newline was not rendered in its escaped two-character form: {message!r}"


def test_message_contains_a_visible_rendering_of_a_nul_bearing_key() -> None:
    r"""Plan 14-02 / STORE-01 / D-13: a NUL in the key renders visibly rather than invisibly.

    A raw NUL interpolated into a message is invisible in most terminals and in
    most log viewers, so the reader would see ``'xy'`` and be unable to tell why
    it was refused. ``repr()`` renders it as ``\x00`` — exactly when the reader
    most needs to see it.
    """
    key = "x\x00y"
    with pytest.raises(StoreKeyError, match=re.escape(CLAUSE_CONTROL)) as excinfo:
        validate_store_key(key, _CACHE_DIR)
    message = str(excinfo.value)

    assert "\\x00" in message, f"the NUL was not rendered visibly in the refusal message: {message!r}"
    assert "\x00" not in message, f"the refusal message carries a raw NUL byte: {message!r}"


# ---------------------------------------------------------------------------
# Group 9 — the exception hierarchy (D-12)
# ---------------------------------------------------------------------------


def test_exception_hierarchy_is_value_error_and_deliberately_not_key_error() -> None:
    """Plan 14-02 / STORE-01 / D-12: ``StoreKeyError`` is a ``ValueError`` and is **not** a ``KeyError``.

    The negative assertion is the one that matters. ``add_data_to_store``
    already raises ``KeyError`` for "key exists", so a key-shaped ``KeyError``
    from validation would be indistinguishable from a duplicate insert at the
    call site — and the two call for opposite responses.

    ``ValueError`` is the chosen base because every existing ``except
    ValueError`` in pc2img's callers keeps catching it, while iof3D gains the
    precise ``except`` it needs to fail *one tile* without swallowing every
    numeric ``ValueError`` raised inside that tile's computation.
    ``StoreContainmentError`` subclasses ``StoreKeyError`` so a broad
    ``except StoreKeyError`` still catches both, while a per-tile handler
    written against the narrow type cannot silently swallow evidence that
    something was planted in the cache directory.
    """
    assert issubclass(StoreKeyError, ValueError), "StoreKeyError stopped being a ValueError; existing callers break"
    assert issubclass(StoreContainmentError, StoreKeyError), (
        "StoreContainmentError is no longer a StoreKeyError; a broad `except StoreKeyError` stops catching it"
    )
    assert not issubclass(StoreKeyError, KeyError), (
        "StoreKeyError became a KeyError and is now indistinguishable from add_data_to_store's "
        "pre-existing 'key exists' error"
    )


# ---------------------------------------------------------------------------
# Group 10 — the type axis (Plan 14-09 / WR-02, T-14-33)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", NON_STR_KEYS)
def test_predicate_is_total_and_returns_false_for_a_non_str(value: object) -> None:
    """Plan 14-09 / STORE-01 / WR-02: the published predicate returns ``bool`` for *every* argument.

    ``is_valid_store_key`` documents ``Returns: bool`` and the contract page
    sells it as the supported way to check a composed name *without catching an
    exception*. Before this plan a non-``str`` escaped as a bare
    :exc:`TypeError` out of the control-character scan — so the one published
    call shape the predicate exists for could crash a downstream's per-tile
    pre-check, and neither ``except StoreKeyError`` nor ``except ValueError``
    caught it.

    The ``Path`` case is the one that bites in practice: a consumer pre-checking
    a path-typed identifier got a crash for exactly the escape shape this phase
    is about.
    """
    # The annotation says ``str``; passing something else is the whole point of
    # this test, because the real downstream case is a caller who ignored it
    # (a ``Path``-typed identifier fed straight into the pre-check). The cast
    # keeps ``mypy --strict`` honest about the deviation instead of widening the
    # published annotation, which is surface and did not change.
    assert is_valid_store_key(cast(str, value)) is False, (
        f"is_valid_store_key({value!r}) did not return False; the predicate is not total over its input domain"
    )


@pytest.mark.parametrize("value", NON_STR_KEYS)
def test_validator_refuses_a_non_str_with_a_message_naming_the_received_type(value: object) -> None:
    """Plan 14-09 / STORE-01 / WR-02: the raising validator refuses a non-``str`` as ``StoreKeyError``.

    The type of the exception is the assertion that matters: ``store[Path('../victim')] = v``
    must refuse with the *documented* type — a :exc:`StoreKeyError`, therefore a
    :class:`ValueError` — rather than an undocumented :exc:`TypeError` that no
    caller was told to expect.

    The message must name the received type, because the caller's mistake is a
    type mistake and ``'..'`` versus ``Path('..')`` is otherwise invisible in a
    log. It carries a ``repr`` of the value for the same reason every other
    refusal does (D-13).
    """
    expected_type_name = type(value).__name__
    with pytest.raises(StoreKeyError) as excinfo:
        validate_store_key(cast(str, value), _CACHE_DIR)  # see the cast note above
    message = str(excinfo.value)

    assert expected_type_name in message, (
        f"the refusal message does not name the received type {expected_type_name!r}: {message!r}"
    )
    assert repr(value) in message, f"the refusal message does not carry repr({value!r}): {message!r}"
    assert not isinstance(excinfo.value, StoreContainmentError), (
        "a non-str key reported as a containment failure; it is a key-shape refusal, not environment evidence"
    )
