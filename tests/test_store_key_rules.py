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

import ast
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

import pytest

from GSEGUtils.lazy_disk_cache.paths import (
    _WIN_DEVICE_NAMES,
    CLAUSE_ABSOLUTE,
    CLAUSE_CONTROL,
    CLAUSE_RESERVED,
    CLAUSE_SEPARATOR,
    CLAUSE_TRAILING,
    StoreContainmentError,
    StoreKeyError,
    _device_stem,
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
#:
#: ↻ EXTENDED by Plan 14-19 (D-30, § WR-01) with the **crossover family**, and
#: the reason it was missing is the finding. The two families above are the two
#: **pure** directions — a device stem followed by a trailing run *and no dot*,
#: and a device stem whose trailing run *is* the dot. Neither is a key carrying a
#: device stem, an interior space **and** a dot, which is where the two clauses'
#: territories actually meet. Round 3's dot-gated strip closed the dotless half
#: only; ``'con .'`` contains a dot, so the strip fired, ``rstrip(' ')`` turned
#: the pre-dot stem ``'con '`` into ``'con'``, and four already-refused keys
#: repointed from the trailing clause onto the reserved one with nothing going
#: red — because no key of this shape was pinned anywhere. The combination is
#: pinned here now, so the boundary is a named case rather than an
#: unrepresented one.
DEVICE_STEM_BOUNDARY_KEYS: list[tuple[str, str]] = [
    ("com1 ", CLAUSE_TRAILING),
    ("con ", CLAUSE_TRAILING),
    ("nul  ", CLAUSE_TRAILING),
    ("con.", CLAUSE_RESERVED),
    ("com1. ", CLAUSE_RESERVED),
    # The crossover family (Plan 14-19 / D-30): a device stem, an interior
    # space and a dot, all in one key. The key's own trailing run reaches back
    # through the dot to the space, so the space belongs to the trailing clause
    # and the device strip must not consume it.
    ("con .", CLAUSE_TRAILING),
    ("CON .", CLAUSE_TRAILING),
    ("CON . ", CLAUSE_TRAILING),
    ("nul .", CLAUSE_TRAILING),
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

#: Every clause constant, so a test can assert that a key reports **one** of them
#: rather than only that it reports the expected one.
ALL_CLAUSES: tuple[str, ...] = (
    CLAUSE_RESERVED,
    CLAUSE_CONTROL,
    CLAUSE_ABSOLUTE,
    CLAUSE_SEPARATOR,
    CLAUSE_TRAILING,
)

#: The five keys § WR-05 reproduced as **repointed** by the round-2 device
#: widening, each paired with the clause it reported *before the device half
#: existed* (Plan 14-13 / D-24). D-24 moves the device test after the control,
#: absolute and separator tests so each of them reports its original clause
#: again.
#:
#: ⚠ ``'com1.\\\\server\\share\\x'`` is pinned at :data:`CLAUSE_SEPARATOR`, which
#: **contradicts § WR-05's stated** ``CLAUSE_ABSOLUTE``. Measured, twice, against
#: a scratch build with ``_WIN_DEVICE_NAMES`` emptied: a leading ``'com1.'``
#: prefix means :class:`~pathlib.PureWindowsPath` sees no UNC anchor, so the
#: separator test is what catches it. The measurement is what is pinned here; the
#: review's value is not.
PRE_EXISTING_CLAUSE_KEYS: list[tuple[str, str]] = [
    ("con.a/b", CLAUSE_SEPARATOR),
    ("nul.x\n", CLAUSE_CONTROL),
    ("aux.C:evil", CLAUSE_ABSOLUTE),
    ("com1.\\\\server\\share\\x", CLAUSE_SEPARATOR),
    ("lpt1.a/", CLAUSE_SEPARATOR),
]

#: The **frozen pre-widening snapshot** (Plan 14-13 / D-24): every key in
#: :data:`REFUSED_KEYS` that was *already refused* before the device half
#: existed, paired with the clause it reported then.
#:
#: **Written out as an explicit literal on purpose.** Deriving it from
#: :data:`REFUSED_KEYS` — the very thing it checks — would produce a test that
#: cannot fail. The duplication is the mechanism.
#:
#: **Measured, not copied.** Each value was read off a scratch build of
#: ``paths.py`` with ``_WIN_DEVICE_NAMES`` emptied, which is exactly "before the
#: device half existed". That is deliberately *not* the shipped rule at
#: ``56c8306``: ``56c8306`` is the post-widening, pre-relocation state, so a
#: snapshot taken there would be a guard against *future* clause additions rather
#: than the guard for "every pre-existing refusal" this test's name claims — and
#: it structurally could not see a repoint on a key that was not already in
#: :data:`REFUSED_KEYS`, which is exactly how the ``'com1 '`` family hid.
PRE_WIDENING_CLAUSE: dict[str, str] = {
    "../victim": CLAUSE_SEPARATOR,
    "a/b": CLAUSE_SEPARATOR,
    "tile_03/range": CLAUSE_SEPARATOR,
    "a/": CLAUSE_SEPARATOR,
    "..\\..\\x": CLAUSE_SEPARATOR,
    "/etc/passwd": CLAUSE_ABSOLUTE,
    "C:evil": CLAUSE_ABSOLUTE,
    "C:/abs": CLAUSE_ABSOLUTE,
    "\\\\server\\share\\x": CLAUSE_ABSOLUTE,
    "": CLAUSE_RESERVED,
    ".": CLAUSE_RESERVED,
    "..": CLAUSE_RESERVED,
    "a\\": CLAUSE_SEPARATOR,
    "x\n": CLAUSE_CONTROL,
    "x\x00y": CLAUSE_CONTROL,
    "\uff0e\uff0e/victim": CLAUSE_SEPARATOR,
    "foo.": CLAUSE_TRAILING,
    "a ": CLAUSE_TRAILING,
    "x.": CLAUSE_TRAILING,
    "...": CLAUSE_TRAILING,
    " ": CLAUSE_TRAILING,
    "ab:cd": CLAUSE_ABSOLUTE,
    "com1 ": CLAUSE_TRAILING,
    "con ": CLAUSE_TRAILING,
    "nul  ": CLAUSE_TRAILING,
    "con.": CLAUSE_TRAILING,
    "com1. ": CLAUSE_TRAILING,
    # ↻ EXTENDED by Plan 14-19 (D-30, § WR-01). This table is the guard the
    # docstring above calls the one for *every pre-existing refusal*, and it had
    # the identical hole as DEVICE_STEM_BOUNDARY_KEYS: it pinned the two pure
    # directions and never the crossover. Extending it is what makes that
    # description true. Each value is measured — with the device set emptied and
    # at 56c8306 alike, these four report the trailing clause, so they are
    # pre-existing refusals and not new ones.
    "con .": CLAUSE_TRAILING,
    "CON .": CLAUSE_TRAILING,
    "CON . ": CLAUSE_TRAILING,
    "nul .": CLAUSE_TRAILING,
}

#: The **two** pairs that disagree with the snapshot above, carried as named
#: entries with their reason and the round that moved them rather than quietly
#: excluded (Plan 14-13 / D-24). An exception you can read is a decision; a key
#: silently missing from the comparison is a hole.
#:
#: Adding a third entry here means a clause was repointed, and it requires the
#: same justification these two carry.
ROUND_2_CLAUSE_MOVES: dict[str, tuple[str, str]] = {
    "con.": (
        CLAUSE_RESERVED,
        "moved TRAILING -> RESERVED by D-20 in round 2, when the device half was added ahead of the "
        "trailing test. D-24's chosen placement (device test between the separator and trailing tests) "
        "deliberately preserves that rather than moving it a second time.",
    ),
    "com1. ": (
        CLAUSE_RESERVED,
        "same round-2 move as 'con.': a device stem that also ends in a trailing run. Putting the device "
        "test *after* the trailing test would repoint both back — which is why 'just put the new clause "
        "last' is the wrong reading of D-24.",
    ),
}

#: Keys :data:`REFUSED_KEYS` carries that were **accepted** before the device
#: half existed (Plan 14-13). They are new refusals from D-20's and D-23's
#: widenings, so there is no pre-existing clause for them to preserve — but they
#: are enumerated rather than skipped, so the three tables together partition
#: :data:`REFUSED_KEYS` exactly and a key cannot fall out of the comparison
#: unnoticed.
#: It is exactly :data:`WIN_DEVICE_KEYS` — every one of the seven D-20 names and
#: every one of D-23's eight additions was accepted by the pre-widening rule, and
#: nothing else in :data:`REFUSED_KEYS` was.
NEWLY_REFUSED_BY_THE_DEVICE_WIDENING: tuple[str, ...] = WIN_DEVICE_KEYS


def reported_clause(key: str) -> str | None:
    """Return the ``CLAUSE_*`` constant ``validate_store_key`` reports for ``key``.

    ``None`` when the key is **accepted** — which is a distinct outcome from
    "refused under some other clause" and the tests below need to tell them
    apart, because disabling a sub-clause turns a refusal into an acceptance
    rather than into a different clause.

    The message is split at the fixed generic tail before the clause constants
    are searched for. The tail paraphrases the whole rule (``"… not an absolute
    or drive-relative path … and not ending in a space or a dot"``), and although
    no constant is currently a verbatim substring of it, relying on that would
    make this helper silently wrong the first time the tail is reworded.
    """
    try:
        validate_store_key(key, _CACHE_DIR)
    except StoreKeyError as excinfo:
        head = str(excinfo).split("; a store key must")[0]
        found = [clause for clause in ALL_CLAUSES if clause in head]
        assert len(found) == 1, f"{key!r} reported {len(found)} clauses ({found!r}) in {head!r}"
        return found[0]
    return None


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

    Rows four and five pull the other way and constrain the *placement* rather
    than the strip: ``'con.'`` and ``'com1. '`` are device stems that also end in
    a trailing run, and they must keep reporting the reserved clause. That is
    what forbids the naive "just put the device test last" reading of D-24.

    ↻ EXTENDED by Plan 14-19 (D-30, § WR-01). The group is no longer five rows
    and no longer green-on-arrival: the last four are the **crossover family**
    (``'con .'``, ``'CON .'``, ``'CON . '``, ``'nul .'``), which round 3 repointed
    from the trailing clause onto the reserved one. They were green before D-23
    and red between D-23 and D-30, so they are a driver here rather than a
    control. The two pure families above them remained green throughout, which is
    precisely why nothing caught the repoint.
    """
    with pytest.raises(StoreKeyError, match=re.escape(clause)) as excinfo:
        validate_store_key(key, _CACHE_DIR)
    assert not is_valid_store_key(key), f"the predicate accepted the boundary-family key {key!r}"

    other = CLAUSE_RESERVED if clause == CLAUSE_TRAILING else CLAUSE_TRAILING
    assert other not in str(excinfo.value), (
        f"{key!r} is now reported under {other!r} as well as {clause!r}; the device clause and the "
        "trailing clause have stopped dividing the space cleanly"
    )


@pytest.mark.parametrize(
    "key",
    [
        "con .",
        "CON .",
        "CON . ",
        "nul .",
        "com1 .",
        "con  .",
        "con . ",
        "aux .",
    ],
)
def test_a_device_stem_never_consumes_the_keys_own_trailing_run(key: str) -> None:
    """Plan 14-19 / STORE-01 / D-30 / § WR-01: the stem strip stops at the key's own trailing run.

    Asserted on :func:`~GSEGUtils.lazy_disk_cache.paths._device_stem`'s **own
    return value** rather than through a clause, deliberately. The corpus test
    above tells you *that* a clause moved; this one tells you *why* — the stem
    the membership test is handed still ends in a character belonging to the
    key's trailing run, so it cannot upper-case into
    ``_WIN_DEVICE_NAMES`` however that set is later widened.

    That distinction is the finding. Round 3's docstring claimed *"No clause
    position fixes that; only the conditional does"*, and the conditional closes
    the **dotless** half only. A key carrying a device stem, an interior space
    **and** a dot slips through it: the dot makes the strip fire and the strip
    then reaches into territory the trailing clause owns. Naming the mechanism
    here means a future widening of the device set reddens this test rather than
    silently repointing a family again.
    """
    boundary = len(key.rstrip(" ."))
    stem = _device_stem(key)
    assert stem[boundary:], (
        f"_device_stem({key!r}) returned {stem!r}, which has consumed the key's own trailing run "
        f"(it begins at index {boundary}); the device clause has reached into the trailing clause's "
        "territory and the key will now report the reserved clause instead"
    )
    assert stem.upper() not in _WIN_DEVICE_NAMES, (
        f"_device_stem({key!r}) returned {stem!r}, which upper-cases into the reserved device set; "
        "the membership test will fire and repoint a key that ends in a space or a dot"
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
    r"""Plan 14-09 / D-20, extended by Plan 14-13 / D-24: evaluation order holds for every pre-existing refusal.

    **The name has finally caught up with the parametrisation.** Two groups of
    cases, covering the two clauses this phase added and the two positions they
    are held at.

    *The four original cases* — ``'a/'``, ``'a\'``, ``'..'`` and ``'.'`` — cover
    **clause 6's** placement. The trailing test is placed after the separator
    test, and that position is load-bearing rather than tidy: ``'a/'`` and
    ``'a\'`` are already refused by the separator clause and existing tests
    assert *that* clause, while ``'.'`` and ``'..'`` end in dots and would be
    re-reported under the trailing clause from any position ahead of the reserved
    test.

    *The five cases added by Plan 14-13* cover **clause 1's device half**, which
    the previous round moved without noticing. Clause 5's docstring argued that
    an earlier placement "would silently repoint a currently-refused key onto a
    new clause" — and clause 1 was widened in that same round, sits **first**,
    and did exactly that: ``con.a/b`` reported RESERVED where it had reported
    SEPARATOR, and four more. This test was named and docstringed as covering
    *every* pre-existing refusal while parametrising four keys that exercise
    clause 6's placement only, so it could not detect the widening that actually
    repointed — and it is the test a future reader trusts. D-24 relocates the
    device test after the control, absolute and separator tests; these five rows
    are what hold it there.

    Each expected clause was **measured** against a scratch build with
    ``_WIN_DEVICE_NAMES`` emptied rather than copied from the review — see
    :data:`PRE_EXISTING_CLAUSE_KEYS` for the one value where the two disagree.

    The assertions are symmetric by construction: every row asserts the exact
    clause reported and that **no other** clause constant appears, so a key that
    reported SEPARATOR cannot start mentioning the reserved wording either.
    """
    for key, clause in (
        ("a/", CLAUSE_SEPARATOR),
        ("a\\", CLAUSE_SEPARATOR),
        ("..", CLAUSE_RESERVED),
        (".", CLAUSE_RESERVED),
        *PRE_EXISTING_CLAUSE_KEYS,
    ):
        with pytest.raises(StoreKeyError, match=re.escape(clause)) as excinfo:
            validate_store_key(key, _CACHE_DIR)
        assert reported_clause(key) == clause, (
            f"{key!r} now reports {reported_clause(key)!r} rather than {clause!r}; a clause test ran "
            "ahead of the one that used to catch it and repointed a pre-existing refusal"
        )
        for other in ALL_CLAUSES:
            if other == clause:
                continue
            assert other not in str(excinfo.value), (
                f"{key!r} is reported under {other!r} as well as {clause!r}; the clause vocabulary "
                "BC-GSEG-006 publishes as a grep target has stopped being one-to-one"
            )


def test_every_pre_existing_key_clause_pairing_survives_the_widening() -> None:
    """Plan 14-13 / STORE-01 / D-24: the whole refused matrix is compared against a frozen pre-widening snapshot.

    The mechanical companion to the named test above. That one pins nine keys a
    reader can reason about; this one pins **every** key in
    :data:`REFUSED_KEYS` against :data:`PRE_WIDENING_CLAUSE`, a literal snapshot
    measured against a build with ``_WIN_DEVICE_NAMES`` emptied.

    Three tables partition :data:`REFUSED_KEYS` and the partition itself is
    asserted first, because a key that falls out of all three would be exempt
    from the comparison and nothing else here would notice:

    * :data:`PRE_WIDENING_CLAUSE` — refused before the device half existed, so
      its clause must not have moved;
    * :data:`ROUND_2_CLAUSE_MOVES` — the **two** pairs that did move, in round 2
      rather than this one, each carrying its reason. They are compared against
      the exception value, so this test still fails if either moves *again*;
    * :data:`NEWLY_REFUSED_BY_THE_DEVICE_WIDENING` — accepted before, so there is
      no earlier clause to preserve; they must simply still be refused.

    Why the snapshot is a literal rather than a comprehension over
    :data:`REFUSED_KEYS`: a test that derives its expectations from the thing it
    is checking cannot fail. The duplication is the mechanism, not an oversight,
    and updating it is meant to be a deliberate act.
    """
    partitioned = set(PRE_WIDENING_CLAUSE) | set(NEWLY_REFUSED_BY_THE_DEVICE_WIDENING)
    assert partitioned == set(EXPECTED_CLAUSE), (
        "the snapshot tables no longer partition REFUSED_KEYS; unclassified: "
        f"{sorted(set(EXPECTED_CLAUSE) - partitioned)!r}, unknown: {sorted(partitioned - set(EXPECTED_CLAUSE))!r}"
    )
    assert not set(PRE_WIDENING_CLAUSE) & set(NEWLY_REFUSED_BY_THE_DEVICE_WIDENING), (
        "a key is listed both as refused pre-widening and as newly refused by it"
    )
    assert set(ROUND_2_CLAUSE_MOVES) <= set(PRE_WIDENING_CLAUSE), (
        "an exception was recorded for a key that has no pre-widening clause to be an exception to"
    )

    for key, baseline in PRE_WIDENING_CLAUSE.items():
        moved = ROUND_2_CLAUSE_MOVES.get(key)
        expected = baseline if moved is None else moved[0]
        actual = reported_clause(key)
        assert actual is not None, (
            f"{key!r} is no longer refused at all; a sub-clause of the rule has been removed or disabled"
        )
        if moved is None:
            assert actual == expected, (
                f"{key!r} reported {baseline!r} before the device half existed and now reports "
                f"{actual!r}; BC-GSEG-006 publishes these strings as a grep target, so a downstream "
                f"grepping {baseline!r} has stopped finding it"
            )
        else:
            assert actual == expected, (
                f"{key!r} is a recorded round-2 exception expected to report {expected!r} and now "
                f"reports {actual!r}. {moved[1]}"
            )

    for key in NEWLY_REFUSED_BY_THE_DEVICE_WIDENING:
        assert reported_clause(key) == CLAUSE_RESERVED, (
            f"{key!r} was accepted before the device widening and must now be refused under "
            f"{CLAUSE_RESERVED!r}; it reports {reported_clause(key)!r}"
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
def test_predicate_returns_false_for_a_non_str_argument(value: object) -> None:
    """Plan 14-09 / STORE-01 / WR-02: the published predicate returns ``False`` for a non-``str``.

    ``is_valid_store_key`` documents ``Returns: bool`` and the contract page
    sells it as the supported way to check a composed name *without catching an
    exception*. Before Plan 14-09 a non-``str`` escaped as a bare
    :exc:`TypeError` out of the control-character scan — so the one published
    call shape the predicate exists for could crash a downstream's per-tile
    pre-check, and neither ``except StoreKeyError`` nor ``except ValueError``
    caught it.

    The ``Path`` case is the one that bites in practice: a consumer pre-checking
    a path-typed identifier got a crash for exactly the escape shape this phase
    is about.

    ↻ RENAMED and REWORDED by Plan 14-19 (D-33, § IN-01, cross-AI review RV-05).
    This test was called ``test_predicate_is_total_and_returns_false_for_a_non_str``
    and its docstring and failure message both asserted an **unqualified**
    totality — that the predicate returns a ``bool`` for *any* argument
    whatsoever. That claim is false and is withdrawn this round: a ``str``
    subclass with a raising ``__hash__`` propagates out of the reserved-key
    membership test. Withdrawing the claim in the docstring and on the contract
    page while the suite kept asserting it in its own failure text would be a
    half-correction, so the name, the docstring and the message all move here.
    The scope this test actually covers — non-``str`` arguments — is unchanged
    and is now what its name says.
    """
    # The annotation says ``str``; passing something else is the whole point of
    # this test, because the real downstream case is a caller who ignored it
    # (a ``Path``-typed identifier fed straight into the pre-check). The cast
    # keeps ``mypy --strict`` honest about the deviation instead of widening the
    # published annotation, which is surface and did not change.
    assert is_valid_store_key(cast(str, value)) is False, (
        f"is_valid_store_key({value!r}) did not return False; a non-str argument must be refused as a "
        "verdict rather than escaping as an exception"
    )


def test_is_valid_store_key_is_total_over_non_str_arguments_only() -> None:
    """Plan 14-19 / STORE-01 / D-33 / § IN-01: the predicate's totality covers non-``str`` arguments and no more.

    Two halves, held together in one test on purpose, because the claim and its
    exception are a pair and separating them is how the unqualified version
    survived three rounds.

    **First half — every non-``str`` shape produces a verdict.** That much of the
    old claim is true, is what consumers actually use, and is what
    :func:`test_predicate_returns_false_for_a_non_str_argument` pins per shape.
    It is restated in aggregate here so a reader who greps this test's name finds
    the whole picture in one place.

    **Second half — a ``str`` subclass with a raising** ``__hash__`` **raises,
    and that is asserted as the correct behaviour rather than as a known bug.**
    ``key in _RESERVED_KEYS`` hashes the argument, and it does so *before* the
    non-``str`` type guard's position has any bearing on the outcome — the guard
    is ahead of it, but a ``str`` subclass passes the guard, so the hash is
    reached anyway. Converting that exception into a ``False`` would take a
    blanket ``except Exception``, and a blanket handler here would also swallow
    :exc:`StoreContainmentError` — the signal the ``get`` and ``pop`` carve-outs
    were added in Plans 14-14 and 14-17 specifically to keep visible. A
    contributor who "fixes" the predicate that way reddens this test, and this
    docstring tells them what they would have broken.

    The module already reasons about hostile ``str`` subclasses one file over
    (``_assert_contained``'s case 2 and its ``Sneaky`` shape), so this is not an
    exotic argument for this codebase — it is the shape the containment layer
    exists for, arriving at the lexical layer instead.
    """
    for value in NON_STR_KEYS:
        verdict = is_valid_store_key(cast(str, value))
        assert isinstance(verdict, bool), (
            f"is_valid_store_key({value!r}) returned {verdict!r}, which is not a bool; the non-str "
            "half of the documented claim has stopped holding"
        )

    class RaisingHash(str):
        """A ``str`` subclass whose ``__hash__`` raises — the § IN-01 escape shape."""

        def __hash__(self) -> int:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        is_valid_store_key(RaisingHash("safe"))


def test_an_ordinary_str_subclass_is_decided_by_its_characters() -> None:
    """Plan 14-19 / STORE-01 / D-33: a well-behaved ``str`` subclass is decided like any other key.

    The positive boundary of the test above. The narrowing is about a *hostile*
    dunder, not about subclasses as a category: a subclass that behaves like the
    ``str`` it is gets the verdict its characters say, so the narrowed claim does
    not quietly withdraw support for the ordinary case.
    """

    class Ordinary(str):
        """A ``str`` subclass with no dunder overrides at all."""

    assert is_valid_store_key(cast(str, Ordinary("tile_03"))) is True
    assert is_valid_store_key(cast(str, Ordinary("../victim"))) is False


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


# ---------------------------------------------------------------------------
# Plan 14-15 Task 2: the contract page's key literals, checked against the
# shipped rule rather than against a reviewer's one-off probe
#
# Round 2's reviewer verified every key literal on ``docs/source/LazyDiskCache.rst``
# against ``is_valid_store_key`` by hand. Nothing in the repository did that, so
# D-23's widening could have turned a page example from legal to illegal with a
# green build — the strict docs build only fails in the *other* direction (a
# broken reference), and the docs member allowlist means an omission is silent.
# The probe is shipped here so doc/rule drift becomes a red build.
# ---------------------------------------------------------------------------

#: The published contract page, located exactly as ``tests/test_store_containment.py``
#: locates it, so the two modules cannot end up reading different copies.
CONTRACT_PAGE: Path = Path(__file__).resolve().parents[1] / "docs" / "source" / "LazyDiskCache.rst"

#: The RST comment that opens a key-literal group on the page. The page carries
#: one of these immediately above every construct this module parses, and each
#: names this test — the reverse pointer is what stops a future editor
#: restructuring a table into something the parser cannot see.
_PAGE_MARKER = re.compile(r"^\.\.\s+CONTRACT-PAGE-KEYS:\s*(?P<group>[A-Z0-9-]+)\s*$")

#: Extraction floors, one per partition, measured against the page as Plan 14-15
#: shipped it. They exist because an agreement test whose parser matches nothing
#: passes every assertion it makes — zero of them — and reports success. Round 2's
#: own standard, that a test which cannot fail is not evidence, applies to the
#: parser and not only to the rule.
#:
#: Both values are the counts measured on the shipped page, not a slack margin.
#: The refused floor is deliberately tight enough that *each* marked group is
#: load-bearing: the collision table contributes exactly one literal the refusal
#: table does not carry (``"com1.dat"``), so losing that marker alone drops the
#: count below the floor rather than passing unnoticed.
_MIN_LEGAL_PAGE_LITERALS: int = 6
_MIN_REFUSED_PAGE_LITERALS: int = 30

#: The shapes D-23 added to the device set. The page presented a strict subset of
#: the refused device names before Plan 14-15; ``CON .txt`` is the one a reader is
#: most likely to take for a typo, which is exactly why it has to be on the page.
_WIDENED_DEVICE_SHAPES_THE_PAGE_MUST_SHOW: tuple[str, ...] = (
    "COM0",
    "LPT0",
    "CONIN$",
    "CONOUT$",
    "com¹",
    "CON .txt",
)


def _inline_literals(text: str) -> list[str]:
    """Return the contents of every ``double-backtick`` inline literal in ``text``.

    Deliberately refuses to span a newline: every key literal on the page sits on
    one line, and allowing a match across lines would silently pair the opening
    delimiter of one literal with the closing delimiter of another.
    """
    return re.findall(r"``([^`\n]+)``", text)


def _quoted_keys(text: str) -> list[str]:
    r"""Return the key each *double-quoted* inline literal in ``text`` denotes.

    **The partitioning rule, stated once so the page can be written to it.** A key
    literal on the contract page is written as a double-quoted Python string inside
    an inline literal — ``"foo."``, ``"CON .txt"``, ``"x\ny"``. Anything else in an
    inline literal (a clause name, a character class such as ``\x20``, a bare ``/``,
    a module name) is *not* a key and is skipped. That one rule is what lets this
    parser read prose tables without a hand-maintained list, and it is why a cell
    may mention non-key literals freely.

    The quoted form is evaluated with :func:`ast.literal_eval`, so the page's
    escapes mean what they mean in Python: ``"x\ny"`` is a key containing a real
    newline and ``"\\\\server\\share\\x"`` is a UNC path, rather than each being
    the backslashes it is spelled with.
    """
    keys: list[str] = []
    for raw in _inline_literals(text):
        if len(raw) >= 2 and raw.startswith('"') and raw.endswith('"'):
            keys.append(cast(str, ast.literal_eval(raw)))
    return keys


def _code_block_after(lines: list[str], start: int) -> list[str]:
    """Return the body lines of the first ``code-block`` directive at or after ``start``."""
    index = start
    while index < len(lines) and not lines[index].lstrip().startswith(".. code-block::"):
        index += 1
    if index == len(lines):
        return []
    directive_indent = len(lines[index]) - len(lines[index].lstrip())
    body: list[str] = []
    for line in lines[index + 1 :]:
        if not line.strip():
            continue
        if len(line) - len(line.lstrip()) <= directive_indent:
            break
        body.append(line.strip())
    return body


def _list_table_rows(lines: list[str], start: int) -> list[list[str]]:
    """Parse the first ``list-table`` directive at or after ``start`` into rows of cells.

    A row opens with ``* - `` and each further cell of that row with ``- `` at two
    columns further in; anything else indented under the directive is a
    continuation of the cell currently open. Returned rows include the header row,
    because ``:header-rows: 1`` is the page's convention and dropping it here would
    hide a table that lost its header.
    """
    index = start
    while index < len(lines) and not lines[index].lstrip().startswith(".. list-table::"):
        index += 1
    if index == len(lines):
        return []
    directive_indent = len(lines[index]) - len(lines[index].lstrip())
    rows: list[list[str]] = []
    row_indent: int | None = None
    for line in lines[index + 1 :]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= directive_indent:
            break
        stripped = line.strip()
        if stripped.startswith("* - "):
            row_indent = indent
            rows.append([stripped[4:]])
        elif rows and row_indent is not None and indent == row_indent + 2 and stripped.startswith("- "):
            rows[-1].append(stripped[2:])
        elif rows:
            rows[-1][-1] += " " + stripped
    return rows


def _extract_page_key_literals() -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(legal, refused)``, each mapping a page key literal to its page group.

    The group name is carried through so a failure can name the section of the page
    to edit — whoever trips this test is editing prose, not the rule, and needs the
    line rather than the verdict.
    """
    lines = CONTRACT_PAGE.read_text(encoding="utf-8").splitlines()
    legal: dict[str, str] = {}
    refused: dict[str, str] = {}
    for index, line in enumerate(lines):
        match = _PAGE_MARKER.match(line.strip())
        if match is None:
            continue
        group = match.group("group")
        if group == "LEGAL-CODE-BLOCK":
            for body_line in _code_block_after(lines, index):
                tokens = body_line.split()
                if tokens:
                    legal.setdefault(tokens[0], group)
        elif group.startswith("REFUSED-TABLE-COLUMN-"):
            column = int(group.rsplit("-", 1)[1]) - 1
            for row in _list_table_rows(lines, index)[1:]:
                if column < len(row):
                    for key in _quoted_keys(row[column]):
                        refused.setdefault(key, group)
    return legal, refused


def test_contract_page_key_literals_agree_with_the_shipped_rule() -> None:
    """Plan 14-15 / STORE-01 / T-14-67: the published grammar agrees with the shipped one.

    Every key literal the contract page presents as legal is accepted by
    :func:`is_valid_store_key`, and every literal it presents as refused is refused.
    The page is the document a consumer composes keys against; a page that
    advertises a legal key the rule refuses is the same drift the importing scan
    snippet was written to prevent, moved from the rule to the examples.

    The two floor assertions come first on purpose. They are assertions about the
    *parser*, not about the page: a regex that silently matches nothing would make
    every agreement assertion below vacuous and this test green.
    """
    legal, refused = _extract_page_key_literals()

    assert len(legal) >= _MIN_LEGAL_PAGE_LITERALS, (
        f"extracted only {len(legal)} 'legal' key literals from {CONTRACT_PAGE.name} "
        f"(floor {_MIN_LEGAL_PAGE_LITERALS}) — the parser has stopped seeing the page. "
        f"Check the '.. CONTRACT-PAGE-KEYS: LEGAL-CODE-BLOCK' marker and the code block "
        f"below it. Extracted: {sorted(legal)}"
    )
    assert len(refused) >= _MIN_REFUSED_PAGE_LITERALS, (
        f"extracted only {len(refused)} 'refused' key literals from {CONTRACT_PAGE.name} "
        f"(floor {_MIN_REFUSED_PAGE_LITERALS}) — the parser has stopped seeing the page. "
        f"Check the '.. CONTRACT-PAGE-KEYS: REFUSED-TABLE-COLUMN-*' markers and that every "
        f"key literal in those columns is written as a double-quoted string. "
        f"Extracted: {sorted(refused)}"
    )

    for key, group in sorted(legal.items()):
        assert is_valid_store_key(key), (
            f"{CONTRACT_PAGE.name} presents {key!r} as a legal key (page group {group}), "
            f"but the shipped rule refuses it — the page advertises a key the library "
            f"will not accept"
        )
    for key, group in sorted(refused.items()):
        assert not is_valid_store_key(key), (
            f"{CONTRACT_PAGE.name} presents {key!r} as a refused key (page group {group}), "
            f"but the shipped rule accepts it — the page documents a refusal the library "
            f"does not perform"
        )


def test_contract_page_presents_every_shape_the_widened_device_rule_refuses() -> None:
    """Plan 14-15 / STORE-01 / § WR-04: the page's device examples are not a strict subset.

    Agreement alone cannot catch this. Every device example the page carried before
    Plan 14-15 was *correct*; the defect was that it was **short**, and a reader
    composing ``CONIN$`` or ``CON .txt`` would have read the page as permission.
    An omission is invisible to an agreement check, so it needs its own assertion.
    """
    _, refused = _extract_page_key_literals()
    missing = [key for key in _WIDENED_DEVICE_SHAPES_THE_PAGE_MUST_SHOW if key not in refused]
    assert not missing, (
        f"{CONTRACT_PAGE.name} does not present these refused device shapes: {missing!r}. "
        f"They are refused by the shipped rule (D-23) and the page's device row is the "
        f"place a consumer looks before composing a key."
    )
