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
    "get_memmap_path",
    "get_memmap_tmp_path",
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
#: Added by D-20. It earns its own constant rather than joining
#: ``CLAUSE_RESERVED`` because the *reason* is different in kind from every
#: other clause: the others refuse **escape**, this one refuses **collision**,
#: and a reader grepping refusal logs needs to be able to tell the two apart.
CLAUSE_TRAILING: Final[str] = "ends in a space or a dot"

_RESERVED_KEYS: Final[frozenset[str]] = frozenset({"", ".", ".."})
_SEPARATORS: Final[tuple[str, str]] = ("/", "\\")
_DEL_CHAR: Final[str] = "\x7f"

#: The reserved Win32 device names (D-20, widened by D-23), held in a single
#: case and compared against the upper-cased **pre-dot stem** of the key. Held
#: upper-case so the comparison is one ``.upper()`` on the key rather than a scan
#: of the set.
#:
#: ``.upper()`` is a case operation over the exact characters supplied, not a
#: fold to ASCII, so it does not breach the no-normalisation rule: a fullwidth
#: ``ＣＯＮ`` upper-cases to fullwidth ``ＣＯＮ``, stays out of this set, and is
#: accepted — which is correct, because on Win32 it is an ordinary filename and
#: not the console device.
#:
#: **D-23's additions, and why each is here.** The digit comprehensions run from
#: ``0`` rather than ``1``, and the superscript forms ``COM¹ COM² COM³`` /
#: ``LPT¹ LPT² LPT³`` are present, because Microsoft's *Naming Files, Paths, and
#: Namespaces* reserved list — the list this comment already appeals to —
#: includes them. ``CONIN$`` and ``CONOUT$`` are on that list as reserved console
#: names. Round 2 shipped the set without any of them under a residual paragraph
#: that presented the device axis as closed; a public grammar that overstates its
#: own coverage is the sanitise-rather-than-reject failure this phase set out to
#: avoid, one level up.
#:
#: The superscript entries are **exact characters, not a case fold**: ``'¹'``
#: upper-cases to ``'¹'``, so they extend the set without normalising anything,
#: and any non-ASCII spelling of the same name — fullwidth ``ＣＯＮ``, for
#: instance — stays outside the set and stays accepted. That is correct rather
#: than an oversight, because on Win32 such a name is an ordinary filename, and
#: it is what keeps this widening inside D-05.
_WIN_DEVICE_NAMES: Final[frozenset[str]] = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"COM{digit}" for digit in range(0, 10)}
    | {f"LPT{digit}" for digit in range(0, 10)}
    | {f"COM{superscript}" for superscript in "¹²³"}
    | {f"LPT{superscript}" for superscript in "¹²³"}
)

#: The characters stripped from the **pre-dot stem** before the reserved-device
#: membership test (D-23) — ASCII space only. See :func:`_device_stem` for the
#: **two** constraints that scope it — the dot condition and the trailing-run
#: boundary (D-30) — and for why it is not ``_TRAILING_CHARS``.
_DEVICE_STEM_STRIP: Final[str] = " "

#: The characters Win32 strips from the end of a filename (D-20). ASCII space
#: and ASCII full stop **only** — widening this to Unicode whitespace or to the
#: fullwidth full stop would start folding, and the fullwidth-dots acceptance
#: test is the standing proof that the rule folds nothing.
_TRAILING_CHARS: Final[str] = " ."


def _device_stem(key: str) -> str:
    """Return the pre-dot stem of ``key`` used to decide the reserved-device test.

    Everything before the first ``'.'``, right-stripped of
    :data:`_DEVICE_STEM_STRIP` — **but only when ``key`` actually contains a
    ``'.'``, and only as far back as the start of the key's own trailing run**
    (D-30). Both conditions are on the key, not on whether the strip would
    change anything, because they are rules rather than optimisations.

    **Why the strip exists.** Win32 strips trailing spaces from the name
    component *before* device resolution, so ``'CON .txt'`` resolves to the
    console while its unstripped stem ``'CON '`` upper-cases outside
    :data:`_WIN_DEVICE_NAMES` and the key itself ends in ``'t'``. It therefore
    slipped past both the device test and the trailing test (§ WR-04).

    **Why the strip is conditional, which is the part that looks like an
    accident and is not.** :data:`_DEVICE_STEM_STRIP` and
    :data:`_TRAILING_CHARS` look alike and are not alike. The trailing clause
    (``CLAUSE_TRAILING``) already owns the key's **own** trailing run; this
    strip exists solely to reach spaces *interior* to the key — the ones
    sitting between a device stem and its extension. The division is general:
    **interior spaces are the device clause's business, trailing spaces are the
    trailing clause's.** An *unconditional* strip lets the device clause reach
    into the trailing clause's territory and repoints
    ``'com1 '``, ``'con '`` and ``'nul  '`` from ``CLAUSE_TRAILING`` onto
    ``CLAUSE_RESERVED`` — a repoint of pre-existing refusals, introduced by the
    very change (D-23) that D-24 exists to keep from repointing anything.

    **Two constraints, not one, and the rule above is why both are needed
    (D-30, § WR-01).** The division-of-territory rule in the previous paragraph
    is quoted verbatim from round 3 and is **unchanged**: it was correct then and
    it is correct now. What was wrong was the implementation, which honoured only
    half of it, and the sentence that followed — *"No clause position fixes
    that; only the conditional does"* — which told the next reader the axis was
    closed. It is not, and it is withdrawn:

    * The **dot condition** closes the *dotless* half. A key with no ``'.'`` at
      all has no stem/extension division for the strip to reach into, so the
      strip must not fire; that is what keeps ``'com1 '``, ``'con '`` and
      ``'nul  '`` on the trailing clause.
    * The **trailing-run boundary** closes the *crossover* half, which the dot
      condition cannot see. ``'con .'`` contains a dot, so the condition lets the
      strip fire; the pre-dot stem is ``'con '``; ``rstrip(' ')`` turns it into
      ``'con'`` and clause 5 matches. Four already-refused keys — ``'con .'``,
      ``'nul .'``, ``'CON .'``, ``'CON . '`` — repointed from ``CLAUSE_TRAILING``
      onto ``CLAUSE_RESERVED`` that way. The strip is now allowed to consume only
      characters lying *strictly before* the index at which the key's own
      trailing run begins, so it can never reach a character the trailing clause
      owns.

    **The two are not independent, and this was measured rather than assumed.**
    Removing the boundary restores all four crossover repoints — it is
    load-bearing. Removing the dot *condition* while keeping the boundary
    changes **nothing**: the whole suite stays green, and an exhaustive check
    over every key of length ≤ 4 drawn from an alphabet of device letters,
    digits, spaces, dots, a tab and a superscript found zero disagreements
    between the two forms. The reason is structural rather than accidental.
    For a key with no ``'.'`` the stem *is* the whole key, so
    ``key[:own_trailing_run_start]`` ends — by the definition of that index — in
    a character outside :data:`_TRAILING_CHARS`, and :data:`_DEVICE_STEM_STRIP`
    is a subset of :data:`_TRAILING_CHARS`; the strip is therefore the identity
    on it. The boundary subsumes the condition.

    The condition is nevertheless **kept**, deliberately and with its redundancy
    recorded rather than left for a reader to rediscover. It states the *rule*
    that a key with no stem/extension division has nothing interior for the
    strip to reach, which is why it was introduced; the boundary states a
    different rule that happens to imply it. Removing it is a behaviour-neutral
    simplification, which makes it a change for a plan that can prove that claim
    on its own terms, not a side effect of this one.

    **What the strip covers, and what it does not.** It reaches ASCII spaces that
    are interior to the key and lie before the key's own trailing run — which is
    exactly the ``'CON .txt'`` shape D-23 was widened for. It does not reach the
    trailing run itself, and it does not model any other Win32 name-resolution
    behaviour; see :func:`validate_store_key`'s residual paragraph, where the
    device axis is recorded as narrowed rather than closed.

    **The residual attribution nuance, recorded rather than left to be
    rediscovered.** ``'com1 '`` is consequently refused as *"ends in a space or
    a dot"* even though Win32's reason to collide on it is the device name. It
    is refused **either way**, so the collision axis stays closed and no key
    escapes; only the reported clause differs. D-24's stability-of-attribution
    promise wins that tie, because BC-GSEG-006 publishes the clause strings as a
    grep target and downstream log-greppers depend on them not moving.

    The returned stem decides **membership and nothing else**. It is never used
    to build a path: :func:`_build` joins the suffix onto the key exactly as the
    caller supplied it, so the rule stays decidable over those exact characters
    (D-05).

    Parameters
    ----------
    key : str
        The candidate store key, exactly as the caller supplied it.

    Returns
    -------
    str
        The stem to test for membership of :data:`_WIN_DEVICE_NAMES`.
    """
    stem = key.split(".", 1)[0]
    if "." not in key:
        return stem
    # The index at which the key's **own** trailing run begins -- the third
    # member of the _DEVICE_STEM_STRIP / _TRAILING_CHARS family, named rather
    # than inlined because it is the concept the division-of-territory rule
    # above turns on. Everything from here to the end of the key belongs to
    # ``CLAUSE_TRAILING``; the strip may consume only what lies before it.
    own_trailing_run_start = len(key.rstrip(_TRAILING_CHARS))
    return stem[:own_trailing_run_start].rstrip(_DEVICE_STEM_STRIP) + stem[own_trailing_run_start:]


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
        "a store key must be a single path segment — no '/', '\\' or ':', not an absolute or "
        "drive-relative path, not '', '.', '..' or a reserved Win32 device name such as CON or "
        "COM1, free of control characters, and not ending in a space or a dot."
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

    1. ``CLAUSE_RESERVED`` — **the exact reserved keys, first.** The key is
       ``''``, ``'.'`` or ``'..'``. This is one *half* of ``CLAUSE_RESERVED``:
       the reserved **device-name** half is clause 5 below, held at a different
       position on purpose (D-24). A reader who greps for one occurrence of
       ``CLAUSE_RESERVED`` and assumes it is the whole clause will get the
       evaluation order wrong.
    2. ``CLAUSE_CONTROL`` — the key contains a character below ``\x20``, or
       ``\x7f``. This covers newline and NUL.
    3. ``CLAUSE_ABSOLUTE`` — the key has a non-empty ``anchor`` or ``drive``
       under *either* :class:`~pathlib.PurePosixPath` or
       :class:`~pathlib.PureWindowsPath`, **or** it contains a bare ``':'``
       (D-20). This refuses ``/etc/passwd``, ``C:evil``, ``C:/abs``,
       ``\\server\share\x`` and ``ab:cd``. The colon needs its own test rather
       than a tighter ``drive`` test because :class:`~pathlib.PureWindowsPath`
       detects only *single-letter* drives — ``PureWindowsPath('ab:cd').drive``
       is ``''`` — so the structural test can never reach an NTFS
       alternate-data-stream name, however it is tightened.
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

    5. ``CLAUSE_RESERVED`` **again — the reserved-device-name stem test.** The
       key's pre-dot stem upper-cases to a member of :data:`_WIN_DEVICE_NAMES`
       (D-20, widened by D-23). The stem test is what refuses ``con.npy`` and
       ``COM1.dat`` as well as bare ``CON``: on Win32 a device name is still the
       device with a suffix attached. The stem is right-stripped of ASCII spaces
       **when the key contains a dot, and only as far back as the key's own
       trailing run** (D-30), so ``CON .txt`` is refused here while ``CON .`` —
       whose trailing run reaches back through the dot to the space — is left to
       clause 6. See :func:`_device_stem` for why both constraints are rules
       rather than optimisations, and why neither is redundant.

       It reports ``CLAUSE_RESERVED`` rather than a clause of its own because
       that wording — "is empty or a reserved path name" — already describes a
       device name exactly, and because a new ``CLAUSE_DEVICE`` constant would
       repoint the clause of **every** key the device set already covers
       (``CON``, ``nul``, ``PRN``, ``AUX``, ``lpt9``, ``con.npy``,
       ``COM1.dat``). That is a second published-vocabulary change in the same
       round, and it is precisely the repointing this relocation exists to
       prevent — applied to keys refused only one round ago. Two positions for
       one constant is the deliberate alternative.
    6. ``CLAUSE_TRAILING`` — the key's trailing run is ASCII spaces or ASCII
       dots (D-20). Evaluated **last**.

    **The rule the last two positions are really asserting, generalised (D-24).**
    The original argument here read as a fact about one clause: the trailing test
    is last because ``'a/'`` and ``'a\'`` are already refused by clause 4 and
    existing tests assert *that* clause, so any earlier placement would silently
    repoint a currently-refused key. That argument is correct and it is a fact
    about a **pair** of late positions, 5 and 6, both held for the same reason:

        *a clause added later must be placed so that no key already refused
        changes the clause it reports* — because BC-GSEG-006 publishes the clause
        strings as a grep target, so a repointed clause makes a correct
        downstream grep start missing keys it used to find.

    Applied here, that rule fixes both boundaries of position 5, and a reader who
    knows only "put new clauses last" gets the second one wrong:

    * The device test runs **after** the separator test (and after control and
      absolute), so ``con.a/b``, ``nul.x\n``, ``aux.C:evil``,
      ``com1.\\server\share\x`` and ``lpt1.a/`` keep reporting the clauses they
      reported before the device half existed.
    * It runs **before** the trailing test, so a device name that also ends in a
      dot — ``con.``, ``com1. `` — keeps reporting ``CLAUSE_RESERVED``. Moving
      the device test to genuinely last repoints those two the other way.

    ↻ **CORRECTED by Plan 14-13 (D-24, § WR-05).** The previous round widened
    clause 1 while it sat **first**, which repointed the five keys named above —
    against this very docstring's promise. This is the correction of that, not a
    note that the repointing was intended. The original clause-5 argument is
    generalised above rather than erased.

    **Leading and interior dots are legal; only a trailing run is refused**
    (D-06 as amended by D-20). ``.hidden``, ``foo.bar``, ``z1.2345678`` and
    ``a.npy`` all still pass — the obvious reading of "dots are now restricted"
    is wrong, and the rule refuses a trailing dot, not a dot. ``foo.`` *did*
    pass before D-20 and no longer does; that is the amendment, not a
    regression.

    Clause 6 has a **wider consequence than the enumerated cases**, which is
    stated here rather than left to be discovered downstream: because it strips
    a trailing *run*, a key consisting entirely of dots or spaces — ``'...'``,
    a lone ``' '`` — is refused too.

    **Two threats, not one, and the second is why the first argument survives
    unchanged.** The paragraph below is the *escape* argument and it is still
    correct. Clauses 3 (colon), 5 (device names) and 6 (trailing run) address a
    **different** threat — *collision*:

    * Win32 strips trailing dots and spaces from a filename, so ``'a'``,
      ``'a '`` and ``'a.'`` become **one file**: two distinct store keys
      silently overwriting one artefact.
    * ``CON``, ``NUL``, ``COM1`` and their siblings name character devices —
      writes are discarded and reads come back empty.
    * A colon opens an NTFS alternate data stream, hiding bytes outside the
      visible artefact.

    None of those is an escape; every one of them lands lexically *inside* the
    cache directory, so :func:`_assert_contained` cannot see any of them and the
    lexical rule is the only layer that can refuse them. They are the
    sanitise-rather-than-reject failure this phase set out to avoid, arriving
    from the collision direction instead of the traversal direction.

    **The residual, stated so it is accepted rather than assumed closed.**
    ↻ **CORRECTED by Plan 14-13 (D-23, § WR-04).** This paragraph previously
    named only case-folding and Unicode normalisation, which left the *device*
    axis reading as closed when it is a fixed list. What the device test actually
    is, stated precisely:

    * It is a **fixed name list** — :data:`_WIN_DEVICE_NAMES` — matched
      case-insensitively against the **pre-dot stem** after a trailing-ASCII-space
      strip (see :func:`_device_stem`). It covers exactly the names enumerated in
      that set: ``CON``, ``PRN``, ``AUX``, ``NUL``, ``CONIN$``, ``CONOUT$``,
      ``COM0``–``COM9``, ``LPT0``–``LPT9`` and the superscript ``COM¹²³`` /
      ``LPT¹²³`` forms.
    * It does **not** attempt to model every Win32 path-parsing behaviour. Any
      device-resolving shape outside that list, or reached by a mechanism other
      than a trailing-space strip on the stem, is not covered.
    * The mechanism claims behind it — that Win32 resolves these names to
      character devices, and that it strips trailing spaces before doing so — are
      **Win32 filesystem behaviour and cannot be confirmed on a Linux host**.
      What this repository's tests confirm is which keys the shipped rule accepts
      and refuses.

    Two residuals are unchanged and remain unfixable by *any* lexical rule.
    Case-insensitive NTFS and default APFS collapse ``'Foo'`` and ``'foo'`` onto
    one file, and APFS normalises Unicode so NFC and NFD spellings of one name
    collide. Both are filesystem behaviour rather than key shape — refusing one
    spelling of a pair would mean choosing a canonical form, which is precisely
    the normalisation D-05 forbids.

    So the collision axis is **narrowed, not closed**, on all three counts.

    The escape argument, unchanged: **the separator is the entire escape
    mechanism, and dots carry no escape risk.**
    The path builders never join the bare key — they concatenate an extension
    first — so ``Path('/cache') / '...npy'`` (key ``'..'``) is a literal file
    *inside* the cache directory and does not escape, while
    ``Path('/cache') / '../victim.npy'`` does, and the separator is what did it.
    **Refusing ``.`` and ``..`` is therefore defence-in-depth against a future
    bare join, not the closing of a live hole** — no path builder does a bare
    join today.

    The empty key has a separate and concrete reason: key ``''`` builds
    ``<cache>/.npy``, a file no key can own. It is refused because it does not
    round-trip, not because it is untidy.

    ↻ **CORRECTED by Plan 14-11.** This paragraph used to say that file's
    *rescan stem* is ``'.npy'`` — *a different, legal-looking key*. That was
    true of the reopen scan until Plan 14-10 (D-22), which replaced
    :attr:`~pathlib.PurePath.stem` with a derivation off the file **name**, so
    the scan now derives ``''`` for it and refuses it. :attr:`.stem` still
    returns ``'.npy'`` for that name; it is simply no longer what the scan
    calls. The **conclusion is unchanged** — ``''`` is refused because it does
    not round-trip — only its stated mechanism moved. Do not restore the old
    wording.

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

    # Clause 1: the exact reserved keys only. The device-name half of this same
    # clause constant is clause 5, below — see the docstring for why the two
    # halves are held at two positions.
    if key in _RESERVED_KEYS:
        _refuse(key, cache_dir, CLAUSE_RESERVED)

    if any(ord(char) < 32 or char == _DEL_CHAR for char in key):
        _refuse(key, cache_dir, CLAUSE_CONTROL)

    posix = PurePosixPath(key)
    windows = PureWindowsPath(key)
    if posix.anchor or posix.drive or windows.anchor or windows.drive or ":" in key:
        _refuse(key, cache_dir, CLAUSE_ABSOLUTE)

    if any(sep in key for sep in _SEPARATORS) or len(posix.parts) != 1 or len(windows.parts) != 1:
        _refuse(key, cache_dir, CLAUSE_SEPARATOR)

    # Clause 5, relocated here by D-24 — after control/absolute/separator so the
    # five keys the round-2 widening repointed report their original clauses
    # again, and *before* the trailing test so ``con.`` and ``com1. `` keep
    # reporting this one. It reuses CLAUSE_RESERVED deliberately: a new
    # CLAUSE_DEVICE constant would repoint every key the device set already
    # covered, which is the same defect this move exists to remove.
    if _device_stem(key).upper() in _WIN_DEVICE_NAMES:
        _refuse(key, cache_dir, CLAUSE_RESERVED)

    # Clause 6, last: see the docstring's placement rule. Compares against the
    # whole key rather than testing emptiness, so ``'a.'`` is refused as well as
    # ``'...'``. The stem strip above must never be folded into this test — this
    # one is about the key's own trailing run.
    if key.rstrip(_TRAILING_CHARS) != key:
        _refuse(key, cache_dir, CLAUSE_TRAILING)


def is_valid_store_key(key: str) -> bool:
    """Return whether ``key`` is a legal single-segment store key.

    The public form of the rule (D-07). It is implemented as a ``try``/``except``
    around :func:`validate_store_key` so the predicate and the validator can
    never disagree — the migration note's cache-directory scan snippet *imports*
    this predicate rather than restating the rule as a regex, which is what makes
    drift between the snippet and the shipped rule impossible.

    Downstreams can also use it to pre-check a composed feature name *before* it
    becomes a key, or to decide per-tile without catching an exception.

    **For a non-**``str`` **argument the predicate always produces a**
    :class:`bool`. Such a value is refused by :func:`validate_store_key`'s own
    type guard as a :exc:`StoreKeyError`, which this function already catches, so
    it cannot escape as an exception — there is deliberately no second guard
    here. That is what lets a downstream pre-check a path-typed identifier
    without wrapping the call, which is the shape the "decide per-tile without
    catching an exception" guidance above was written for.

    ↻ **NARROWED by Plan 14-19 (D-33, § IN-01).** This paragraph previously
    claimed the predicate produced a ``bool`` for *any* argument at all. It does
    not, and the counter-example is a shape this module reasons about one file
    over: a :class:`str` **subclass** whose ``__hash__`` raises. Such a value
    passes the type guard — it *is* a ``str`` — and then reaches
    ``key in _RESERVED_KEYS``, which hashes it; the exception propagates. The
    guard's position ahead of that membership test is still load-bearing for an
    argument that is both *unhashable* and not a ``str`` — the ``bytearray``
    case Plan 14-09 placed it for — but it has no bearing on this one, because a
    ``str`` subclass passes the guard. Measured at
    :func:`validate_store_key`'s reserved-key membership test.

    The behaviour is deliberately **unchanged** — only the claim moved. Turning
    that exception into a ``False`` would require a blanket
    ``except Exception``, which would also swallow
    :exc:`StoreContainmentError`, the signal the ``get`` and ``pop`` accessors
    re-raise on purpose so that evidence of something planted in the cache
    directory stays visible. A consumer composing keys from third-party objects
    with hostile dunders is the one case that needs a ``try``; a consumer passing
    ordinary values, of any type, is not.

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
# That split has already produced one shipped defect: the finalizer helper
# hardcoded ".npy"/".meta.json" and asserted a FRAG-03 intent that no route in
# the package could reach. The constants live here so there is
# one vocabulary and one seam.
#
# DISCHARGED — plan 15-04, STORE-06. This block used to end "BOUNDARY, and the
# owner has now arrived", pointing forward at a fix Phase 15 owed. The fix has
# landed, so the pointer is replaced by its outcome and the two claims above it
# are corrected in place rather than silently dropped — the pattern D-31 used
# when it extended D-17.
#
# First correction: the hardcoded suffixes are gone. The finalizer callback is
# now ``lazy_disk_cache._purge_memmap_file`` and unlinks exactly one file, the
# entry's own memmap.
#
# Second correction: the FRAG-03 intent not holding was recorded here as the
# defect. It is not — it is the only reason the code was safe. Making that dead
# ``.npy`` branch live was measured to empty the cache directory at entry GC
# and to make the first lazy reload fail with ``KeyError`` (15-CONTEXT D-12),
# so the branch was **deleted rather than completed**. Codec-pair removal is
# owned by ``DiskBackedStore.purge``, at the level where the pair is written.
# The vocabulary move is what made both of those one-liners rather than
# refactors.

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

    **What this check can and cannot discriminate — and where the four rows
    apply.** Those four rows describe :func:`_assert_contained` called
    *directly*, as the tests call it. They do **not** describe live
    discrimination among :func:`_build`'s outputs, and the difference is
    load-bearing enough that reading the matrix as the latter is the mistake
    this paragraph exists to prevent. Once the lexical layer has accepted a key,
    the key carries no separator, so ``(cache_dir / f"{key}{suffix}").parent`` is
    lexically ``cache_dir``; therefore ``candidate.parent.resolve()`` equals
    ``base`` and ``is_relative_to(base)`` is unconditionally ``True``. Verified
    across the whole accepted-key matrix. **For a plain** :class:`str` **key the
    check is unreachable**, and that is what defence-in-depth looks like rather
    than a defect.

    It is here to survive two things the lexical layer does not cover:

    * a **lexical-layer regression** — the layer being deleted, weakened or
      bypassed at one route, which is precisely when the signal is wanted;
    * a :class:`str` **subclass whose** ``__str__`` **differs from its
      characters**, which passes :func:`is_valid_store_key` on its characters
      and then builds a path from something else entirely.

    The second case is not hypothetical and is not left as a claim:
    ``test_builder_containment_a_str_subclass_defeats_the_lexical_layer_and_is_caught_by_containment``
    (Plan 14-08, ``tests/test_store_containment.py``) exercises it with the
    lexical layer **fully intact** and no monkeypatch at all, and its sibling
    ``test_builder_containment_every_builder_refuses_an_escape_with_the_lexical_layer_disabled``
    covers the first case at every registered builder. Deleting the
    :func:`_assert_contained` call from :func:`_build` turns those tests red and
    nothing else — which is the measurement that keeps this paragraph honest.

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


def _assert_write_contained(base_dir: Path, candidate: Path) -> None:
    """Verify that ``candidate`` is safe to **open for writing** under ``base_dir``, or raise.

    The write-flavoured sibling of :func:`_assert_contained` (D-31, STORE-02).
    It performs that function's parent-chain verification — by *calling* it,
    not by restating it — and then adds one thing on top: when the final
    component already exists and is a symlink, the link is resolved and the
    target must also be inside the fully-resolved base.

    **Why there are two functions rather than one, stated as a contrast.**
    :func:`_assert_contained` deliberately leaves the final component
    unresolved (D-17). That is not an oversight: a final-component symlink
    inside the cache directory pointing at a real codec pair outside it is the
    legitimate **adopted entry**, and refusing it would break STORE-03, which
    Phase 14 has already verified against the macOS ``mkdtemp`` and ETH
    ``/scratch`` shapes. Leaving the final component unresolved is therefore
    *correct for a read* and *wrong for a truncating write*: reading through an
    adopted entry returns the caller their own data, while writing through a
    planted one silently overwrites a file the caller never named. So the read
    semantics stay exactly as they were and the extra resolution lives here.
    :func:`_assert_contained` must not be changed to serve this path — that
    would trade one verified requirement (STORE-03) for another (STORE-02).

    **Why the base is a parameter and this function derives none (RV-02).**
    :meth:`LazyDiskCache._init_from_config` produces the memmap path two
    different ways: ``config.cache_path.with_suffix(_MEMMAP_SUFFIX)`` when a
    path is configured, and :func:`tempfile.mkstemp` when one is not. A helper
    that tried to infer its own base would have to guess which branch ran.
    ``config.cache_path`` is also the wrong thing to guess at: it is a
    **file-like path, not a directory**. The only well-defined base on both
    branches is the *derived* path's parent, and only the caller knows it —
    which is why every call site passes ``self._cache_path.parent`` and none
    passes ``config.cache_path``. On the configured branch that parent is the
    store's real cache directory; on the ``mkstemp`` branch it is the **system
    temp directory**, where the check is near-vacuous: it catches a symlink
    swapped in after ``mkstemp`` created the file and nothing else, because
    that branch has no cache-directory root at all. The leak that branch
    creates is deferred item **D-14-01**; the fix belongs to **STORE-08**.

    **The measured mode scope, and the table it rests on.** ``load()`` accepts
    four memmap modes, and callers of this helper guard the ones measured to
    reach the file. Measured on this repository before the guard existed, with
    a 256-byte sentinel outside the cache directory and a ``.dat`` symlink
    planted at it — a fresh entry and a fresh sentinel per row:

    ===== ======== ============ =========== ====================== ==========================
    mode  raised   size before  size after  bytes changed by load  handle mutable after load
    ===== ======== ============ =========== ====================== ==========================
    'r'   no       256          256         no                     no (``ValueError`` on write)
    'r+'  no       256          256         no                     **yes**
    'w+'  no       256          **64**      **yes**                **yes**
    'c'   no       256          256         no                     no (copy-on-write, never
                                                                   reaches disk)
    ===== ======== ============ =========== ====================== ==========================

    So the guard covers ``"r+"`` and ``"w+"``. ``"w+"`` truncates the target
    during :meth:`~LazyDiskCache.load` itself; ``"r+"`` does not write during
    the call but installs a **mutable mapping onto the outside file**, so the
    damage arrives through the handle the caller is then handed — the same
    escape, one statement later. ``"r"`` and ``"c"`` are left unguarded because
    the table shows neither reaches the file, and because forcing this
    write-flavoured final-component resolution onto a read is precisely what
    D-17 and STORE-03 refuse. The scope is decided by that table; a mode later
    measured to mutate the sentinel gets guarded whatever this paragraph says.

    **Accepted residual: a read can still disclose.** ``load(mode="r")``
    through a planted symlink still *reads* a file outside the directory. That
    is disclosure, not corruption, and it is out of scope for a
    containment-on-write decision — closing it is what would break the adopted
    entry.

    **Accepted residual: TOCTOU.** This is a check-then-open sequence, so a
    window remains between the verification here and the ``np.memmap`` call
    that follows: a path verified as an ordinary file can be replaced by a
    symlink before it is opened. The check **narrows** the window; it does not
    close it. What would close it is an open that refuses to follow a
    final-component symlink (``O_NOFOLLOW`` on the final component, or an
    ``openat``-based construction), which :mod:`numpy` does not expose through
    :class:`numpy.memmap`. That is **not this round's work**, and exploiting
    the residual already requires write access to the directory — the same
    precondition as the escape being closed here.

    **BOUNDARY — this is deliberately half a fix.** It adds containment and
    **not** torn-write safety. The temp-file / ``fsync`` / atomic-replace
    machinery :meth:`DiskBackedStore._store_entry` already applies to the codec
    pair does not apply to the memmap artefact, so an interrupted write can
    still leave a partial ``.dat``. That half, and its interaction with the
    purge family, is **STORE-08** (Phase 15), paired with deferred item
    **D-14-01**. A reader who finds a containment check sitting beside an
    unguarded truncating write needs to know the gap is filed rather than
    missed.

    Parameters
    ----------
    base_dir : Path
        The directory the write must stay inside; the authorization boundary.
        Passed explicitly rather than derived — see the reasoning above. It is
        fully resolved here, so a ``base_dir`` that is itself a symlink
        compares equal to itself and every ordinary path inside it is accepted.
    candidate : Path
        The path that is about to be opened for writing. It need **not** exist:
        allocation is the normal case, and a path whose final component is
        absent has no link to resolve, so it is accepted on that ground alone.

    Returns
    -------
    None
        Returns ``None`` when ``candidate`` is safe to open for writing.

    Raises
    ------
    StoreContainmentError
        If ``candidate``'s parent chain leaves ``base_dir`` (delegated to
        :func:`_assert_contained`), or if its final component is a symlink
        resolving outside ``base_dir``. Both report
        :data:`CLAUSE_ESCAPES`, so a consumer greps one clause for both layers.
    """
    # Layer one, delegated rather than restated: the parent-chain semantics
    # cannot drift between the read-flavoured and write-flavoured checks if
    # only one function implements them.
    _assert_contained(base_dir, candidate)

    # Layer two, the only difference: resolve the final component as well.
    # ``is_symlink()`` is False for a path that does not exist, so allocation
    # -- the normal case -- falls straight through.
    if not candidate.is_symlink():
        return

    base = base_dir.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(base):
        raise StoreContainmentError(
            f"Store path {str(candidate)!r} for cache directory {str(base_dir)!r}: the path "
            f"{CLAUSE_ESCAPES} — its final component is a symlink resolving to "
            f"{str(resolved)!r}, which is not inside the resolved directory {str(base)!r}, and "
            "this path is about to be opened for writing. Reading through such a link is the "
            "legitimate adopted-entry case (D-17); writing through it would overwrite or "
            "truncate a file outside the directory, so the write is refused. Inspect the "
            "directory before retrying."
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


def get_memmap_path(cache_dir: Path, key: str) -> Path:
    """Return the on-disk ``<key>.dat`` memmap path inside ``cache_dir``.

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
        The validated, contained memmap path.

    Raises
    ------
    StoreKeyError
        If ``key`` is not a legal single-segment store key.
    StoreContainmentError
        If the path would resolve outside ``cache_dir``.
    """
    return _build(cache_dir, key, MEMMAP_SUFFIX)


def get_memmap_tmp_path(cache_dir: Path, key: str) -> Path:
    """Return the temporary ``<key>.dat.tmp`` write path inside ``cache_dir``.

    Internal construction detail with no downstream caller, and deliberately
    absent from the package's published surface.

    The name exists because ``<key>.dat.tmp`` is a real artefact with a real
    lifetime — it persists from creation until its rename, and a crash in
    between leaves one indefinitely — so a removal verb that did not know the
    name would leak the very file the atomicity fix creates (D-14).

    Parameters
    ----------
    cache_dir : Path
        The configured cache directory.
    key : str
        The store key.

    Returns
    -------
    Path
        The validated, contained temporary memmap path.

    Raises
    ------
    StoreKeyError
        If ``key`` is not a legal single-segment store key.
    StoreContainmentError
        If the path would resolve outside ``cache_dir``.
    """
    return _build(cache_dir, key, f"{MEMMAP_SUFFIX}{TMP_SUFFIX}")


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
    "get_memmap_path": get_memmap_path,
    "get_memmap_tmp_path": get_memmap_tmp_path,
}
