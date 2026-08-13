Lazy Disk Cache
===============

.. _StoreKeyContract:

Store key contract
------------------

A :class:`~GSEGUtils.lazy_disk_cache.DiskBackedStore` key becomes a **filename**
inside the configured cache directory. Since 0.5.x the store validates every key
before it builds a path from it, at every route that reads, writes or restores an
entry, and every path it builds is additionally checked to land inside the cache
directory.

This page is the contract. If you compose store keys programmatically — feature
names, tile identifiers, anything assembled from user or configuration input —
read the *What is refused* table and then run the scan below against your
existing cache directories.

What a key may be
~~~~~~~~~~~~~~~~~

**A key is a single path segment.** Within that, the character set is deliberately
wide open. The rule is a *property denylist*, never an allowlist charset,
precisely so that a composed name does not become a breaking change the next time
the composition grows a new token.

All of these are legal, and all of them round-trip through the on-disk name
exactly — with one filesystem-level caveat noted under *Where the exact
round-trip stops* below:

.. CONTRACT-PAGE-KEYS: LEGAL-CODE-BLOCK
   The first whitespace-delimited token of every line in the block below is
   asserted to be *accepted* by the shipped rule, in
   ``tests/test_store_key_rules.py::test_contract_page_key_literals_agree_with_the_shipped_rule``.
   Keep one key per line with its commentary after it, and keep this marker
   directly above the directive.

.. code-block:: text

   rrim_pack_(range,r16,d8,z1e-05)     parentheses, commas, hyphens, underscores
   rrim_component_(structure,range)    a composed feature name
   z1.2345678                          interior dots
   foo.bar                             interior dots
   .hidden                             a LEADING dot
   a.npy                               a key that looks like a filename

Dots are legal, **including leading dots and interior dots**. A key may not be
the exact string ``.`` or ``..``, and it may not *end* in a dot — those are two
separate rules with two separate reasons, and neither is the reason most readers
assume. See *Why the reserved names are refused* and *Why collision shapes are
refused* below.

What is refused
~~~~~~~~~~~~~~~

Clauses are evaluated in a fixed order, so a key violating several always reports
the same one. The clause wording below is the wording the exception message
carries, so you can grep your logs for it. **The table is in evaluation order**,
and one clause string is produced at *two* of those positions — the reserved-name
clause covers the exact strings ``''``/``'.'``/``'..'`` first and the Win32 device
names fifth. That is deliberate rather than untidy: the device test was added
after the others, and putting it last-but-one is what keeps every key that was
already refused for another reason reporting the clause it always reported.

.. CONTRACT-PAGE-KEYS: REFUSED-TABLE-COLUMN-3
   Every double-quoted literal in the *Examples* column below is asserted to be
   refused by the shipped rule, in
   ``tests/test_store_key_rules.py::test_contract_page_key_literals_agree_with_the_shipped_rule``.
   Keep key literals double-quoted and on one line, and keep this marker directly
   above the directive, or that test stops seeing this table and fails on its floor.
   Each example is **also** asserted to report the clause named in its own row's
   first column, in
   ``tests/test_store_key_rules.py::test_contract_page_refusal_table_agrees_on_the_clause_not_only_the_verdict``
   (Plan 14-19). Keep the clause in column 1 written as a double-backtick inline
   literal, exactly as the message spells it, or that test loses the pairing.

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - Clause (as it appears in the message)
     - Refuses
     - Examples
   * - ``is empty or a reserved path name`` — first position
     - the exact strings ``''``, ``'.'`` and ``'..'``
     - ``""``, ``"."``, ``".."``
   * - ``contains a control character``
     - any character below ``\x20``, and ``\x7f``
     - ``"x\ny"``, ``"x\x00y"``
   * - ``is an absolute or drive-relative path``
     - a non-empty anchor or drive under *either* POSIX or Windows semantics,
       **and** a bare ``:`` anywhere in the key
     - ``"/etc/passwd"``, ``"C:evil"``, ``"\\\\server\\share\\x"``, ``"ab:cd"``
   * - ``contains a path separator``
     - ``/`` or ``\`` anywhere, including a *trailing* separator, and anything
       ``pathlib`` reads as multi-segment under either flavour
     - ``"../victim"``, ``"tile_03/range"``, ``"a/"``, ``"..\\..\\x"``
   * - ``is empty or a reserved path name`` — **same clause string, evaluated
       here**
     - any key whose pre-dot stem is a Win32 device name, matched
       case-insensitively, after a trailing-ASCII-space strip that stops at the
       start of the key's own trailing run. The names are
       ``CON``, ``PRN``, ``AUX``, ``NUL``, ``CONIN$``, ``CONOUT$``,
       ``COM0``–``COM9``, ``LPT0``–``LPT9`` and the superscript ``COM¹²³`` /
       ``LPT¹²³`` forms
     - ``"CON"``, ``"nul"``, ``"con.npy"``, ``"COM1.dat"``, ``"COM0"``,
       ``"LPT0"``, ``"CONIN$"``, ``"CONOUT$"``, ``"com¹"``, ``"lpt³"``,
       ``"CON .txt"``
   * - ``ends in a space or a dot``
     - a trailing run of ASCII spaces or ASCII dots. Evaluated **last**, so a
       key violating an earlier clause still reports that earlier clause
     - ``"foo."``, ``"x."``, ``"..."``, ``"a "``, ``" "``

A key that is not a :class:`str` at all is refused too, with a message that
carries **no clause and no cache directory** — neither is meaningful for a value
that is not a key. It reads ``Invalid store key <repr>: a store key must be a
str, but got <type>.`` and it is a
:exc:`~GSEGUtils.lazy_disk_cache.StoreKeyError` like every other refusal, so a
``except StoreKeyError`` handler already covers it. Before 0.5.x this case
escaped as a bare :exc:`TypeError`, which neither that handler nor
``except ValueError`` caught.

Windows separators are refused **on Linux too**. The key is validated under both
:class:`~pathlib.PurePosixPath` and :class:`~pathlib.PureWindowsPath`
interpretations regardless of the host, so ``..\..\x`` cannot slip through a
POSIX reading that sees it as one harmless segment.

No normalisation of any kind is applied before validation. The exact characters
you supply are the characters validated *and* the characters used to build the
path — a fullwidth ``．．/victim`` is not folded into ``../victim`` before the
check, because validating one string and building a path from another is the
classic bypass.

A fifth clause, ``resolves outside its cache directory``, is the second layer:
it fires on a path that would leave the cache directory anyway, which — with the
lexical rule in place at every route — can now only happen because of the
environment, most likely a symlinked or replaced directory component inside the
cache directory.

Why the reserved names are refused
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Not because they escape.** The separator is the entire escape mechanism. The
path builders never join a bare key — they concatenate a suffix first — so:

.. list-table::
   :header-rows: 1
   :widths: 20 42 38

   * - key
     - array path built
     - escapes?
   * - ``..``
     - ``<cache>/...npy``
     - **no** — a literal file inside the cache directory
   * - ``.``
     - ``<cache>/..npy``
     - **no** — a hidden file inside the cache directory
   * - ``../victim``
     - ``<cache>/../victim.npy`` → ``<parent>/victim.npy``
     - **yes** — and the separator is what did it

So ``.`` and ``..`` are refused as **defence-in-depth against a future bare
join**, not as the closing of a live hole. Stating only the rule would leave the
next reader with a false model of where the danger is.

The empty key has its own, concrete reason: ``''`` builds ``<cache>/.npy``, a
file no key can own. It is refused because it does not round-trip.

Reading a ``<cache>/.npy`` left behind by an older version is the other half of
the same story, and it is worth stating because the library used to get it wrong.
The reopen scan derived that file's key with :attr:`~pathlib.PurePath.stem`,
which returns ``'.npy'`` for it — a *different, legal-looking* key the store then
adopted and could never load, because building a path from ``'.npy'`` gives
``<cache>/.npy.npy``. Since 0.5.x the scan strips the suffix off the file *name*
and additionally requires the derived key to rebuild the very file it came from,
so such a file is warned about and skipped rather than advertised as a key.

Why collision shapes are refused — a different threat from escape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything above this point is about **escape** — a key becoming a path outside
the cache directory. The three clauses below are about something else entirely:
**collision**, where two distinct keys end up naming *one file*. They are listed
separately because a reader who has absorbed the escape argument will otherwise
read them as arbitrary — none of them escapes anything.

.. CONTRACT-PAGE-KEYS: REFUSED-TABLE-COLUMN-1
   Every double-quoted literal in the *Refused* column below is asserted to be
   refused by the shipped rule, in
   ``tests/test_store_key_rules.py::test_contract_page_key_literals_agree_with_the_shipped_rule``.
   Only the first column is read, which is why the second may freely mention
   ``"a"`` and other *legal* keys while explaining what collides with what.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Refused
     - What the filesystem does with it
   * - a trailing dot or space — ``"foo."``, ``"x."``, ``"a "``
     - **Windows strips trailing dots and spaces from a filename.** So ``"a"``,
       ``"a "`` and ``"a."`` all resolve to the same file, and two distinct
       store keys silently overwrite one artefact.
   * - a device name — ``"CON"``, ``"nul"``, ``"com1.dat"``, ``"COM0"``,
       ``"LPT0"``, ``"CONIN$"``, ``"CONOUT$"``, ``"com¹"``, ``"lpt³"``,
       ``"CON .txt"``
     - These name character devices, not files. A write is discarded and the
       read comes back empty. The suffix does not help: ``con.npy`` is still
       the device, which is why the match is against the pre-dot stem. The set
       is the reserved list Microsoft's *Naming Files, Paths, and Namespaces*
       enumerates, so it includes the zero-suffixed ports ``COM0``/``LPT0``, the
       console names ``CONIN$``/``CONOUT$`` and the superscript port spellings.
       **The last example is the one that looks like a typo and is not:**
       ``"CON .txt"`` is refused because Win32 strips trailing spaces from the
       name component *before* resolving it, so the space does not save the key
       — the stem it resolves is ``CON``. **The strip has a boundary, and it is
       where this clause stops and the trailing-run clause starts:** it reaches
       spaces that are *interior* to the key and stops at the start of the key's
       own trailing run. So ``"CON .txt"``, whose space is interior, is refused
       here as a reserved name, while ``"CON ."`` and ``"com1 "`` — whose
       trailing runs reach back to the space — are refused under
       ``ends in a space or a dot`` instead. Both are refused; only the reported
       clause differs.
       Matching is case-insensitive over the exact characters supplied and
       folds nothing: a fullwidth ``ＣＯＮ`` stays outside the set and stays
       legal, because on Win32 it is an ordinary filename.
   * - a bare colon — ``"ab:cd"``
     - On NTFS this opens an *alternate data stream* on the file ``ab`` rather
       than creating a file called ``ab:cd``.

These apply **on every host**, exactly like the Windows separator rules above: a
cache directory written on Linux may be read on Windows, and a key that is legal
in one place and collides in the other is not a key you want the library to have
accepted. Silently *sanitising* such a key would be worse than refusing it —
sanitising is precisely how two keys become one file without anybody noticing.

.. _RoundTripResidual:

Where the exact round-trip stops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The exact round-trip promised at the top of this page holds for the key *shapes*
the rule accepts. It **cannot** hold against two filesystem behaviours, and no
key rule can fix either, because neither is visible in the key:

* **Case-insensitive filesystems** — NTFS, and APFS in its default
  configuration — collapse ``Foo`` and ``foo`` onto one file.
* **Unicode-normalising filesystems** — APFS again — collapse the composed and
  decomposed forms of the same name onto one file.

Both are properties of the filesystem, not of the string, so a lexical rule
inspecting the key has nothing to inspect. This residual is **accepted and
stated rather than silently carried**: if your keys differ only by case, or only
by Unicode normalisation form, do not rely on them being distinct entries on
those filesystems. On a case-sensitive, non-normalising filesystem — Linux
ext4/xfs, and APFS configured case-sensitive — the exact round-trip holds as
written.

There is a third residual, and it is a different kind: **the device-name refusal
is narrowed, not closed.** Naming only the two filesystem behaviours above left
the device axis reading as complete when it is a fixed list, so, precisely:

* It is a **fixed name list** — ``CON``, ``PRN``, ``AUX``, ``NUL``, ``CONIN$``,
  ``CONOUT$``, ``COM0``–``COM9``, ``LPT0``–``LPT9`` and the superscript
  ``COM¹²³`` / ``LPT¹²³`` forms — matched case-insensitively against the pre-dot
  stem after a trailing-ASCII-space strip that reaches only spaces *interior* to
  the key and stops at the start of the key's own trailing run. It covers exactly
  those names, in exactly that position.
* It does **not** model every Win32 path-parsing behaviour. Any device-resolving
  shape outside that list, or reached by a mechanism other than a trailing-space
  strip on the stem, is not covered.
* The mechanism claims behind it — that Win32 resolves these names to character
  devices, and that it strips trailing spaces before doing so — are **Win32
  filesystem behaviour and cannot be confirmed on the hosts this library is
  tested on**. What the test suite confirms is which keys the shipped rule
  accepts and refuses, which is a narrower claim than the mechanism.

So the collision axis is **narrowed, not closed**, on all three counts. The
device list is enumerable and is enumerated; the other two are not fixable by
any lexical rule at all, because refusing one spelling of a colliding pair would
mean choosing a canonical form — which is exactly the normalisation this rule
forbids.

Why nested keys are refused — this one is a bug fix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A key such as ``tile_03/range`` is refused by the separator clause. If you were
using nested keys, the refusal is **fixing a silent data-loss bug rather than
taking a feature away.**

A nested key inserts fine, offloads fine and reads back fine — until the store is
reopened. The reopen rescan globs the cache directory **non-recursively**, so the
nested file is never rediscovered: the key is untracked, nothing points at the
file, and it stays on disk forever. The failure is invisible at write time and
surfaces as a cache miss in a later session, which is the hardest shape of bug to
attribute.

Per-directory nesting is still available, and always was — extend the cache
directory rather than the key, with
:meth:`~GSEGUtils.lazy_disk_cache.LazyDiskCacheConfig.extend_cache_path`. That
route takes one segment at a time and is validated by the same rule.

Reading the store mapping
~~~~~~~~~~~~~~~~~~~~~~~~~

:attr:`~GSEGUtils.lazy_disk_cache.DiskBackedStore.store` returns a **read-only
view** of the entry mapping, not the live dictionary. Reading through it is
unchanged — iteration, ``len()``, ``in``, ``.keys()``, ``.values()``,
``.items()`` and subscript reads all behave exactly as before — but **mutating
through it raises**, because a key inserted that way would bypass every
validation route this page describes.

Insert through the supported routes instead: ``store[key] = entry``, or
:meth:`~GSEGUtils.lazy_disk_cache.DiskBackedStore.add_data_to_store`. Both
validate the key. See ``BC-GSEG-007`` in ``MIGRATION-v1.0.md`` for the migration
detail, including the one downstream annotation that needs widening.

What to catch
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Exception
     - Meaning
   * - :exc:`~GSEGUtils.lazy_disk_cache.StoreKeyError`
     - The key is not a legal single path segment. Subclasses
       :class:`ValueError`, so every existing ``except ValueError`` keeps
       catching it. It is deliberately **not** a :class:`KeyError` — the store
       already raises that one for "key exists".
   * - :exc:`~GSEGUtils.lazy_disk_cache.StoreContainmentError`
     - A built path would resolve outside the cache directory. Subclasses
       :exc:`~GSEGUtils.lazy_disk_cache.StoreKeyError`, so a broad handler
       catches both.
   * - :exc:`~GSEGUtils.lazy_disk_cache.StorePurgeRefusedError`
     - **The root of the purge-refusal family**, whose shared guarantee is that
       a refused purge touched nothing. It has two members: the *foreign
       process* case — :meth:`~GSEGUtils.lazy_disk_cache.DiskBackedStore.purge`
       was called from a process that did not construct the store — and the
       *foreign artefact* case in the next row. **So one**
       ``except StorePurgeRefusedError`` **covers both**, and code that needs to
       tell them apart catches the subclass first. Read the type as "refused,
       nothing touched" rather than as the process case alone; it used to
       describe only the latter. Subclasses :class:`RuntimeError` — **not**
       :exc:`~GSEGUtils.lazy_disk_cache.StoreKeyError` — because nothing is
       wrong with the key; the refusal is about *the caller*. So the broad
       ``except RuntimeError`` that worker code already writes catches it
       without a new handler. Raised before any mutation, so a refused purge is
       a bit-for-bit no-op.
   * - :exc:`~GSEGUtils.lazy_disk_cache.StorePurgeForeignArtefactError`
     - An artefact of the key — a built path's resolved target, or a live
       entry's own ``cache_path`` — resolves outside the store's cache
       directory, so ``purge`` refused rather than reaching out there to unlink
       it. Subclasses
       :exc:`~GSEGUtils.lazy_disk_cache.StorePurgeRefusedError`, so an existing
       broad handler keeps working and inherits the same no-op guarantee. The
       entry's finalizer is deliberately **left armed**, so the file is still
       reclaimed on collection; see :ref:`RemovingAKeyAndItsFiles`.
   * - :exc:`~GSEGUtils.lazy_disk_cache.StorePurgeIncompleteError`
     - One or more of the key's artefacts could not be unlinked. Subclasses
       :class:`OSError`, so an existing handler around a deleting operation
       keeps working. Raised **once**, after every artefact has been attempted,
       with the survivors named in the message. The key stays dropped.

The two key-flavoured types are separate on purpose. A ``StoreKeyError`` is evidence about
*the caller's key*; a ``StoreContainmentError`` is evidence about *the
environment* — something planted a symlink in the cache directory. A per-item
handler that skips one bad key should not silently swallow the second kind, so
catch the base type only where you mean "this key was bad".

.. CONTRACT-PAGE-ROUTES: BEGIN

**Not every route raises, so not every route needs a handler.** The list below
is every mapping route that can meet a key — all ten of them — and it is
*enforced* to be every one, not merely intended to be: the regression test
``test_contract_page_route_paragraph_names_every_overridden_mapping_route``
derives the set by introspection and fails the build if a route is missing from
this page. An earlier draft named five and called itself exhaustive; the two it
omitted were the two that were then found to misbehave, which is the usual way
round.

* ``key in store`` (``__contains__``) — answers ``False`` for an illegal key.
* ``store.get(key, default)`` (``get``) — returns its default.
* ``store.pop(key, default)`` (``pop``, defaulting form) — returns its default.
* ``store.pop(key)`` (``pop``, bare form) — **raises**, because answering a miss
  with ``None`` is not what a mapping ``pop`` does.
* ``store[key]`` (``__getitem__``) — **raises**; this is the route the others
  are defined against.
* ``store[key] = entry`` (``__setitem__``) — **raises**, before any filesystem
  call.
* ``del store[key]`` (``__delitem__``) — removes tracking; it builds no path, so
  it has no key to refuse.
* ``store.setdefault(key, default)`` (``setdefault``) — **raises**, because
  ``setdefault`` is a **write** route and refusing an illegal key on a write is
  the point of this contract. Its resemblance to ``get`` is not a reason to give
  it ``get``'s behaviour.
* ``store.update(other)`` (``update``) — **raises**, per key, through
  ``__setitem__``.
* ``store.clear()`` (``clear``) — empties the store or raises; it can no longer
  return with keys still tracked.
* ``store.popitem()`` (``popitem``) — left as ``MutableMapping`` supplies it.
  Its ``KeyError`` carries the key on a non-empty store and carries nothing on
  an empty one, so the two cases are distinguishable.

The last four are supplied by ``collections.abc.MutableMapping`` rather than
written in this library, so they appear in no diff that changes their behaviour.
That is why they were missed, and why the enumeration above is now derived from
the ABC rather than remembered.

A ``StoreContainmentError`` propagates out of *all* of them, including the
defaulting ones — it is evidence about the environment rather than about the
key, and swallowing it behind a default would hide exactly the case worth
seeing. Nothing is removed when it propagates.

**What a removal leaves behind.** Every removal route above — ``pop``,
``del store[key]`` and ``clear`` — drops the key from the store's **in-memory**
tracking and **leaves the entry's files on disk**. So a removed key that had
been offloaded is re-adopted by the next ``store[key]``, which falls back to the
loader for any untracked key, and again by the rescan when the store is
reopened. Between the removal and that read, ``key not in store`` and
``store[key]`` both succeed. Treat this as a known limitation rather than a
guarantee. **You no longer have to unlink the artefacts yourself:** the atomic
drop-key-and-delete-files operation (**STORE-04**) ships as
:meth:`~GSEGUtils.lazy_disk_cache.DiskBackedStore.purge` — see
:ref:`RemovingAKeyAndItsFiles` below. An earlier revision of this page said it
was "not in this release", which was true when written and is not now.

The full per-route enumeration with migration detail lives in ``BC-GSEG-006`` in
``MIGRATION-v1.0.md``.

.. CONTRACT-PAGE-ROUTES: END

Pre-checking a key
~~~~~~~~~~~~~~~~~~

:func:`~GSEGUtils.lazy_disk_cache.is_valid_store_key` is the supported way to
check a composed name **before** it becomes a key, without catching an
exception:

.. code-block:: python

   from GSEGUtils.lazy_disk_cache import is_valid_store_key

   name = f"{feature}_{variant}"
   if not is_valid_store_key(name):
       raise ValueError(f"{name!r} cannot be used as a cache key")

The call is pure: it touches no filesystem, mutates nothing, and returns the same
verdict every time. It always produces a verdict **for a non-**:class:`str`
**argument**: anything that is not a :class:`str` — including a
:class:`~pathlib.Path`, which is the shape a path-typed identifier arrives in —
returns ``False`` rather than raising, so for those arguments the predicate does
not need to be wrapped in a ``try``.

There is exactly one argument shape for which it does. A :class:`str`
**subclass** whose ``__hash__`` raises passes the non-:class:`str` guard — it *is*
a :class:`str` — and its exception then propagates out of the reserved-name
membership test. If you compose keys out of third-party objects that subclass
:class:`str` with overridden dunders, wrap the call; if you pass ordinary values
of any type, you do not need to. The predicate deliberately does **not** catch
that exception: a blanket handler would also swallow
:class:`~GSEGUtils.lazy_disk_cache.StoreContainmentError`, which is the signal
that something was planted in your cache directory and is the one thing this
page asks you never to hide.

Scanning your existing cache directories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The snippet below reports which of the keys already on disk will now be refused.
It **imports the predicate** rather than restating the rule as a pattern, so it
cannot drift away from what the library actually enforces. It also derives the
key from a filename exactly the way the store's own reopen scan does — that is
what ``store_key_for`` is — so the two cannot disagree about *which key a given
file belongs to*, which is a second way a scan and a library can drift apart even
when they share one rule.

It is read-only: it opens no file, writes nothing and creates nothing, so it is
safe to interrupt and safe to re-run.

It scans for ``.npy``, which is the array extension the store writes — the
snippet and the store's own reopen scan read that extension from the same place,
so neither can be pointed at a set of files the other does not see.

.. caution::
   Exclude type-checker caches, virtual environments and ``site-packages``. The
   ``.meta.json`` sidecar extension collides with mypy's own cache format, and a
   scan that did not exclude them reported **thousands** of false hits during
   this change's research. The ``SKIP`` set below is doing real work.

.. This code block is executed verbatim by
   tests/test_store_containment.py (-k migration_snippet). Keep them in step:
   the test extracts this block from this file, so editing one edits both.

.. code-block:: python

   import os
   from pathlib import Path

   from GSEGUtils.lazy_disk_cache import is_valid_store_key

   SKIP = {".mypy_cache", ".pytest_cache", ".ruff_cache", ".git", ".venv", "venv", "site-packages"}


   def store_key_for(npy: Path, cache_dir: Path) -> str:
       """Return the store key the array file *npy* belongs to.

       The suffix is stripped off the *file name* and the relative parent is
       rejoined, which is exactly how the store's own reopen scan derives a key.
       """
       relative = npy.relative_to(cache_dir)
       key = npy.name[: -len(".npy")]
       parent = str(relative.parent)
       return key if parent == "." else os.path.join(parent, key)


   def refused_keys(cache_dir: Path) -> list[str]:
       """Return the keys already on disk under *cache_dir* that are now refused."""
       found = set()
       for npy in Path(cache_dir).rglob("*.npy"):
           relative = npy.relative_to(cache_dir)
           if SKIP.intersection(relative.parts):
               continue
           found.add(store_key_for(npy, cache_dir))
       return sorted(key for key in found if not is_valid_store_key(key))

Every key it returns is one that used to work and now raises. A nested key it
reports is also a file that your store has not been able to see since the last
time the process restarted.

.. note::
   The scan covers whichever directories you point it at. Cache directories are
   often configuration-driven and may not all live where you expect, so run it
   over each root your application configures rather than over one known path.

.. _RemovingAKeyAndItsFiles:

Removing a key and its files
----------------------------

:meth:`~GSEGUtils.lazy_disk_cache.DiskBackedStore.purge` is the durable
counterpart to ``del store[key]``. Where the removal routes above drop tracking
and leave everything on disk, ``store.purge(key)`` drops tracking **and**
unlinks the key's artefacts — and it is the only removal that sticks, because
with nothing on disk there is nothing for the next read or the reopen rescan to
re-adopt.

What it removes
~~~~~~~~~~~~~~~

**Every artefact whose name derives from the key** — stated as a rule rather
than as a list, so it stays true when an artefact is added. Today that is
``<key>.dat``, ``<key>.dat.tmp``, ``<key>.npy``, ``<key>.npy.tmp``,
``<key>.meta.json`` and ``<key>.meta.json.tmp``. The ``.tmp`` names are in the
set deliberately: each persists from creation until its rename, and a crash in
between leaves one behind indefinitely, so a purge that skipped them would leak
the very files the atomicity work creates.

You can check it the way the rule is written — list the directory afterwards and
find nothing derived from **that** key. Note the qualifier: a prefix glob such as
``glob(f"{key}.*")`` also matches a *longer* key's artefacts, so ``feat`` and
``feat2`` are not independent under a naive check.

**And one file whose name does not derive from the key.** Each built path is
followed to its **resolved target**, so where ``<key>.dat`` is a symlink pointing
at a payload **inside** the cache directory — the legitimate *adopted entry* —
the purge removes that payload along with the link. A file whose name is not
derived from the key may therefore be removed by a purge of that key.

The reason is mechanical. ``Path.unlink`` removes the link and never its target,
so a purge that only unlinked names left the key's exact bytes sitting in the
directory under another name **and reported a complete removal**.

**The removal set in full, and it has no other members.** ``purge`` unlinks the
six names built from the key and — where one of those names is a symlink — the
single path that name resolves to. There is no third category: it does not reach
another key's artefacts, it does not reach a live entry's own ``cache_path``, and
it does not reach outside the cache directory. **That boundary is held by two
refusals rather than asserted as a rule**, because as a bare rule it was once
false on an input a caller can reach — before the second refusal below existed,
one planted in-cache ``aggressor.dat`` aimed at a genuine, store-written
``victim.npy`` made ``purge("aggressor")`` unlink it and return cleanly, leaving
``"victim"`` tracked by the store and unreadable through it. Both refusals are
raised before the first unlink and leave the directory bit-for-bit unchanged:

* a built name whose resolved target lies **outside** the cache directory raises
  :exc:`~GSEGUtils.lazy_disk_cache.StorePurgeForeignArtefactError`;
* a built name whose resolved target lies **inside** it but carries a store
  artefact name that is not one of this key's own six raises
  :exc:`~GSEGUtils.lazy_disk_cache.StorePurgeAliasedArtefactError`.

So *belonging* is decided by what a path resolves to rather than by what it is
named, and the two refusals are what stop that resolution reaching past the key
it was derived from. Both are described in full under
:ref:`the refusal family <PurgeRefusalFamily>` below.

**The residual limit, stated rather than guarded.** The cross-key shape above is
now **refused**, and the earlier statement of this limit was wrong in three ways
a review measured: it said two planted links were needed when **one** suffices;
it described the file at risk as a shared adopted payload when the file actually
destroyed was a **store-owned artefact** of a live key; and it argued that
guarding the shape would cost a scan of every other key's link target, which the
shipped refusal disproves — the test is a comparison of the target's *name* and
scans nothing.

What remains unguarded is narrower. **Two keys whose ``<key>.dat`` links point at
the same payload, where that payload's own name is not one the store builds** —
purging either removes the other's data. No store-owned write route can produce
that shape, since each key builds its own ``<key>.dat`` name. The name test
cannot see it either, and deliberately so: a payload named ``shared.bin`` is
exactly the legitimate *adopted entry* shape the refusal has to let through.
Telling the two apart needs ownership information a file name does not carry, so
this is published as a limit rather than guarded by a rule that would break the
adopted entry.

The legacy ``<key>.pkl`` is **not** removed. It is unreadable by design — the
store refuses it rather than invoking the pickle reader — and it is now also
unremovable by the only removal verb, so a pre-0.5 cache directory keeps it
forever and a directory listing after a *complete* purge still shows the key's
name. That is a known, deliberate residue rather than a failed purge; nothing in
the library will read it.

When it raises :class:`KeyError`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Only when the key is absent from tracking and from disk.** Untracked-but-on-disk
counts as present and is purgeable — and that is the case you will meet most,
because it is exactly the state ``del store[key]`` leaves behind. A stricter
reading ("tracked only") would make the orphan case unpurgeable through the one
verb built to purge it.

.. _PurgeRefusalFamily:

When it refuses a foreign artefact
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When **any** artefact of the key resolves **outside** the store's cache directory
— a built path's resolved target, or a live entry's own ``cache_path`` —
:meth:`~GSEGUtils.lazy_disk_cache.DiskBackedStore.purge` raises
:exc:`~GSEGUtils.lazy_disk_cache.StorePurgeForeignArtefactError` and touches
nothing at all.

Three consequences follow that the refusal itself does not tell you.

**The finalizer is deliberately left armed.** The file is still reclaimed when
the entry is collected, exactly as it would have been had you never called
``purge``. Refusing *and* disarming would turn a garbage-collectable file into a
permanent leak created by the removal verb itself, which is the defect this
refusal exists to prevent.

**Dropping the entry is not the remedy, and the difference is the whole of it.**
Letting the entry be collected reclaims **the entry's own backing file and
nothing else**: the finalizer removes exactly one memmap by explicit design, and
never the store-owned ``<key>.npy`` and ``<key>.meta.json`` written for that key.
So while this refusal stands, that codec pair is reclaimed by **no removal verb
this package exposes**. The remedy that does work is to **repoint or remove the
offending path and purge again** — for a symlink pointing out of the cache
directory, remove or repoint the link; for an entry constructed with an explicit
``cache_path`` outside the cache directory, give it a path inside that directory
or let the store derive one from the key. The exception message says the same
thing, so an operator can act on it without reading this page.

**A purge will not delete outside its own cache directory**, and that is a
boundary rather than a limitation. A removal verb that followed a
caller-supplied path wherever it pointed is precisely the shape the containment
layer exists to prevent.

The outward symlink is the same case and refuses for the same reason, and **this
is a change**: a ``<key>.dat`` symlink whose target lies outside the cache
directory now raises, where a previous release unlinked the link and left the
target on disk. Remove or repoint the link and purge again.

An explicit purge overrides ``purge_disk_on_gc=False``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``purge_disk_on_gc`` governs *implicit, garbage-collection-time* deletion. **It
is not a write-protect bit**, so an explicit
:meth:`~GSEGUtils.lazy_disk_cache.DiskBackedStore.purge` proceeds regardless of
its value — otherwise the method would be unusable in precisely the
configuration that accumulates the most artefacts. Every purge that exercises
the override logs an ``INFO`` record naming the key, so the override is
transparent rather than merely permitted.

**The counterweight is named in the same breath, because it is what makes the
override safe rather than reckless:** a purge issued from a process that did not
construct the store **refuses**, raising
:exc:`~GSEGUtils.lazy_disk_cache.StorePurgeRefusedError` before touching
anything. So a stray purge inside a ``joblib`` / ``loky`` worker cannot delete
the parent process's session data. The guard is on this method and nothing else:
workers legitimately *write* — pickling a store force-offloads it — so guarding
the write routes would break the worker path outright. Deletion is the only
operation where "wrong process" means "destroying someone else's data".

That type is now the **root of a refusal family** rather than the name of this
one case: the foreign-artefact refusal above subclasses it, and both members
share the guarantee that a refused purge touched nothing. A single
``except StorePurgeRefusedError`` therefore covers the wrong-process case and
the foreign-artefact case alike.

If some artefacts survive
~~~~~~~~~~~~~~~~~~~~~~~~~

POSIX gives no atomicity across several ``unlink`` calls, so **a partial
directory state is the contract rather than a surprise**. Every artefact is
attempted, the failures are collected, and one
:exc:`~GSEGUtils.lazy_disk_cache.StorePurgeIncompleteError` names the survivors.
**The key stays dropped** — re-tracking it would point a live entry at a
half-deleted artefact set and would make the method non-idempotent, so calling
``purge`` again after fixing the cause is the supported recovery.

The residue you are most likely to meet is a surviving ``<key>.dat``. The
``.dat`` memmap is *entry*-owned while the ``.npy`` + ``.meta.json`` codec pair
is *store*-owned, and the two are unlinked in that order — sidecars first,
payload last. Re-adding the key afterwards yields **the new data**, not the
stale payload: the write routes replace the ``.dat`` rather than appending to
it.

What "atomic" does and does not mean here
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read this before building a call site around the word.

**What the guarantee is.** ``purge`` is atomic with respect to
**store-owned ordering and refusal**:
validation precedes every mutation, a refused purge
touches nothing at all, and the key is dropped before the first unlink, so the
tracking state never describes a half-deleted artefact set.

**What it is not — first.** It is
**not safe against concurrent mutation of the same key**.
:class:`~GSEGUtils.lazy_disk_cache.DiskBackedStore` holds no
store-level lock, so a ``store[key] = entry`` or
:meth:`~GSEGUtils.lazy_disk_cache.DiskBackedStore.add_data_to_store` racing the
call may have its freshly-written artefact unlinked underneath it, or may slip
one in after the existence check.

**What it is not — second.** It is **not globally atomic** across threads, or
across processes sharing a cache directory. POSIX offers nothing that would make
it so.

**The supported model, stated rather than left as a bare warning:**
single-threaded use per store. That is the same constraint this project states
elsewhere — a ``PointCloudData`` is not multi-thread-mutable either. If several
processes share a cache directory, coordinate above the store.

One lock *is* taken, and it should not be mistaken for a store-wide guarantee:
the live entry's ``RLock``, held for the finalizer detach alone and never across
the unlinks. Holding it across the unlinks would be a guarantee that exists only
when the key happens to be tracked — the untracked-orphan case has no entry and
therefore no lock — and a guarantee that sometimes exists is a filter, not an
invariant.

Where the ``.dat`` atomicity guarantee holds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``<key>.dat`` memmap is written through a temporary and renamed over it, so
an interrupted conversion cannot tear a previously-valid ``.dat``. **That route
is POSIX-only.** It is selected on ``os.name == "posix"``; off POSIX the
conversion falls back to a direct write on the final name, which is
**not torn-write-safe** — an interruption there can leave a partially-written
``.dat``.

The reason is platform semantics rather than an unfinished port: replacing an
open, memory-mapped file is not permitted on Windows, so the
temporary-and-rename sequence cannot complete there. Note also that GSEGUtils
declares **no OS classifiers** and is tested on Linux — so this is a scoped
guarantee stated with its scope, not an unqualified one you should read as
holding everywhere the package installs.

**The destination's permissions are carried across that rename.** ``os.replace``
moves the temporary's *inode*, and with it the mode, owner and ACLs, so without
an explicit step the artefact would silently inherit the process umask default
in place of whatever the destination had. Two consequences you can rely on: an
entry created without an explicit ``cache_path`` keeps the ``0600`` that
``tempfile.mkstemp`` gave its backing file, and an operator who tightened an
existing artefact keeps that mode across a reconversion.

**Both limits belong in the same breath as the guarantee.** A *first* write into
a configured cache directory still lands at the process umask default — there is
no prior destination to read a mode from, and the library deliberately does not
substitute a value of its own choosing, so this is unchanged rather than
tightened. And there is a brief window between the temporary's creation and its
mode being set, during which the temporary carries the umask default: on a
directory other local users can read, the payload is exposed momentarily rather
than permanently. The step narrows that exposure; it does not eliminate it.
Closing the residue needs the temporary created through
``tempfile.mkstemp(dir=...)`` with ``O_EXCL`` semantics, which is deferred.

Modules
-------

.. toctree::
   :maxdepth: 1

   GSEGUtils.lazy_disk_cache
