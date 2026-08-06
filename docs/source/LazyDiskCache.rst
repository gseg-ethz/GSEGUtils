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
carries, so you can grep your logs for it:

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - Clause (as it appears in the message)
     - Refuses
     - Examples
   * - ``is empty or a reserved path name``
     - the exact strings ``''``, ``'.'`` and ``'..'``, **and** any key whose
       pre-dot stem is a Win32 device name, matched case-insensitively
     - ``""``, ``"."``, ``".."``, ``"CON"``, ``"nul"``, ``"con.npy"``,
       ``"COM1.dat"``
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

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Refused
     - What the filesystem does with it
   * - a trailing dot or space — ``"foo."``, ``"x."``, ``"a "``
     - **Windows strips trailing dots and spaces from a filename.** So ``"a"``,
       ``"a "`` and ``"a."`` all resolve to the same file, and two distinct
       store keys silently overwrite one artefact.
   * - a device name — ``CON``, ``nul``, ``com1.dat``
     - These name character devices, not files. A write is discarded and the
       read comes back empty. The suffix does not help: ``con.npy`` is still
       the device, which is why the match is against the pre-dot stem.
   * - a bare colon — ``ab:cd``
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

The two types are separate on purpose. A ``StoreKeyError`` is evidence about
*the caller's key*; a ``StoreContainmentError`` is evidence about *the
environment* — something planted a symlink in the cache directory. A per-item
handler that skips one bad key should not silently swallow the second kind, so
catch the base type only where you mean "this key was bad".

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
verdict every time. It is also **total over its argument**: anything that is not
a :class:`str` — including a :class:`~pathlib.Path`, which is the shape a
path-typed identifier arrives in — returns ``False`` rather than raising, so the
predicate never needs to be wrapped in a ``try``.

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

Modules
-------

.. toctree::
   :maxdepth: 1

   GSEGUtils.lazy_disk_cache
