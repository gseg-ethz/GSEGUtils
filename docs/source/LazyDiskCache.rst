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
exactly:

.. code-block:: text

   rrim_pack_(range,r16,d8,z1e-05)     parentheses, commas, hyphens, underscores
   rrim_component_(structure,range)    a composed feature name
   z1.2345678                          interior dots
   foo.bar                             interior dots
   foo.                                a TRAILING dot
   .hidden                             a LEADING dot
   a.npy                               a key that looks like a filename

Dots are legal, including leading and trailing dots. Only the exact strings
``.`` and ``..`` are refused, and the reason is not the one most readers assume —
see *Why the reserved names are refused* below.

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
     - the exact strings ``''``, ``'.'`` and ``'..'``
     - ``""``, ``"."``, ``".."``
   * - ``contains a control character``
     - any character below ``\x20``, and ``\x7f``
     - ``"x\ny"``, ``"x\x00y"``
   * - ``is an absolute or drive-relative path``
     - a non-empty anchor or drive under *either* POSIX or Windows semantics
     - ``"/etc/passwd"``, ``"C:evil"``, ``"\\\\server\\share\\x"``
   * - ``contains a path separator``
     - ``/`` or ``\`` anywhere, including a *trailing* separator, and anything
       ``pathlib`` reads as multi-segment under either flavour
     - ``"../victim"``, ``"tile_03/range"``, ``"a/"``, ``"..\\..\\x"``

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

The empty key has its own, concrete reason: ``''`` builds ``<cache>/.npy``, whose
rescan stem is ``'.npy'`` — a *different, legal-looking* key. It is refused
because it does not round-trip.

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
verdict every time.

Scanning your existing cache directories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The snippet below reports which of the keys already on disk will now be refused.
It **imports the predicate** rather than restating the rule as a pattern, so it
cannot drift away from what the library actually enforces.

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

   from pathlib import Path

   from GSEGUtils.lazy_disk_cache import is_valid_store_key

   SKIP = {".mypy_cache", ".pytest_cache", ".ruff_cache", ".git", ".venv", "venv", "site-packages"}


   def refused_keys(cache_dir: Path) -> list[str]:
       """Return the keys already on disk under *cache_dir* that are now refused."""
       found = set()
       for npy in Path(cache_dir).rglob("*.npy"):
           relative = npy.relative_to(cache_dir)
           if SKIP.intersection(relative.parts):
               continue
           found.add(str(relative)[: -len(".npy")])
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
