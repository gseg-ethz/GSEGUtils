---
type: migration-spec
spec_version: 1.0
repo: GSEGUtils
baseline_ref: [2eae789, bfff748]
target_ref: e413d2ad8e8afc521ebefa87b18e569906cdc031
generated_at: "2026-06-11T11:19:03Z"
bc_id_prefix: BC-GSEG
---

# GSEGUtils MIGRATION-v1.0

**Baselines:** `doc@2eae789` (consumers via `pchandler@v2.0.0rc9`'s transitive pin) AND `v0.4.4` (`bfff748`, direct GSEGUtils consumers)
**Target:** `refactor/gsd` HEAD (SHA `e413d2ad8e8afc521ebefa87b18e569906cdc031`)
**Generated:** 2026-06-11

## Summary

GSEGUtils v1.0 ships the GSEG-research-group milestone delivered alongside `pchandler@v1.0`: Phase 1 toolchain modernisation + public angle-helper promotion (D-16); Phase 2 swap of the `pickle`-based `DiskBackedStore` codec for a safe `.npy + .meta.json` sidecar (SEC-01) plus `LazyDiskCache.__setstate__` finalizer re-registration (FRAG-03); Phase 3 BUG-01/02 hardening of `DiskBackedNDArray`'s ufunc behaviour and offload lifecycle; Phase 4 normalisation contract (explicit `source_range=` kwarg + NaN/Inf rejection — COUPLE-05) and behaviour-preserving streaming/lock-free perf wins (PERF-04/05); Phase 6 hygiene sweep. This file documents both consumer paths simultaneously: apps that reach GSEGUtils via `pchandler@v2.0.0rc9`'s pyproject `git+ssh` pin (`doc@2eae789` baseline) and apps that depend on GSEGUtils directly (`v0.4.4` baseline). Zero entries are classified `surface-removed`; zero entries are classified `must-edit` — PROJECT.md's "no breaking public import paths" hard constraint holds across both baselines.

## Public API stability invariant

The public import surface of `GSEGUtils` is byte-for-byte stable from both `doc@2eae789` and `v0.4.4` through `refactor/gsd` HEAD. Every name in `30_GSEGUtils/src/GSEGUtils/__init__.pyi`'s `__all__` (`base_arrays`, `base_types`, `config`, `constants`, `generate_init_stubs`, `logging_setup`, `singleton`, `util`, `validators`, `__author__`, `__email__`, `__version__`, `version`, `__version_tuple__`, `version_tuple`) and every name in `30_GSEGUtils/src/GSEGUtils/lazy_disk_cache/__init__.py`'s `__all__` (`LazyDiskCache`, `LazyDiskCacheKw`, `LazyDiskCacheConfig`, `DiskBackedNDArray`, `DiskBackedStore`) continues to resolve at its documented import path. The §"Verifier (inline)" section ships a Tier 1 AST walk that asserts every `BC-GSEG-NNN` entry's top-level `affected_symbols` resolves against this public-surface union, plus a Tier 2 `inspect.signature` runtime check against `GSEGUtils.validators.normalize_uint8` / `normalize_uint16` / `linear_map_dtype` confirming the Phase 4 COUPLE-05 `source_range` keyword is present at HEAD. Per D-08, the table schema below carries TWO migration columns (`migration_from_doc`, `migration_from_v044`); when an entry applies to only one baseline, the inapplicable column carries the exact filler `no change from this baseline — already present` so downstream tooling (the Plan 07-04 dry-run simulator) handles the generic case uniformly. Across BC-GSEG-001..006 this filler does NOT appear in practice — all six documented changes apply to both baselines (Phase 0..6 and Phase 14 changes are all post-`doc@2eae789` and post-`v0.4.4`) — but the convention is documented here so the simulator's parser remains agnostic.

BC-GSEG-006 adds six names to that second `__all__` — `StoreKeyError`, `StoreContainmentError`, `is_valid_store_key`, `get_npy_path`, `get_meta_path`, `get_legacy_pickle_path` — which is purely additive and leaves the invariant intact. The behaviour deltas that entry documents are changes to what those and the pre-existing callables *accept*, not to where any name resolves.

**The invariant is about where names resolve, and BC-GSEG-007 is the one entry that goes further than that — stated here so the stability claim above is not read as covering it.** `DiskBackedStore.store` still resolves at exactly the same import path and is still a mapping you can read, but its **published return type is narrowed** from `dict[str, Optional[T]]` to `Mapping[str, Optional[T]]`, and the object it hands back is a read-only `MappingProxyType`. No name was added, removed or moved, so the byte-for-byte *import-surface* invariant holds unchanged; but a downstream whose own annotation restates the old `dict` type has a type error rather than a resolution error, which the invariant paragraph alone would not have warned them about. See BC-GSEG-007 for the one-line fix.

## Breaking changes & behavior changes

| BC-ID | category | severity | affected_symbols | origin | migration_from_doc | migration_from_v044 |
|---|---|---|---|---|---|---|
| BC-GSEG-001 | on-disk-format | should-review | `GSEGUtils.lazy_disk_cache.DiskBackedStore`, `GSEGUtils.lazy_disk_cache.LazyDiskCache` (`__setstate__`, `offload`) | Phase 2 D-02..D-07 + D-18..D-21 / SEC-01 + FRAG-03 / Plans 02-01 + 02-04 | `DiskBackedStore` now persists arrays via `np.save` (`.npy`) + JSON sidecar (`.meta.json`) written atomically (`tmp + fsync + os.replace`); the legacy `pickle`-based codec is gone. Legacy `.pkl` cache files on disk are refused with `KeyError` + an INFO log entry — downstream code that materialised caches under `doc@2eae789` must re-materialise them via the upstream `DiskBackedStore(...)` factory. `LazyDiskCache.__setstate__` re-registers its weakref finalizer through the canonical `enable_purge()` path; round-tripping a pickled `LazyDiskCache` no longer leaks file handles. Cross-repo: this is the GSEGUtils half of pchandler's BC-PCH-005 (caching pathways that transit `DiskBackedStore`). | Same as `migration_from_doc` (Phase 2 swap landed after both baselines; the behaviour is identical for direct consumers). |
| BC-GSEG-002 | signature-shape | should-review | `GSEGUtils.validators.normalize_uint8`, `GSEGUtils.validators.normalize_uint16`, `GSEGUtils.validators.linear_map_dtype` | Phase 4 D-12..D-18 / COUPLE-05 / Plan 04-06 | All three callables now accept a keyword-only `source_range: tuple[float, float] = (0.0, 1.0)` parameter that locks the precise scaling envelope. Out-of-range floats clip-and-saturate silently to the integer dtype's range. NaN / Inf inputs now raise `ValueError` (previously they were propagated through the rescale, producing dtype-truncation garbage). Integer-typed input is unchanged. Migrate by passing `source_range=` explicitly at every call site — pchandler's call sites have already been migrated (cross-references the pchandler-side COUPLE-05 audit). | Same as `migration_from_doc` (Phase 4 is the only origin; no divergence between baselines). |
| BC-GSEG-006 | signature-shape | should-review | `GSEGUtils.lazy_disk_cache.DiskBackedStore`, `GSEGUtils.lazy_disk_cache.LazyDiskCache`, `GSEGUtils.lazy_disk_cache.LazyDiskCacheConfig`, `GSEGUtils.lazy_disk_cache.StoreKeyError`, `GSEGUtils.lazy_disk_cache.StoreContainmentError`, `GSEGUtils.lazy_disk_cache.is_valid_store_key`, `GSEGUtils.lazy_disk_cache.get_npy_path`, `GSEGUtils.lazy_disk_cache.get_meta_path`, `GSEGUtils.lazy_disk_cache.get_legacy_pickle_path` | Phase 14 D-01..D-18 / STORE-01 + STORE-02 + STORE-03 + STORE-07 / Plans 14-01..14-07 | A `DiskBackedStore` key becomes a filename, and it is now validated before any path is built from it — at every route that reads, writes, rescans or unpickles an entry — with every built path additionally checked to resolve *inside* the cache directory. **Four behaviour deltas.** **(1) Previously-accepted keys are now refused.** The one that matters is the nested key (`tile_03/range`), and the refusal is a **bug fix, not a feature removal**: a nested key inserts, offloads and reads back fine today, and then vanishes on store reopen, because the reopen rescan globs the cache directory *non-recursively* — the file stays on disk, no key points at it, and it leaks forever. Also refused: `''`, `'.'`, `'..'`, absolute and drive-relative paths (`/etc/passwd`, `C:evil`, validated under **both** POSIX and Windows semantics regardless of host, so `..\..\x` is refused on Linux), any `/` or `\` including a *trailing* one, and control characters including newline and NUL. **The refusal set was widened again after the first round, to cover filesystem *collision* as well as traversal — check this list even if you already migrated against an earlier draft of this note.** Newly refused: **(a)** any key ending in a run of ASCII spaces or dots — `foo.`, `x.`, `...`, `a ` and `' '` — because **Win32 strips trailing dots and spaces from a filename**, so `'a'`, `'a '` and `'a.'` would resolve to one file and two distinct keys would silently overwrite one artefact; **(b)** any key whose pre-dot stem is a Win32 device name, matched case-insensitively, so `CON`, `nul`, `con.npy` and `COM1.dat` are all refused, because a write to a character device is discarded and the read comes back empty — **and this half was widened again in a later round; the enumerated list is two paragraphs down and it is longer than these four examples**; **(c)** any key containing a bare `:`, such as `ab:cd`, because that opens an NTFS alternate data stream (note this is *not* reachable by tightening the drive test — `PureWindowsPath('ab:cd').drive` is `''`); and **(d)** any **non-`str`** argument, which previously escaped as a bare `TypeError` from the control-character scan and is now a `StoreKeyError` naming the received type, so `is_valid_store_key(Path('../victim'))` returns `False` instead of raising. **Leading and interior dots are unaffected** — `.hidden`, `foo.bar`, `z1.2345678` and `a.npy` all still pass, and the strip is over ASCII space and ASCII dot only, so composed feature names containing fullwidth or non-ASCII characters are not folded. **The device half of (b) was then widened a second time, so check this list again even if you already migrated against the collision draft above — this is the second widening of the refusal set in this release, and migrating once no longer takes you out of scope.** The device set is now the reserved list Microsoft's *Naming Files, Paths, and Namespaces* enumerates, which the previous draft had shipped as its commonly-cited subset. Refused on top of the four examples in (b): **`COM0` and `LPT0`** (the previous set's port comprehensions started at `1`; the zero-suffixed forms are reserved too), **`CONIN$` and `CONOUT$`** (the reserved console input and output names), the **superscript port spellings** `COM¹` `COM²` `COM³` / `LPT¹` `LPT²` `LPT³` (on that same reserved list), and — the shape most likely to be read as a typo — **a reserved name followed by ASCII spaces and then an extension**, such as `CON .txt`, because **Win32 strips trailing spaces from the name component *before* resolving it**, so the space does not save the key. The full refused set is `CON`, `PRN`, `AUX`, `NUL`, `CONIN$`, `CONOUT$`, `COM0`–`COM9`, `LPT0`–`LPT9` and the superscript `COM¹²³` / `LPT¹²³` forms, matched case-insensitively against the pre-dot stem. **The widening folds nothing:** the superscript entries are exact characters (`'¹'.upper()` is `'¹'`), and the fullwidth spelling `ＣＯＮ` is still accepted, which is D-05's no-normalisation rule holding under a rule that now compares against a name list. The trailing-space strip is applied to the stem for the membership test **only when the key contains a dot, and it reaches only spaces that are *interior* to the key — it stops at the start of the key's own trailing run** (D-30). Stated as the boundary rather than as one example, because the example alone was misleading: `com1 ` keeps reporting the trailing clause, and so do all four **crossover** shapes, which carry a device stem, an interior space *and* a dot — `con .`, `nul .`, `CON .` and `CON . `. Each of those has a trailing run that reaches back through the dot to the space, so the space belongs to the trailing clause and the strip cannot consume it. A reserved name whose space is genuinely interior — `CON .txt` — is still refused as reserved. Both halves are refused either way; only the reported clause differs. **If you migrated against a draft published between the round-2 and round-4 states of this note, re-check this family:** in that window the four crossover shapes reported `is empty or a reserved path name`, and they now report `ends in a space or a dot` again, which is what they reported before the device widening. A downstream that greps clause text and pinned the intermediate attribution should re-run its grep. **The exposure this round could not measure, stated rather than omitted:** a downstream that derives a name from a *filename stem* can produce a stem that ends in a dot or names a device, and that route is under the same rule — iof3D's `extend_cache_path(path_ext=pcd_id.stem)` (`image_generation.py:156-160`) is exactly that shape. A refused key raises `StoreKeyError`, a subclass of `ValueError` — deliberately **not** a `KeyError`, which `add_data_to_store` already raises for "key exists". Grep your logs for the message shape ``Invalid store key '<key>' for cache directory '<dir>': the key <clause>``, where `<clause>` is one of `is empty or a reserved path name` (which now also covers the Win32 device names), `contains a control character`, `is an absolute or drive-relative path` (which now also covers the bare colon), `contains a path separator`, or — **new in the collision widening** — `ends in a space or a dot`. **A grep written against the pre-widening list will miss every refusal this round added.** **The second widening changed no clause string** — the device half still reports `is empty or a reserved path name` — **so a grep written against the list above keeps matching, and there is nothing to add to it.** What it *did* change is the **evaluation order**: the device test now runs after the control-character, absolute-path and separator tests rather than first, so a key that violates several clauses at once may now report a different one than it did in the previous draft. Concretely and by measurement: `con.a/b`, `lpt1.a/` and `com1.\\server\share\x` reported `is empty or a reserved path name` under the previous draft and now report `contains a path separator`; `nul.x\n` now reports `contains a control character`; and `aux.C:evil` now reports `is an absolute or drive-relative path`. In every case that is the clause the key reported *before* the collision draft introduced the device test, so this is a restoration rather than a third vocabulary state. The order was changed deliberately, so that finishing the device list repoints no key that was already refused for another reason; the cost is that five compound shapes move back to the clause they reported before the collision draft. **If you count or route on the clause, treat the refusal as the event and the clause as diagnostic detail** — a compound key's clause is a statement about which test fired first, not about which properties the key has. Separately, and on a different message: the reopen rescan's skip WARNING now renders both the skipped filename and the cache directory with `repr` rather than interpolating them raw. If you match on that message's shape, the quoting moved; the reason is that an unescaped filename could carry newlines and forge a whole log record, which is a filename an attacker can plant in a shared cache directory. One exception to the shape above, so a log search does not come up empty on it: the non-`str` refusal in (d) carries **no clause and no cache directory**, because neither is meaningful for a value that is not a key at all; its message reads ``Invalid store key <repr>: a store key must be a str, but got <type>. Convert it explicitly — …`` A resolved-containment violation raises the `StoreContainmentError` subclass with the clause `resolves outside its cache directory`; it is separately typed because it is evidence about the *environment* (a symlink planted in the cache directory), not about your key. **One more consequence, and it is the reason to check this list even if you already migrated against an earlier draft: the reopen rescan now reads the artefact extension from a fixed module constant rather than off the instance.** A store that subclassed `DiskBackedStore` to repoint that extension therefore adopts the **base-suffix** artefacts it was in fact writing all along — which is a fix, not a regression: measured, such a subclass never wrote its own extension, because the write path ignored the attribute. In the previous draft that same store adopted keys it could then raise `KeyError` on, with no warning; it now adopts the artefacts it really has, and reads them. **(2) The entry path setter is sealed.** Assigning to `LazyDiskCache.cache_path` now raises `AttributeError` naming the entry, the attempted path and the alternative. Construct through the store, or pass `cache_path=` at construction time. This takes no deprecation cycle because a survey of four repositories found **zero** assigners — every reference was a read. **The previous draft continued *“whereas the builder aliases below take a full cycle precisely because they have measured live callers”*, and that contrast is withdrawn: the aliases are removed in this release too. Delta (5) records why, and the reasoning error is worth reading before you write your own deprecation.** **(3) A subscript read with an illegal key now raises `StoreKeyError` — a `ValueError` — where it previously raised `KeyError`.** Before, `store['../victim']` reached the loader, found no files and raised `KeyError` from the cache-miss branch; it is now refused lexically at the mapping surface, with a type that is deliberately not a lookup error. A caller wrapping a read in `except KeyError` stops catching it and must add the new type — or catch `ValueError`, which already covers it. **State this next to what did *not* change, because the obvious inference is wrong — and take the following as an exhaustive enumeration of the routes that answer rather than raise, because the previous draft named only two of the three and the missing one had in fact changed.** **↻ That enumeration was itself short, twice, and the third draft below is the one with an enforcer rather than an author.** The list this paragraph introduced named five routes and omitted `clear()` and `popitem()`; `clear()` was then measured returning normally with the store still populated. **Here is why an enumeration keeps coming up short at exactly this spot, which is worth more than the corrected list: the routes that kept going missing are supplied by `collections.abc.MutableMapping` and travel this library's overridden `__getitem__` / `__setitem__` / `__delitem__` without appearing anywhere in its source.** They are invisible in the module, and — the part that actually bites — invisible in the diff that changes their behaviour, because that diff touches an accessor and they are not in it. If you are adding an override to a `MutableMapping` subclass, the set to check is not the methods you can see; it is `vars(MutableMapping)` minus what your class defines. **The exhaustiveness claim survives this draft only because it is now enforced:** `tests/test_store_containment.py::test_contract_page_route_paragraph_names_every_overridden_mapping_route` derives both sets by introspection — the class's own mapping methods, and `MutableMapping`'s concrete mixins that the class does not define — and fails the build if a route is missing from the published list. A one-set derivation was tried against a deliberately shortened list and passed, so the two-set shape is not decoration. Three routes answer an illegal key instead of raising: **`key in store` answers `False`**; **`.get(key, default)` returns its default**, since `get` is explicitly overridden to catch the refusal (the inherited `Mapping.get` catches only `KeyError` and would have propagated); and **`.pop(key, default)` returns its default**, through the same explicitly-overridden mechanism, added one round later. Rewriting `.get()` or `.pop(key, default)` call sites defensively is pointless work. **Two neighbouring routes do raise, and the divergence is deliberate rather than an oversight.** The **bare** `.pop(key)` — with no default — raises `StoreKeyError` for an illegal key, because widening the fix to the bare form would make `pop` the one mapping route that answers a miss with `None`. And **`.setdefault(key, default)` raises** `StoreKeyError`, because `setdefault` is a **write** route, and refusing an illegal key on a write is this change's entire point; the fact that its name and signature resemble `get`'s is not a reason to give it `get`'s behaviour. **`pop(key, default)` is a *restoration*, and it needs reading as one.** Before Phase 14 it returned its default; the first Phase 14 round made it raise, without deciding to and without saying so here; it returns its default again. If you read this note during that window and hardened your `pop` call sites with `except ValueError` (or `except StoreKeyError`), you are **not** broken by the restoration — but that handler is now dead code on the defaulting form, and it is still live on the bare form. **The full route list, all ten of it.** `key in store` (`__contains__`) answers `False`; `.get(key, default)` returns its default; `.pop(key, default)` returns its default; the bare `.pop(key)` raises; `store[key]` (`__getitem__`) raises; `store[key] = entry` (`__setitem__`) raises; `del store[key]` (`__delitem__`) builds no path and so has no key to refuse; `.setdefault(key, default)` raises, as a write route; `.update(other)` raises per key through `__setitem__`; `.clear()` empties the store or raises; and `.popitem()` is left as `MutableMapping` supplies it, its `KeyError` carrying the key on a non-empty store and carrying nothing on an empty one. The same list, with the per-route reasoning, is on the *Store key contract* page in `docs/source/LazyDiskCache.rst`. **A behaviour change on `pop`, and it affects a caller who was using it as a probe.** `pop(key, default)` on a key that is *tracked but whose payload cannot be loaded* — the state a reopened store is in when an artefact has gone missing from the cache directory — now **removes the key** and then returns the default. Previously it returned the default and left the key tracked, which meant a removal loop written as `while store: store.pop(k, None)` completed having removed nothing and having raised nothing. `dict.pop(k, d)` removes `k` when `k` is present, and this now matches it. **If you used the defaulting `pop` as a non-mutating probe — asking whether a key is readable without disturbing the store — it is no longer one**; use `key in store` for the tracking question and `.get(key, default)` for the readability question, neither of which mutates. The bare `pop(key)` on the same key still raises, and now also removes, so the two forms differ only in what they return. `clear()` changed on the same grounds: it previously stopped at the first key whose payload could not be read and returned normally with the rest still tracked, and it now empties the store. A caller that relied on the surviving keys was relying on a defect. **What a removal leaves behind, stated because the word *remove* invites the wrong inference.** Every removal route — `pop` in either form, `del store[key]` and `clear()` — drops the key from the store's **in-memory** tracking and **leaves the entry's files on disk**: the `.npy`, the `.meta.json` and the `.dat` all stay. So a removed key that had been offloaded is re-adopted by the next `store[key]`, which falls back to the loader for any untracked key, and again by the rescan when the store is reopened; between the removal and that read, `key not in store` and `store[key]` both succeed, which is a `Mapping` contract violation. **This is a known limitation, not a designed guarantee.** If you are removing entries to reclaim disk, unlink the artefacts yourself — nothing in this release does it for you. The atomic drop-key-and-delete-files operation, with no partial effect on refusal and no stale finalizer able to delete a later entry created under the same key, is **STORE-04**, and it is not in this release. **A downstream that reads *removed* here and skips its own cleanup is the reader this clause exists for.** **`key` is positional-only on both `get` and `pop`.** This matches `dict.get` / `dict.pop` and is defensible, but it is a **silent narrowing** of `MutableMapping`'s signature on the very route this delta otherwise documents as a *restoration* — so a caller who wrote `store.pop(key=name, default=None)` against the pre-Phase-14 inherited method is broken by the fix rather than by the regression the fix restores, and would otherwise find nothing here to explain it. Measured, so you can match the traceback: `DiskBackedStore.pop() got some positional-only arguments passed as keyword arguments: 'key'`, and the same message with `DiskBackedStore.get()`. Both accessors are covered by this one clause: `get` was narrowed earlier in the phase and `pop` a round later. **Migration is one edit per call site** — drop the `key=` keyword: `store.pop(name, None)`, `store.get(name, None)`. **(4) `LazyDiskCacheConfig.extend_cache_path` now applies the same single-segment rule, so a multi-segment extension that worked before is refused.** `GSEGUtils.lazy_disk_cache.LazyDiskCacheConfig.extend_cache_path('sub/dir')` raises `StoreKeyError` with the `contains a path separator` clause. **Migration: replace one multi-segment call with chained single-segment calls** — `cfg.extend_cache_path('sub').extend_cache_path('dir')`. **Why this route is in scope at all, since a reader will otherwise read it as unrelated to store keys:** it is a configuration helper and no store exists when it runs, but it joins a caller-supplied string straight into the cache root — and STORE-02's guarantee that no path the store builds lands outside the cache directory is **hollow if the root those paths hang off can itself be walked upward**. The check runs before the join and regardless of whether `cache_path` is `None`. **The live call sites, so you can tell whether you are one of them — re-derived by grep across both downstream trees rather than carried over, because the list published in the previous draft was wrong in both directions:** it named a site that does not call this symbol and omitted three that do. **pc2img calls it directly in five places:** `strategies/interpolation.py:203` (a computed settings hash as the segment), `tiled_generator.py:71` and `:74` (inside `TIGSettings.extend_cache_paths`, once on the interpolation kwargs' configuration and once on the top-level one), and `tiled_generator.py:162` and `:171` (a per-tile id). A sixth pc2img occurrence, `image_cache/disk_backed_image_store.py:147`, is **prose in a docstring, not a call**. **iof3D calls it directly in two places:** `image_generation.py:157` and `:160`, both passing `pcd_id.stem`. **iof3D reaches it indirectly in one further place:** `v2/services/tiles.py:133` calls **`extend_cache_paths`** — plural — which is pc2img's `TIGSettings` wrapper, not this method; it is affected transitively through `tiled_generator.py:71,74`. That distinction is why the entry is marked indirect: a reader who greps `extend_cache_path` at that line finds a differently-named wrapper and concludes this note is stale. **These sites are enumerated here and nowhere else.** The method's own docstring names the repositories and the direct-versus-indirect split but deliberately does **not** repeat the line numbers, because line numbers in another repository drift and two copies of a drifting fact is exactly how the previous list went wrong — please do not helpfully re-duplicate them. Per-tile and per-session nesting is unaffected as long as each call supplies one segment; only a call that packed several segments into one string changes behaviour — but note that the widened refusal set in delta (1) also reaches this route, so a `path_ext` derived from a filename stem that ends in a dot or names a device is now refused here too. **Additive alongside the deltas:** `is_valid_store_key(key) -> bool` is published as the supported way to pre-check a composed name before it becomes a key, and the three path builders are promoted to public free functions `get_npy_path(cache_dir, key)` / `get_meta_path(cache_dir, key)` / `get_legacy_pickle_path(cache_dir, key)`. **(5) The whole artefact-naming override surface is withdrawn — and an earlier draft of this entry promised the opposite, so read this delta even if you migrated already.** **Removed, six private names, all on `DiskBackedStore`:** the three builder-alias methods `_get_npy_path`, `_get_meta_path` and `_get_legacy_pickle_path`, and the three artefact-suffix class attributes `_DBNDArrayFileExt`, `_DBNDArrayMetaExt` and `_LegacyPickleExt`. **The replacement for the methods** is the three module-level free functions this same entry publishes above — `get_npy_path(cache_dir, key)`, `get_meta_path(cache_dir, key)` and `get_legacy_pickle_path(cache_dir, key)`, each taking the cache directory explicitly rather than reading it off `self`, and each validating the key and verifying containment. **The attributes have no replacement, and that is stated plainly rather than left to be inferred: choosing the store's artefact extension is no longer possible.** There is no configuration hook, no protected constant and no supported subclass hook that restores it. **Three clauses of the previous draft were wrong, in three different ways, and each is named here rather than quietly softened — a deprecation note that reads plausible and is false is worse than none.** *“still work”* — false at this release: the methods are deleted. *“still support being overridden by a subclass”* — **false since commit `d83c22d`, that is, before the sentence was written.** That commit (*“feat: route every store path through the shared builders”*) removed the last library-internal call site, so nothing has routed through these methods since; measured across branches, `v0.5.3`, `origin/main` and `origin/develop/gsd` each carry **7** internal `self._get_*_path` call sites and this release carries **0**. An override has therefore been inert for the whole of Phase 14. **Deleting the methods does not break overriding — it makes visible a break that had already happened silently**, which is the part of this record most worth carrying away. *“precisely because they have measured live callers”* — the measurement was of **callers**, and these methods' callers are downstream, not internal, so the survey and the promise were about different populations. The deprecation warning's own advice, *call the free function instead*, is advice for a **caller**; for an **overrider** it restores nothing, because there is no longer any call for the override to intercept. That mismatch is the reasoning error that produced the false promise, and it is available to anyone writing the next deprecation. **This breaks a named live consumer, and it says so in those words: pc2img is broken by this release, and GSEGUtils ships clean while pc2img is knowingly red until it migrates.** It overrides both `_get_npy_path` and `_get_meta_path` in `image_cache/disk_backed_image_store.py` and calls them again in its `__delitem__` unlink path, and its test suite asserts on them. The call sites are enumerated **once**, in this repository's phase-14 `deferred-items.md` under **D-14-03**, and deliberately not duplicated here — two copies of a drifting fact is how delta (4)'s call-site list went wrong, and the same one-place discipline applies. iof3D is unaffected: a grep of its tree for all six names returns nothing. **The migration is a deletion, not a repoint, and that is a security claim so it is recorded as one with its numbers.** pc2img's overrides exist to add a containment guard the base store did not have. It now has one. Measured by pc2img's own absorption spike at commit `7893bda`, with its override removed and its Phase 5 escape corpus replayed through every disk-touching route: **all 12** escape-corpus cells are refused by this release's upstream lexical and resolved-containment layers, and **all 6** realistic feature names still round-trip. The downstream guard is therefore redundant and deletable rather than merely relocatable — but delete it deliberately, having read that measurement, not because this note made it syntactically necessary. **What the class has afterwards, stated with the qualifier it needs.** `DiskBackedStore` now holds **one codec artefact vocabulary** — `.npy`, `.meta.json` and `.pkl` — read from a single module by discovery, retrieval and writing alike, so a reopen scan can no longer advertise a key that a read cannot resolve. **It does not hold one artefact vocabulary unqualified, and do not read it that way:** the `.dat` memmap artefact that backs each entry keeps its own suffix vocabulary, still duplicated between the shared path module and `LazyDiskCache`, **unchanged by this release**. Unifying those is **STORE-08**. If you are reading this entry to decide what you no longer have to handle, the `.dat` path is not covered by this round's guarantees. **To find out which of your existing keys stop working**, run the read-only cache-directory scan in the *Store key contract* section of the documentation (`docs/source/LazyDiskCache.rst`); it imports `is_valid_store_key` rather than restating the rule, so it cannot disagree with what the library enforces, and it must exclude type-checker caches, virtual environments and `site-packages` — the `.meta.json` sidecar extension collides with mypy's own cache format and an unfiltered scan reports thousands of false hits. | Same as `migration_from_doc` (Phase 14 is the only origin; both baselines predate every part of this change). |
| BC-GSEG-007 | signature-shape | should-review | `GSEGUtils.lazy_disk_cache.DiskBackedStore` (`store`) | Phase 14 D-19 / STORE-01 / Plan 14-08 | **The `DiskBackedStore.store` property now returns a read-only view, and mutating through it raises.** It hands back `types.MappingProxyType(self._store)` annotated `Mapping[str, Optional[T]]`, where it previously handed back the live `dict` itself — `s.store is s._store` was `True`, so `s.store['../victim'] = entry` succeeded and put an illegal key into the store behind every validation route. **Reads are completely unaffected:** iteration, `len()`, `in`, `.keys()`, `.values()`, `.items()` and subscript reads all behave exactly as before, and the object still compares equal to the same mapping contents. Only mutation changes, and the exception type is **not uniform** — measured on CPython 3.12, the two *syntactic* forms `p[k] = v` and `del p[k]` raise `TypeError` (refused through type slots), while `p.update(...)`, `p.pop(...)`, `p.clear()` and `p.setdefault(...)` raise `AttributeError` (the proxy simply does not have the attribute). Both are refusals and neither mutates; do not write a handler that expects one type for all six. **Insertion is unchanged and goes where it always did:** `store[key] = entry` and `add_data_to_store(...)`, both of which validate the key. **This takes no deprecation cycle, on measured evidence rather than impatience:** a survey across `30_GSEGUtils`, `41_pchandler`, `/scratch/31_pc2img` and `/scratch/34_iof3d` found **zero write-through sites** — every usage of the accessor was a read — which is the same evidence principle that sealed the `cache_path` setter in BC-GSEG-006's second delta. **The previous draft contrasted this with the builder aliases, *“which get a full cycle precisely because they have measured live callers”*; that contrast is withdrawn — the aliases are removed in this release, for the reasons BC-GSEG-006 delta (5) sets out.** **The one known downstream fix, named precisely enough to apply without reading this repository's source:** pc2img's `DiskBackedImageStore.image_data` legacy alias (`src/pc2img/image_cache/disk_backed_image_store.py:160-163`) is a property that `return self.store` under the annotation `-> dict[str, DiskBackedImageData \| None]`. That annotation is now wrong and a strict type check will report it. **Widen it to the read-only form** — `-> Mapping[str, DiskBackedImageData \| None]`, importing `Mapping` from `collections.abc` — and change nothing else; the property's runtime behaviour is unchanged, since it was already returning whatever `store` returned. Any other downstream that restates the old `dict` return type in an annotation, a cast or a `TypeVar` bound needs the same one-line widening. | Same as `migration_from_doc` (Phase 14 is the only origin; both baselines predate this change). |

## Additive changes

| BC-ID | category | severity | affected_symbols | origin | migration_from_doc | migration_from_v044 |
|---|---|---|---|---|---|---|
| BC-GSEG-003 | additive-or-fixed | additive | `GSEGUtils.util.rad2deg`, `GSEGUtils.util.rad2gon`, `GSEGUtils.util.deg2rad`, `GSEGUtils.util.deg2gon`, `GSEGUtils.util.gon2rad`, `GSEGUtils.util.gon2deg` | Phase 1 D-16 / COUPLE-06 / Plan 01-04 | Six new public angle-conversion functions promoted from the previously private `_rad2deg` / `_deg2rad` / `_rad2gon` / `_gon2rad` / `_deg2gon` / `_gon2deg` aliases. Calling the underscore-prefixed names still works (deprecation shims) but emits `DeprecationWarning(stacklevel=2)` on call; the shims will be removed in v0.6 (one full release cycle). Migrate by switching imports to the public names. | Same as `migration_from_doc`. |
| BC-GSEG-005 | additive-or-fixed | additive | `GSEGUtils.lazy_disk_cache.LazyDiskCache` (`_convert_to_memmap` streaming path), `GSEGUtils.singleton.SingletonMeta.__call__` (lock-free fast path) | Phase 4 D-04..D-11 / PERF-04 + PERF-05 / Plans 04-04 + 04-05 | Both behaviour-preserving optimisations. `LazyDiskCache._convert_to_memmap` now streams chunked writes through `np.memmap` instead of materialising the full array in memory; the streaming path introduces `psutil` as a runtime dependency of GSEGUtils (previously dev-only). `SingletonMeta.__call__` uses double-checked locking — the fast path is GIL-dependent (the source's Notes block documents the 3.13t free-threaded caveat that's deferred to v2 per Phase 4 D-09). No downstream code changes required. | Same as `migration_from_doc`. |

## Internal & sweep changes

- **BC-GSEG-004 (internal/informational)** — `[project.dependencies] sphinx` swapped from a git+https commit pin to `sphinx ~= 8.2` (Phase 1 D-18 / DEP-02 / Plan 01-05). Resolver-level only; no API impact. Aligns the GSEGUtils sphinx pin with pchandler's `sphinx ~= 8.2` choice.
- Phase 1 D-14 — `validate_in_range` + `BaseArray._coerce_array` docstrings decoupled from pchandler-specific assumptions; observable behaviour identical. Commits `9ee480b`, `96dc8ed`.
- Phase 1 D-26 — `mypy.ini` `files = src, scripts, tests` → `files = src, tests`; dead `[mypy-GSEGUtils.*]` block removed. Type-checker scope only.
- Phase 1 D-08 + D-24 ruff sweep — `style(01-02a)`/`chore(01-02a)`/`docs(01-02a)` commits across `src/` (7404d1c, 25efc69, a482697, dfa780a, c36e314, df728a3, f10de42). NumPy-style docstrings now enforced; no public-surface change.
- Phase 2 D-14..D-17 — `Private :: Do Not Upload` classifier removed from `pyproject.toml`; `## Publication Policy` README section added. Cross-repo (same change in pchandler). No `twine` step active — structural absence.
- Phase 3 BUG-01 + BUG-02 — `DiskBackedNDArray` honours `NDArrayOperatorsMixin` (commit `96f7c3e`); `LazyDiskCache.offload` drops `_data` instead of writing `None` (commit `d4173e7`). Observable behaviour for in-memory consumers identical to pre-fix expectations; the prior buggy code paths raised `AttributeError` / produced corrupt offload state. Cross-references pchandler BC-PCH-005 for the consumer-facing `should-review` callout.
- Phase 6 D-18 — dead imports / commented code cleanup in `lazy_disk_cache.py:20`, `disk_backed_store.py:35`, `tests/test_base_arrays.py:256`. Pure hygiene.
- Step 2 untraceable-commit list (both baselines `doc@2eae789..refactor/gsd` and `v0.4.4..refactor/gsd` on `src/`): **zero untraceable commits.** Every `feat:`/`fix:`/`refactor:` commit in the non-trivial commit set (8 commits total per baseline, identical lists) cites a Phase N plan (`01-04`, `02-01`, `02-04`, `03-01`, `04-04`, `04-05`, `04-06a`). The Phase 0/1/2/4/6 verification chain (per-phase `*-VERIFICATION.md`) closed every change; no public-surface change slipped through unreferenced.

## Verifier (inline)

```python
#!/usr/bin/env python3
"""Phase 7 Plan 07-02 inline verifier.

Tier 1: AST walk of GSEGUtils public-surface files. Confirms every BC-GSEG
entry's top-level affected_symbols resolves against the declared union.

Tier 2: Runtime import of GSEGUtils.validators. Confirms the three callables
referenced by BC-GSEG-002 (normalize_uint8, normalize_uint16, linear_map_dtype)
accept the Phase 4 COUPLE-05 `source_range` keyword.

Run from the workspace root:
    python3.12 30_GSEGUtils/MIGRATION-v1.0.md   # not directly executable;
                                                # extract via awk per PLAN.md.
"""
from __future__ import annotations

import ast
import inspect
import sys
from importlib import import_module
from pathlib import Path

# Public-surface files (note: lazy_disk_cache has no .pyi — walk the .py;
# _extract_declared_names is pure-AST and agnostic to .py vs .pyi).
PUBLIC_SURFACE_FILES = [
    "30_GSEGUtils/src/GSEGUtils/__init__.pyi",
    "30_GSEGUtils/src/GSEGUtils/lazy_disk_cache/__init__.py",
]


def _extract_declared_names(pyi_text: str) -> set[str]:
    """Return the set of symbol names a ``__init__.pyi`` declares.

    Copied verbatim from 41_pchandler/tests/test_stubs_drift.py:31-61.
    """
    tree = ast.parse(pyi_text)
    declared: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                    value = node.value
                    if isinstance(value, ast.List):
                        for elt in value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                declared.add(elt.value)
        if isinstance(node, ast.ImportFrom) and node.level >= 1:
            for alias in node.names:
                declared.add(alias.asname or alias.name)
    return declared


BC_ENTRIES: list[dict[str, str | list[str]]] = [
    # `affected_symbols` lists the top-level names the AST union must cover.
    # Subpackage paths (e.g. `GSEGUtils.lazy_disk_cache`) are NOT re-exported
    # from the top-level `__init__.pyi.__all__` — they live in the second
    # public-surface file (`lazy_disk_cache/__init__.py`) which the AST walk
    # unions in. We only assert the classes / modules that those files declare.
    {
        "id": "BC-GSEG-001",
        "category": "on-disk-format",
        "severity": "should-review",
        "affected_symbols": ["DiskBackedStore", "LazyDiskCache"],
    },
    {
        "id": "BC-GSEG-002",
        "category": "signature-shape",
        "severity": "should-review",
        "affected_symbols": ["validators"],
    },
    {
        "id": "BC-GSEG-003",
        "category": "additive-or-fixed",
        "severity": "additive",
        "affected_symbols": ["util"],
    },
    {
        "id": "BC-GSEG-005",
        "category": "additive-or-fixed",
        "severity": "additive",
        "affected_symbols": ["singleton", "LazyDiskCache"],
    },
    # Phase 14 / STORE-01..03 + STORE-07. The six store-key-contract names below
    # are exactly the package's published key surface; the raising validator and
    # the two temporary-name builders are internal by decision (D-07) and are
    # therefore absent here — naming one would fail Tier 1, because the AST union
    # only covers what the two surface files declare.
    {
        "id": "BC-GSEG-006",
        "category": "signature-shape",
        "severity": "should-review",
        "affected_symbols": [
            "DiskBackedStore",
            "LazyDiskCache",
            # The configuration helper joined this list in the gap-closure
            # round: `extend_cache_path` gained a hard refusal (delta 4), and
            # its measured live downstream call sites make that a documented
            # break rather than an internal tightening. The count is NOT
            # restated here on purpose — delta (4) is the single place it is
            # enumerated, because the earlier duplicate of it drifted (WR-06).
            "LazyDiskCacheConfig",
            "StoreKeyError",
            "StoreContainmentError",
            "is_valid_store_key",
            "get_npy_path",
            "get_meta_path",
            "get_legacy_pickle_path",
        ],
    },
    # Phase 14 D-19 / Plan 14-08. Registered HERE and not only in the table
    # above: this list is hardcoded and the markdown table is never parsed, so
    # a table-only edit yields a verifier that reports success over an entry it
    # has never heard of. The success line prints len(BC_ENTRIES), which is the
    # observable signal that this registration took.
    {
        "id": "BC-GSEG-007",
        "category": "signature-shape",
        "severity": "should-review",
        "affected_symbols": ["DiskBackedStore"],
    },
]


def main() -> int:
    workspace_root = Path(__file__).resolve().parent if __file__ != "<stdin>" else Path.cwd()
    # When extracted to /tmp/07-02-verifier.py, Path(__file__) lives outside
    # the workspace — fall back to cwd which the PLAN.md sets to the workspace.
    if not (workspace_root / PUBLIC_SURFACE_FILES[0]).exists():
        workspace_root = Path.cwd()
    public_surface: set[str] = set()
    for rel in PUBLIC_SURFACE_FILES:
        path = workspace_root / rel
        if not path.exists():
            print(f"[fail] public-surface file missing: {path}", file=sys.stderr)
            return 1
        public_surface |= _extract_declared_names(path.read_text(encoding="utf-8"))

    failures: list[str] = []
    for entry in BC_ENTRIES:
        if entry["category"] == "surface-removed":
            for sym in entry["affected_symbols"]:
                if sym in public_surface:
                    failures.append(f"{entry['id']}: documented surface-removed but {sym!r} still present")
        else:
            for sym in entry["affected_symbols"]:
                if "." in sym:
                    continue
                if sym not in public_surface:
                    failures.append(f"{entry['id']}: symbol {sym!r} not in public surface")

    # Tier 2: runtime check of BC-GSEG-002 source_range kwarg.
    try:
        validators = import_module("GSEGUtils.validators")
    except ImportError as exc:
        failures.append(f"BC-GSEG-002 (Tier 2): cannot import GSEGUtils.validators: {exc}")
    else:
        for name in ("normalize_uint8", "normalize_uint16", "linear_map_dtype"):
            fn = getattr(validators, name, None)
            if fn is None:
                failures.append(f"BC-GSEG-002 (Tier 2): GSEGUtils.validators.{name} missing")
                continue
            try:
                params = inspect.signature(fn).parameters
            except (TypeError, ValueError) as exc:
                failures.append(f"BC-GSEG-002 (Tier 2): cannot read signature of {name}: {exc}")
                continue
            if "source_range" not in params:
                failures.append(
                    f"BC-GSEG-002 (Tier 2): {name!r} signature has no 'source_range' parameter "
                    f"(got {list(params)})"
                )

    if failures:
        for f in failures:
            print(f"[fail] {f}", file=sys.stderr)
        return 1
    print(f"[ok] verified {len(BC_ENTRIES)} BC-GSEG entries against public surface")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
