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

The public import surface of `GSEGUtils` is byte-for-byte stable from both `doc@2eae789` and `v0.4.4` through `refactor/gsd` HEAD. Every name in `30_GSEGUtils/src/GSEGUtils/__init__.pyi`'s `__all__` (`base_arrays`, `base_types`, `config`, `constants`, `generate_init_stubs`, `logging_setup`, `singleton`, `util`, `validators`, `__author__`, `__email__`, `__version__`, `version`, `__version_tuple__`, `version_tuple`) and every name in `30_GSEGUtils/src/GSEGUtils/lazy_disk_cache/__init__.py`'s `__all__` (`LazyDiskCache`, `LazyDiskCacheKw`, `LazyDiskCacheConfig`, `DiskBackedNDArray`, `DiskBackedStore`) continues to resolve at its documented import path. The §"Verifier (inline)" section ships a Tier 1 AST walk that asserts every `BC-GSEG-NNN` entry's top-level `affected_symbols` resolves against this public-surface union, plus a Tier 2 `inspect.signature` runtime check against `GSEGUtils.validators.normalize_uint8` / `normalize_uint16` / `linear_map_dtype` confirming the Phase 4 COUPLE-05 `source_range` keyword is present at HEAD. Per D-08, the table schema below carries TWO migration columns (`migration_from_doc`, `migration_from_v044`); when an entry applies to only one baseline, the inapplicable column carries the exact filler `no change from this baseline — already present` so downstream tooling (the Plan 07-04 dry-run simulator) handles the generic case uniformly. Across BC-GSEG-001..005 this filler does NOT appear in practice — all five documented changes apply to both baselines (Phase 0..6 changes are all post-`doc@2eae789` and post-`v0.4.4`) — but the convention is documented here so the simulator's parser remains agnostic.

## Breaking changes & behavior changes

| BC-ID | category | severity | affected_symbols | origin | migration_from_doc | migration_from_v044 |
|---|---|---|---|---|---|---|
| BC-GSEG-001 | on-disk-format | should-review | `GSEGUtils.lazy_disk_cache.DiskBackedStore`, `GSEGUtils.lazy_disk_cache.LazyDiskCache` (`__setstate__`, `offload`) | Phase 2 D-02..D-07 + D-18..D-21 / SEC-01 + FRAG-03 / Plans 02-01 + 02-04 | `DiskBackedStore` now persists arrays via `np.save` (`.npy`) + JSON sidecar (`.meta.json`) written atomically (`tmp + fsync + os.replace`); the legacy `pickle`-based codec is gone. Legacy `.pkl` cache files on disk are refused with `KeyError` + an INFO log entry — downstream code that materialised caches under `doc@2eae789` must re-materialise them via the upstream `DiskBackedStore(...)` factory. `LazyDiskCache.__setstate__` re-registers its weakref finalizer through the canonical `enable_purge()` path; round-tripping a pickled `LazyDiskCache` no longer leaks file handles. Cross-repo: this is the GSEGUtils half of pchandler's BC-PCH-005 (caching pathways that transit `DiskBackedStore`). | Same as `migration_from_doc` (Phase 2 swap landed after both baselines; the behaviour is identical for direct consumers). |
| BC-GSEG-002 | signature-shape | should-review | `GSEGUtils.validators.normalize_uint8`, `GSEGUtils.validators.normalize_uint16`, `GSEGUtils.validators.linear_map_dtype` | Phase 4 D-12..D-18 / COUPLE-05 / Plan 04-06 | All three callables now accept a keyword-only `source_range: tuple[float, float] = (0.0, 1.0)` parameter that locks the precise scaling envelope. Out-of-range floats clip-and-saturate silently to the integer dtype's range. NaN / Inf inputs now raise `ValueError` (previously they were propagated through the rescale, producing dtype-truncation garbage). Integer-typed input is unchanged. Migrate by passing `source_range=` explicitly at every call site — pchandler's call sites have already been migrated (cross-references the pchandler-side COUPLE-05 audit). | Same as `migration_from_doc` (Phase 4 is the only origin; no divergence between baselines). |

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
