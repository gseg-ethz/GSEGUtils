import importlib.metadata
import os
import sys

# Phase 09.1 Plan 01 residual warnings: 208
# (after numpydantic fix; toc.not_included and PointCloudData/load_file duplicate-object
# warnings not applicable to GSEGUtils; the 208 are all nitpicky-mode ref.class/ref.func/
# ref.meth/ref.attr/ref.obj/ref.data surfaced cross-ref issues + some pre-existing
# duplicate object descriptions in lazy_disk_cache; Plan 09.1-03 will add
# nitpick_ignore_regex entries for genuinely unresolvable refs)
from sphinx.util import logging

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath("../../src"))


project = "GSEGUtils"
copyright = "2024, Nicholas Meyer"
author = "Nicholas Meyer"

# The documented version is read from the INSTALLED distribution metadata rather
# than from a literal that the release tooling rewrites in place
# (DESIGN-DECISIONS entry 87). The extra-release-file declaration is gone from
# release-please-config.json, so a generated release pull request now touches no
# documentation input at all — which is what lets every required status check
# short-circuit on release-artifact-only content, and it retires the
# version-stamping drift bug class rather than re-checking it.
#
# Display consequence, accepted rather than discovered later: on a non-release
# build there is no released number to show, so this renders whatever
# setuptools_scm derived for the tree the installed distribution was BUILT from.
# This repository sets version_scheme = "post-release" and local_scheme =
# "no-local-version", so the derived string carries no local-version suffix.
#
# CORRECTION (review finding CR-01, 2026-08-18): that derivation needs TAGS, and
# without them it does not fail — it DEGRADES, silently. From a shallow, tagless
# clone setuptools_scm emits a Python UserWarning and returns the fallback
# "0.0.post1". Sphinx's -W promotes SPHINX warnings, not Python ones, so such a
# build renders a wrong version and still reports success. This was not
# hypothetical: it is exactly what the CI docs job did, because it checked out at
# actions/checkout's default depth of 1. Reproduced by execution, not inferred.
# The tags are therefore kept reachable deliberately, and asserted rather than
# assumed: .github/workflows/ci.yml gives the docs checkout `fetch-depth: 0` and
# fails the build when the derived version is the fallback.
#
# Read the Docs was MEASURED rather than assumed: its `latest` build of the first
# post-unstamp commit (69e89a8, 2026-08-18) rendered "0.6.0.post2", so RTD's own
# clone does reach the tags and is NOT affected. That measurement covers the
# default branch's `latest` version; it does not promise the same for a release
# tag that ever falls outside RTD's clone depth.
version = importlib.metadata.version("GSEGUtils")
release = version


extensions = [
    "sphinx.ext.autodoc",  # For generating documentation from docstrings
    "sphinx.ext.napoleon",  # For Google-style and NumPy-style docstrings
    "sphinx.ext.autosummary",  # For summary tables
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
]

# Intersphinx Config
intersphinx_mapping = {
    "open3d": ("https://www.open3d.org/docs/release/", None),
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "numpydantic": ("https://numpydantic.readthedocs.io/en/latest/", None),
}

# General Config
python_use_unqualified_type_name = True  # False

# ======= Autodoc Config =========
autoclass_content = "both"  # 'both'
autodoc_class_signature = "separated"  # 'mixed' / 'separated
autodoc_member_order = "bysource"  # 'alphabetical'
autodoc_default_options = {"exclude-members": "model_config"}  # {}
autodoc_docstring_signature = True  # True
autodoc_mock_imports = []  # []
autodoc_typehints = "description"  # 'signature', 'init, 'both', 'none'
autodoc_typehints_description_target = "all"  # 'all', 'documented', 'documented_params'
autodoc_type_aliases = {"LowerStr": "str"}  # {}

# Phase 09.1 baseline: nitpicky mode so new self-inflicted broken refs fail the build.
# DOC-V2-01 (Phase 09.1-03): nitpick_ignore_regex filled after build-iterate.
nitpicky = True
nitpick_ignore_regex: list[tuple[str, str]] = [
    # ── GSEGUtils own type aliases ──────────────────────────────────────────────
    # These aliases are declared as py:attribute in the inventory (they are
    # TypeAlias/NDArray subtypes) but autodoc emits py:class cross-refs for them.
    # Role mismatch → unresolvable locally; RTD resolves fine via intersphinx.
    (r"py:class", r"(ArrayT|VectorT|IndexLike|SfNameT|LowerStr|DtypeDict)"),
    (r"py:class", r"Array_[A-Za-z0-9_]+_T"),
    (r"py:class", r"Vector_[A-Za-z0-9_]+_T"),
    (r"py:class", r"ArrayDtypes"),
    (r"py:data", r"ArrayDtypes"),
    # ── deliberately unpublished lazy_disk_cache internals ──────────────────────
    # Plan 15-09 added `_register_entry` and `_reconcile_artefact_targets`, which
    # the lazy_disk_cache page renders because its automodule carries
    # `:private-members:`. Their docstrings cross-reference symbols that are
    # intentionally NOT part of the documented surface, so the refs can never
    # resolve and the roles are prose pointers for a source reader rather than
    # broken links to fix:
    #   * `paths._assert_contained` / `paths._assert_write_contained` — the
    #     containment helpers, which `lazy_disk_cache/__init__.py`'s own module
    #     docstring records as "deliberately unpublished"; no page documents
    #     `GSEGUtils.lazy_disk_cache.paths`.
    #   * `__setitem__` — a real `DiskBackedStore` mapping route, but a dunder,
    #     and this automodule does not enable `:special-members:`.
    #   * `_LAZY_DISK_CACHE_CLASS_REGISTRY` / `_resolve_lazy_disk_cache_class` —
    #     module-private registry internals behind the published
    #     `register_lazy_disk_cache_class` extension point.
    # Scoped to these exact names rather than to a private-name pattern, so a
    # future broken ref to some OTHER private symbol still fails the build.
    (r"py:func", r"paths\._assert(_write)?_contained"),
    (r"py:meth", r"__setitem__"),
    (r"py:data", r"_LAZY_DISK_CACHE_CLASS_REGISTRY"),
    (r"py:func", r"_resolve_lazy_disk_cache_class"),
    # `register_lazy_disk_cache_class` is the one entry here that IS published
    # (it is in `lazy_disk_cache.__all__`); it is simply absent from this page's
    # automodule `:members:` allow-list, which predates plan 15-09. Ignored
    # rather than documented, because adding it widens the published docs
    # surface — a product decision, not a build fix, and it drags two of its own
    # private refs in with it. Adding it to `:members:` is the alternative if a
    # later phase wants the extension point documented.
    (r"py:func", r"register_lazy_disk_cache_class"),
    # ── numpydantic vendor internals ────────────────────────────────────────────
    # numpydantic re-exports NDArray from its vendor copy of nptyping;
    # the vendor path is never in any inventory.
    (r"py:.*", r"numpydantic\.vendor\..*"),
    (r"py:class", r"numpydantic\.NDArray"),
    (r"py:class", r"NDArray"),
    # numpydantic 1.10 made NDArray a runtime Protocol; autodoc now renders the
    # fully-parametrised numpydantic.ndarray.NDArray[...] form (submodule-qualified,
    # expanded dtype tuple) for raw NDArray-typed members, which no inventory exports.
    # Pre-1.10 rendered the shorter numpydantic.NDArray form covered above.
    (r"py:class", r"numpydantic\.ndarray\.NDArray.*"),
    # ── numpy internal / private types ─────────────────────────────────────────
    # numpy._typing is not part of the public API and not in the numpy inventory.
    (r"py:class", r"numpy\._typing\..*"),
    (r"py:class", r"npt\.DtypeLike"),
    (r"py:class", r"numpy\.typing\.DTypeLike"),
    (r"py:class", r"np\.\w+"),
    (r"py:class", r"numpy\.\w+"),  # full numpy.XXX forms (e.g. numpy.int32)
    # numpy concrete types rendered as "<class 'numpy.float32'>" etc.
    (r"py:class", r"<class '.*'>"),
    # numpy functions not indexed in local numpy inventory (network-restricted)
    (r"py:func", r"numpy\.iinfo"),
    (r"py:func", r"numpy\..*"),
    # ── pydantic internals ──────────────────────────────────────────────────────
    # pydantic v2 internal names not in the public intersphinx inventory
    (r"py:class", r"(Validator|PydanticDataclass|Factory)"),
    (r"py:class", r"GSEGUtils\.lazy_disk_cache\.disk_backed_store\.Factory"),
    # ── Python stdlib (network-restricted) ─────────────────────────────────────
    # weakref.finalize exists in stdlib but inventory isn't cached locally
    (r"py:func", r"weakref\.finalize"),
    (r"py:obj", r"typing\.T"),
    # Generic TypeVar T used in DiskBackedStore[T: LazyDiskCache]
    (r"py:class", r"T"),
    # ── Private helpers (not in __all__, not in inventory) ──────────────────────
    (r"py:func", r"_normalize_base"),
    (r"py:meth", r"_coerce_array"),
    # Phase 14 (STORE-07): the store key contract publishes six names —
    # is_valid_store_key, StoreKeyError, StoreContainmentError and the three
    # get_*_path builders — all of which DO resolve and are deliberately NOT
    # ignored here. The entries below cover references that are unresolvable by
    # construction; none of them hides a broken published name.
    #
    # Every one of them is written to its OWN target rather than to a module
    # prefix, and that is deliberate (WR-08). A `nitpick_ignore_regex` entry
    # cannot fail: an over-broad pattern silences real breakage forever and
    # nothing reports it, so the only check is the pattern against its comment.
    # A submodule-wide pattern here would disable the strict build for a module
    # whose surface is still growing.
    #
    # `validate_store_key` is the raising validator behind the published
    # predicate. D-07 publishes "a predicate ... alongside the INTERNAL raising
    # validator", so it is absent from the package __all__ and from the docs
    # member allowlist by decision; its docstring in paths.py cross-references
    # it and cannot resolve it. Same class as _normalize_base above.
    (r"py:func", r"validate_store_key$"),
    # The three deprecating builder wrappers on DiskBackedStore (D-15) point
    # their explicit link target at the defining module rather than at the
    # published re-export, e.g.
    # :func:`GSEGUtils.lazy_disk_cache.get_npy_path <...paths.get_npy_path>`.
    # The displayed name is the published one and resolves; the submodule-
    # qualified target does not, because `paths` is documented through the
    # package, not on a page of its own. Repointing those targets at the
    # published names belongs to whoever next edits disk_backed_store.py — and
    # this entry then expires with them, which a module-wide pattern would not.
    # Anchored to exactly the three targets named above: get_npy_path,
    # get_meta_path and get_legacy_pickle_path, and nothing else in `paths`.
    (r"py:func", r"GSEGUtils\.lazy_disk_cache\.paths\.get_(npy|meta|legacy_pickle)_path"),
    # `_assert_contained` is the private containment check. It is referenced by
    # name from `_store_entry`'s docstring and is not a documented member — the
    # same class as `_normalize_base` and `validate_store_key` above, and it
    # needs its own entry now that the pattern above no longer blankets the
    # module. It was previously swallowed by that blanket, which is the finding.
    (r"py:func", r"GSEGUtils\.lazy_disk_cache\.paths\._assert_contained"),
    # ── Stdlib short forms and deprecated typing aliases ───────────────────────
    # `pathlib.Path` is in the python inventory; the bare short form autodoc
    # renders under autodoc_typehints_format="short" is not. Surfaced by the
    # Phase 14 path builders and the three deprecating wrappers, all of which
    # take and return a Path.
    (r"py:class", r"Path$"),
    # `typing.Mapping` is the deprecated alias of collections.abc.Mapping and
    # carries no method entries in the python inventory.
    (r"py:meth", r"typing\.Mapping\.get$"),
    # `typing.MutableMapping` is the same case one interface down. Surfaced by
    # Plan 14-14's `pop` docstring, which contrasts the overridden `pop` with the
    # inherited `setdefault` (D-25) and links the latter to the ABC that defines
    # it. Anchored to that one method, not to the class, so a future reference to
    # a *different* MutableMapping method still has to resolve or be justified.
    (r"py:meth", r"typing\.MutableMapping\.setdefault$"),
    # The justification the entry above asks for, one method along. Plan 14-17's
    # `clear` docstring (D-29) records why `popitem` is deliberately left
    # inherited, and links it to the ABC that supplies it for exactly the reason
    # the `setdefault` reference exists: a reader must be able to see that the
    # neighbour is a decision rather than an oversight. Same deprecated-alias
    # cause, same per-method anchoring — not widened to the class.
    (r"py:meth", r"typing\.MutableMapping\.popitem$"),
    # dunder methods — Sphinx can't resolve short-form bare `__array__` etc.
    # `__setstate__` and `__delitem__` joined the list in Phase 14: the pickle
    # protocol hook is referenced from `cache_dir`'s docstring (Plan 14-14,
    # § WR-03) and `__delitem__` from `pop`'s (D-25). Both are real methods on
    # `DiskBackedStore`; neither is in the docs member allowlist, so neither has
    # a target to resolve to — the same class as the entries above, and each is
    # named exactly rather than covered by a `__.*__` pattern.
    (r"py:meth", r"(__array__|__getitem__|__array_interface__|__setstate__|__delitem__)"),
    # `_POP_DEFAULT_MISSING` is the module-level sentinel distinguishing "no
    # default supplied" from `default=None` in `pop` (D-25). It is private by
    # construction — the whole point of a sentinel is that no caller can name it
    # — so it is absent from `__all__` and from the member allowlist, exactly
    # like `validate_store_key` and `_normalize_base` above. `pop`'s docstring
    # names it because a reader needs to know a sentinel exists.
    (r"py:data", r"_POP_DEFAULT_MISSING$"),
    (r"py:attr", r"(__array_interface__|H)"),
    # ── autodoc_preserve_defaults false positives ───────────────────────────────
    # autodoc_preserve_defaults=True exposes keyword-only, default=..., optional
    # and numeric literals as apparent cross-ref targets.
    (r"py:class", r"optional"),
    (r"py:class", r"keyword-only"),
    (r"py:class", r"default.*"),
    (r"py:class", r"\d+\.\d+"),  # numeric literals like 0.0, 1.0
]
autodoc_typehints_format = "short"  # 'short'
autodoc_preserve_defaults = True  # False
autodoc_use_type_comments = True  # True
autodoc_warningiserror = True  # True
autodoc_inherit_docstrings = True  # True

autosummary_generate = True

extlinks = {
    "npd_docs": ("", None),
}

# Defaults
templates_path = ["_templates"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_extra_path = [r"../../LICENSE"]


redirects = {"index.html": "about.html"}


def setup(app):
    app.add_css_file("gseg_utils_theme.css")


rst_epilog = """
.. |NDArray| replace:: :external+numpydantic:py:class:`NDArray <numpydantic.ndarray.NDArray>`
"""
