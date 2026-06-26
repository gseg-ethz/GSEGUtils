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

# Automatic version updates via release-please
version = "0.5.2"  # x-release-please-version
release = "0.5.2"  # x-release-please-version


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
    # dunder methods — Sphinx can't resolve short-form bare `__array__` etc.
    (r"py:meth", r"(__array__|__getitem__|__array_interface__)"),
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
