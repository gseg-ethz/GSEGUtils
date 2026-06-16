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
version = "1.0.1"  # x-release-please-version
release = "1.0.1"  # x-release-please-version


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
# nitpick_ignore_regex entries will be filled by Plan 09.1-03 after build-iterate.
nitpicky = True
nitpick_ignore_regex: list[tuple[str, str]] = []
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
