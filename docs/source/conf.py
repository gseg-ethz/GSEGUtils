import os
import sys


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
]

# Keep as is
autodoc_class_signature = 'separated'       # This stops the big signature being displayed at the top
autodoc_member_order = "groupwise"          # This clusters methods, classmethods, properties etc. together

# # Testing
# autoclass_content = 'class'
# autodoc_typehints_description_target = 'all'
# autodoc_typehints = 'both'
# autodoc_typehints_format = 'fully-qualified'

intersphinx_mapping = {'open3d': ('https://www.open3d.org/docs/release/', None),
                       'numpy [stable]': ('https://numpy.org/doc/stable/', None)}


templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
