import os
import sys

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
    # "sphinx.ext.autosummary",  # For summary tables
    "sphinx.ext.intersphinx",
]

# External links
intersphinx_mapping = {'open3d': ('https://www.open3d.org/docs/release/', None),
                       'python': ('https://docs.python.org/3/', None),
                       'numpy [stable]': ('https://numpy.org/doc/stable/', None)}

# Keep as is
autodoc_class_signature = 'separated'       # This stops the big signature being displayed at the top
autoclass_content = 'both'
autosummary_generate = True

# # # Testing
autodoc_typehints_description_target = 'all'
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'


# Defaults
templates_path = ["_templates"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
