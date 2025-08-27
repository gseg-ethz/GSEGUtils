import os
import sys
import pdb

from sphinx.application import Sphinx

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
]

# Keep as is
autodoc_class_signature = 'separated'       # This stops the big signature being displayed at the top
# autodoc_member_order = "groupwise"          # This clusters methods, classmethods, properties etc. together

# # Testing
autoclass_content = 'both'
autodoc_typehints_description_target = 'all'
autodoc_typehints = 'description'
# autodoc_typehints_format = 'fully-qualified'

intersphinx_mapping = {'open3d': ('https://www.open3d.org/docs/release/', None),
                       'python': ('https://docs.python.org/3/', None),
                       'numpy [stable]': ('https://numpy.org/doc/stable/', None)}


templates_path = ["_templates"]
exclude_patterns = []

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'exclude-members': 'model_config',
    'ignore-module-all': False,
    'member-order': 'bysource',
    'class-doc-from': 'both'
}

autodoc_typehint_aliases = {
    "npt.DtypeLike": "npt.DtypeLike",
}
linkcheck_allowed_redirects = {}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
#
# def swap_uri_path(app: Sphinx, uri: str):
#     map_type_aliases: dict[str, str] = {
#         'Array_Float_T': 'ArrayT',
#         'Array_Integer_T': 'ArrayT',
#         'Array_SignedInteger_T': 'ArrayT',
#         'Array_UnsignedInteger_T': 'ArrayT',
#         'Array_Bool_T': 'ArrayT',
#         'Array_Float32_T': 'ArrayT',
#         'Array_Float64_T': 'ArrayT',
#         'Array_Int8_T': 'ArrayT',
#         'Array_Int16_T': 'ArrayT',
#         'Array_Int32_T': 'ArrayT',
#         'Array_Int64_T': 'ArrayT',
#         'Array_Uint8_T': 'ArrayT',
#         'Array_Uint16_T': 'ArrayT',
#         'Array_Uint32_T': 'ArrayT',
#
#         'Vector_Float_T': 'VectorT',
#         'Vector_Integer_T': 'VectorT',
#         'Vector_SignedInteger_T': 'VectorT',
#         'Vector_UnsignedInteger_T': 'VectorT',
#         'Vector_Bool_T': 'VectorT',
#         'Vector_Float32_T': 'VectorT',
#         'Vector_Float64_T': 'VectorT',
#         'Vector_Int8_T': 'VectorT',
#         'Vector_Int16_T': 'VectorT',
#         'Vector_Int32_T': 'VectorT',
#         'Vector_Int64_T': 'VectorT',
#         'Vector_Uint8_T': 'VectorT',
#         'Vector_Uint16_T': 'VectorT',
#         'Vector_Uint32_T': 'VectorT',
#
#         'Array_Nx3_Float_T': 'Array_Nx3_T',
#         'Array_Nx3_Integer_T': 'Array_Nx3_T',
#         'Array_Nx3_SignedInteger_T': 'Array_Nx3_T',
#         'Array_Nx3_UnsignedInteger_T': 'Array_Nx3_T',
#         'Array_Nx3_Bool_T': 'Array_Nx3_T',
#         'Array_Nx3_Float32_T': 'Array_Nx3_T',
#         'Array_Nx3_Float64_T': 'Array_Nx3_T',
#         'Array_Nx3_Int8_T': 'Array_Nx3_T',
#         'Array_Nx3_Int16_T': 'Array_Nx3_T',
#         'Array_Nx3_Int32_T': 'Array_Nx3_T',
#         'Array_Nx3_Int64_T': 'Array_Nx3_T',
#         'Array_Nx3_Uint8_T': 'Array_Nx3_T',
#         'Array_Nx3_Uint16_T': 'Array_Nx3_T',
#         'Array_Nx3_Uint32_T': 'Array_Nx3_T',
#
#         'Array_NxM_Float_T': 'Array_NxM_T',
#         'Array_NxM_Integer_T': 'Array_NxM_T',
#         'Array_NxM_SignedInteger_T': 'Array_NxM_T',
#         'Array_NxM_UnsignedInteger_T': 'Array_NxM_T',
#         'Array_NxM_Bool_T': 'Array_NxM_T',
#         'Array_NxM_Float32_T': 'Array_NxM_T',
#         'Array_NxM_Float64_T': 'Array_NxM_T',
#         'Array_NxM_Int8_T': 'Array_NxM_T',
#         'Array_NxM_Int16_T': 'Array_NxM_T',
#         'Array_NxM_Int32_T': 'Array_NxM_T',
#         'Array_NxM_Int64_T': 'Array_NxM_T',
#         'Array_NxM_Uint8_T': 'Array_NxM_T',
#         'Array_NxM_Uint16_T': 'Array_NxM_T',
#         'Array_NxM_Uint32_T': 'Array_NxM_T',
#
#         'Vector_3_Float_T': 'Vector_3_T',
#         'Vector_3_Integer_T': 'Vector_3_T',
#         'Vector_3_SignedInteger_T': 'Vector_3_T',
#         'Vector_3_UnsignedInteger_T': 'Vector_3_T',
#         'Vector_3_Bool_T': 'Vector_3_T',
#         'Vector_3_Float32_T': 'Vector_3_T',
#         'Vector_3_Float64_T': 'Vector_3_T',
#         'Vector_3_Int8_T': 'Vector_3_T',
#         'Vector_3_Int16_T': 'Vector_3_T',
#         'Vector_3_Int32_T': 'Vector_3_T',
#         'Vector_3_Int64_T': 'Vector_3_T',
#         'Vector_3_Uint8_T': 'Vector_3_T',
#         'Vector_3_Uint16_T': 'Vector_3_T',
#         'Vector_3_Uint32_T': 'Vector_3_T',
#
#         'Array_4x4_Float_T': 'Array_4x4_T',
#         'Array_4x4_Float32_T': 'Array_4x4_T',
#         'Array_4x4_Float64_T': 'Array_4x4_T',
#
#         'Array_3x3_Float_T': 'Array_3x3_T',
#         'Array_3x3_Float32_T': 'Array_3x3_T',
#         'Array_3x3_Float64_T': 'Array_3x3_T',
#
#         'Array_NxM_3_Uint8_T': 'Array_NxM_3_T',
#     }
#
#     for key, value in map_type_aliases.items():
#         if key in uri:
#             pdb.set_trace()
#             logger.info(f"key: {key} | uri: {uri}  |-> value: {value}")
#             return uri.replace(key, value)


# def _callback_html_page_context(app: Sphinx, pagename: str, templatename: str, context: dict, doctree: any):
#     pdb.set_trace()
#     if templatename != "page.html":
#         return
#
# def setup(app: Sphinx):
#     # app.connect("html-page-context", _callback_html_page_context)
#     app.connect("linkcheck-process-uri", swap_uri_path)
