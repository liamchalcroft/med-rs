# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------

project = 'medrs'
copyright = f'{datetime.now().year}, Liam Chalcroft'
author = 'Liam Chalcroft'
version = '0.1.2'
release = '0.1.2'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
    'sphinx_design',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.extlinks',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**/.pytest_cache',
    '**/__pycache__',
    '**/target',
]

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# The master toctree document.
master_doc = 'index'

# -- Options for autodoc extension ------------------------------------------

# autodoc default options
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '_weakref_,_dict,_hash,_module,_weakref_,_weakvalueweakref_,_weakrefproxy_,_finalizer,_delattr,_setattr,_format_,_make_,_repr_,_str_,_bytes_,_lt_,_le_,_gt_,_ge_,_eq_,_ne_,_hash_,_bool_,_neg_,_pos_,_abs_,_invert_,_round_,_floor_,_ceil_,_trunc_,_add_,_sub_,_mul_,_div_,_mod_,_divmod_,_pow_,_lshift_,_rshift_,_and_,_or_,_xor_,_contains_,_getitem_,_setitem_,_delitem_,_len_,_iter_,_reversed_,_enter_,_exit_,_new_,_init_,_call_,_getattr_,_getattribute_,_setattr_,_delattr_,_dir_,_get_,_set_,_delete_,_instancecheck_,_subclasscheck_,_prepare_,_init_subclass_,_format_,_sizeof_,_class_getitem_,_class_getattr_,_class_setattr_,_class_delattr_'
}

# -- Options for Napoleon extension ------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None

# -- Options for intersphinx extension -----------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'nibabel': ('https://nipy.org/nibabel/', None),
    'monai': ('https://docs.monai.io/en/latest/', None),
}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for coverage extension ---------------------------------------

# Coverage options
coverage_show_missing_items = True

# -- Options for mathjax --------------------------------------------------

mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@2/es5/Tex-MML-AM_CHTML.js'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 4,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS
html_css_files = [
    'custom.css',
]

# Custom JS
html_js_files = [
    'custom.js',
]

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'medrsdoc'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{hyperref}
\usepackage{graphicx}
''',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'medrs.tex', 'medrs Documentation',
     author, 'manual'),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'medrs', 'medrs Documentation',
     [author], 1)
]

# -- Options for Texinfo output ------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'medrs', 'medrs Documentation',
     author, 'medrs', 'One line description of project.',
     'Miscellaneous'),
]

# -- Options for MyST parser ----------------------------------------------

# Enable MyST extensions
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "attrs_inline",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Register source parser for markdown files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for copybutton ----------------------------------------------

copybutton_prompt_text = r">>>|\.\.\.|\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}\.\.: "
copybutton_prompt_is_regexp = True

# -- Options for sphinx.ext.autosectionlabel -------------------------

autosectionlabel_prefix_document = True

# -- Options for sphinx.ext.extlinks ----------------------------------

extlinks = {
    'issue': ('https://github.com/liamchalcroft/med-rs/issues/%s', 'issue %s'),
    'pr': ('https://github.com/liamchalcroft/med-rs/pull/%s', 'PR %s'),
    'user': ('https://github.com/%s', '@%s'),
}

# -- Custom configuration -------------------------------------------

# Add custom roles
def setup(app):
    app.add_css_file('custom.css')
    app.add_js_file('custom.js')
