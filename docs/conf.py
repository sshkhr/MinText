# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MinText'
copyright = '2025, Shashank Shekhar'
author = 'Shashank Shekhar'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'nbsphinx',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Set the master document (required for older Sphinx versions)
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- nbsphinx configuration --------------------------------------------------
# Execute notebooks before conversion: 'always', 'never', 'auto' (default)
nbsphinx_execute = 'never'

# If True, the build process is continued even if an exception occurs
nbsphinx_allow_errors = True

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}

.. note::

   This page was generated from `{{ docname }}`__.
   
   __ https://github.com/sshkhr/MinText/blob/main/{{ docname }}
"""

# Create necessary folders for Sphinx
os.makedirs(os.path.join(os.path.dirname(__file__), '_static'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), '_templates'), exist_ok=True)

# Syntax highlighting
highlight_language = 'python3'