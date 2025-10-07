# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../python/lorann"))

project = "LoRANN"
copyright = "2025, Elias Jääsaari"
author = "Elias Jääsaari"
release = "0.4.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_math_dollar",
    "sphinx_autodoc_typehints",
    "breathe",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

breathe_projects = {"LoRANN": "../doxygen/xml"}
breathe_default_project = "LoRANN"

autoclass_content = "both"
autodoc_mock_imports = ["lorannlib"]
autodoc_default_flags = ["members"]
autodoc_member_order = "bysource"
autodoc_typehints = "both"
autosummary_generate = True

math_dollar_inline_macros = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_add_permalinks = ""
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
