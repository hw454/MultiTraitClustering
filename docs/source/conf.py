# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
autodoc_mock_imports = ['checks', 'clustering_methods', 'data_processing',
                        'data_setup', 'multi_trait_clustering', 'plotting_funcs',
                        'string_funcs']

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MultiTraitClustering'
copyright = '2025, Hayley Wragg'
author = 'Hayley Wragg'
release = '0.1.23'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add napoleon to the extensions list
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = []

pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
