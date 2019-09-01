# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'SeeMPS'
copyright = '2019, Juan Jose Garcia-Ripoll'
author = 'Juan Jose Garcia-Ripoll'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['recommonmark', # For Markdown
              'sphinx.ext.autodoc' # For using strings from classes/functions
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# We need this for autodoc to find the modules it will document
import sys
sys.path.append('../')

# -- Markdown parsing of docstrings -----------------------------------------

import commonmark
import re

is_line = re.compile(r'^[-]+$')
var_name = re.compile(r'(?P<name>[^ ]*(,[ ]*[^ ]*)*)[ ]*--[ ]*(?P<desc>.*)')

def docstring(app, what, name, obj, options, lines):
    text = []
    in_arg = False
    arg_txt = ''
    arg_name = None
    for l in lines:
        if arg_name:
            if len(l) and l[0]==' ':
                # Continues argument description
                arg_txt += ' ' + l.strip()
                continue
            text.append('* ' + arg_name + ': ' + arg_txt)
            arg_name = None
        if is_line.match(l):
            continue
        arg_name = var_name.match(l)
        if arg_name:
            arg_txt = arg_name.group('desc')
            arg_name = arg_name.group('name')
            print((arg_txt, arg_name))
        else:
            text.append(l)
    if arg_name:
        text.append('* ' + arg_name + ': ' + arg_txt)
    for l in text:
        print(l)
    md  = '\n'.join(text)
    ast = commonmark.Parser().parse(md)
    rst = commonmark.ReStructuredTextRenderer().render(ast)
    lines.clear()
    for line in rst.splitlines():
        lines.append(line)

def setup(app):
    app.connect('autodoc-process-docstring', docstring)


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bizstyle'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
