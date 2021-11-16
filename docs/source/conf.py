import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'Extending Moving Targets'
# noinspection PyShadowingBuiltins
copyright = '2021, Luca Giuliani, Fabrizio Detassis, Michele Lombardi'
author = 'Luca Giuliani, Fabrizio Detassis, Michele Lombardi'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.intersphinx',
]
autodoc_default_options = {
    'member-order': 'groupwise',
    'undoc-members': True,
    'inherited-members': True,
    'show-inheritance': True,
}
autoclass_content = 'both'

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'scikit-learn': ('https://scikit-learn.org/stable', None),
    'tensorflow': ('https://www.tensorflow.org/api_docs/python',
                   'https://raw.githubusercontent.com/GPflow/tensorflow-intersphinx/master/tf2_py_objects.inv')
}
