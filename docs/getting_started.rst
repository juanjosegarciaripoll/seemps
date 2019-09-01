.. _getting_started:

***************
Getting started
***************

.. _downloading-seemps:

Downloading SeeMPS
------------------

The SeeMPS project is available from at `GitHub <https://github.com/juanjosegarciaripoll/seemps>`_. You can clone it using git
from the command line::

  git clone https://github.com/juanjosegarciaripoll/seemps.git

This will create a copy of the repository with all the notebook files and the
processed Python library. Alternatively, you may download a ZIP file with a copy of the repository from `this link <https://github.com/juanjosegarciaripoll/seemps/archive/master.zip>`_.

.. _installing-python:

Installing Python and libraries
-------------------------------

SeeMPS is written in Python 3 using the following commonly used libraries

* `Numpy <https://numpy.org/>`_
* `Scipy <https://www.scipy.org/scipylib/index.html>`_
* `Matplotlib <https://matplotlib.org/>`_

For novice users and inexperienced programmers I recommend using the `Miniconda distribution <https://docs.conda.io/en/latest/miniconda.html>`_  to install Python portably and self contained in your home directory. This is a bare-bones, very lightweight distribution with nothing preinstalled. Once you have installed it, you can open the Anaconda prompt and type::

  conda install numpy scipy matplotlib

In order to read the notebooks, edit them or recreate the library, you also
need Jupyter or Jupyter Lab. I recommend the latter::

  conda install jupyterlab

.. _reading-the-docs:
  
Reading the documentation
-------------------------

The library is a self-contained distribution of Jupyter notebooks. Each of them
contains both the Python code as well as a lot of text, images and equations
explaining the underlying models and algorithms. If you are interested on this,
you should start from the root notebook and read in whatever order you feel like.
`nbviewer <https://nbviewer.jupyter.org/github/juanjosegarciaripoll/seemps/tree/master/>`_ provides a good interface for doing so.

Using the library
-----------------

There are two ways to use the library. The simplest and crudest one is to
create new notebooks in the root directory and work from there. That is highly
not recommended, but it may useful to experiment tweaking the algorithms and
routines in-place.

More generally, we recommend importing the `mps` library as a Python
module. You simply need to add::
  import sys
  sys.append('directory/of/seemps')

at the beginning of your code and then import the parts that you need, such as::
  from mps import CanonicalMPS
  from mpo import MPO
  import qft
  ...

.. _rebuilding-the-library:

Rebuilding the library
----------------------

If you have made any change to the notebooks, you can rebuild the Python codes
from the command line. On Windows, just type

  make clean
  make all

On Linux or MacOS, make sure that the utility `make` is installed and type
those same lines.
