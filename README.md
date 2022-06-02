# NodePy: A package for the analysis of numerical ODE solvers

[![Build Status](https://travis-ci.com/ketch/nodepy.png)](https://travis-ci.com/ketch/nodepy)
[![Coverage Status](https://coveralls.io/repos/github/ketch/nodepy/badge.svg?branch=master)](https://coveralls.io/github/ketch/nodepy?branch=master)
[![codecov.io](https://codecov.io/github/ketch/nodepy/coverage.svg?branch=master)](https://codecov.io/github/ketch/nodepy?branch=master)
[![](https://readthedocs.org/projects/nodepy/badge)](https://readthedocs.org/projects/nodepy/)
<a href="https://pypi.org/project/nodepy/"><img alt="PyPI" src="https://img.shields.io/pypi/v/nodepy"></a>
<a href="https://pepy.tech/project/nodepy"><img alt="Downloads" src="https://pepy.tech/badge/nodepy"></a>
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-success.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4275157.svg)](https://doi.org/10.5281/zenodo.4275157)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02515/status.svg)](https://doi.org/10.21105/joss.02515)


# Installation
NodePy requires Python 3.5 or later.  To install with pip, do:

    pip install nodepy

This will automatically fetch dependencies also.  It will not fetch
optional dependencies, which include networkx, cvxpy and scipy (that are used
only in a few specialized routines and/or examples).  The optional dependencies
can be installed with `pip`.

# Overview

NodePy (Numerical ODEs in Python) is a Python package for designing, analyzing,
and testing numerical methods for initial value ODEs. Its development was
motivated by my own research in time integration methods for PDEs. I found that
I was frequently repeating tasks that could be automated and integrated.
Initially I developed a collection of MATLAB scripts, but this became unwieldy
due to the large number of files that were necessary and the more limited
capability for code reuse.

NodePy represents an object-oriented approach, in which the basic object is a
numerical ODE solver. The idea is to design a laboratory for such methods in
the same sense that MATLAB is a laboratory for matrices.

Documentation can be found online at

http://nodepy.readthedocs.org/en/latest/

To get started, you can also have a look at the `examples` folder,
beginning with an [introduction as Jupyter notebook](examples/Introduction%20to%20NodePy.ipynb).

The development version can be obtained from

http://github.com/ketch/nodepy

# Citation

If you use NodePy in a published work, please cite it as follows:

    Ketcheson, D. I.  NodePy software version <version number>,
    http://github.com/ketch/nodepy/.

Please insert the version number that you used.

# Support

If you encounter an error or need help, please [raise an issue](https://github.com/ketch/nodepy/issues).

# Contributing

Contributions of new features or other improvements are very welcome!  Please
[submit a pull request](https://github.com/ketch/nodepy/pulls) or contact the authors.

# License

NodePy is distributed under the terms of the [modified Berkeley Software
Distribution (BSD) license](LICENSE.txt).


# Funding

NodePy development has been supported by:

* A U.S. Dept. of Energy Computational Science Graduate Fellowship
* Grants from King Abdullah University of Science & Technology


