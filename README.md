[![Build Status](https://travis-ci.org/ketch/nodepy.png)](https://travis-ci.org/ketch/nodepy)
[![Coverage Status](https://coveralls.io/repos/ketch/nodepy/badge.svg)](https://coveralls.io/r/ketch/nodepy)
[![Stories in Ready](https://badge.waffle.io/ketch/nodepy.png?label=ready&title=Ready)](https://waffle.io/ketch/nodepy)

[![](https://readthedocs.org/projects/nodepy/badge)](https://readthedocs.org/projects/nodepy/)
[![version status](https://pypip.in/v/nodepy/badge.png)](https://pypi.python.org/pypi/nodepy)
[![downloads](https://pypip.in/d/nodepy/badge.png)](https://pypi.python.org/pypi/nodepy)


# Installation

    pip install nodepy

This will automatically fetch dependencies also (numpy, matplotlib, sympy, scipy).  It will not fetch
cvxpy, which is an optional dependency (used only in a few specialized routines).

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

The development version can be obtained from

http://github.com/ketch/nodepy

# Citation

If you use NodePy in a published work, please cite it as follows:

    Ketcheson, D. I.  NodePy software version <version number>,
    http://github.com/ketch/nodepy/.

Please insert the version number that you used (currently 0.6).

# License

NodePy is distributed under the terms of the modified Berkeley Software
Distribution (BSD) license. 


# Funding

NodePy development has been supported by:

    * A U.S. Dept. of Energy Computational Science Graduate Fellowship
    * Grants from King Abdullah University of Science & Technology


