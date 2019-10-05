
.. _changes:

Changes since release 0.7
=========================

- Comppute E-polynomial for RK methods
- Now possible to integrate complex solutions
- Many new specific ODE methods added
- Fixed some bugs in generation of trees and order conditions of very high order
- Added some new example notebooks

What's new in version 0.7
=========================
*Released November 29, 2016*

- Support for Python 3 (thanks to Github user @alexfikl)
- Dense output for Runge-Kutta methods.
- Removal of a circular dependency.

What's new in Version 0.6.1
===========================
*Released May 14, 2015*

- Two algorithms for computing optimal downwind perturbations of Runge-Kutta methods.  One relies on CVXPY for solving linear programs.
- The Numipedia project has been moved to its own repository: https://github.com/ketch/numipedia
- Many new doctests; >80% test coverage.
- Many improvements to the two-step RK module, including stability region plots for arbitrary methods.
- Three-step RK methods removed from master (because most of the module was not working).
- Pretty-printing of linear multistep methods.
- New methods:
  - Several very-high-order RK methods
  - Some singly diagonally-implicit RK methods
  - Nystrom and Milne-Simpson families of multistep methods
- load_ivp() works similarly to load_RKM() (returns a dictionary).
- Improved computation of maximum linearly stable step sizes for semi-discretizations.
- Many bug fixes.

What's new in Version 0.6
==========================
Version 0.6 is a relatively small update, which includes the following:

- Computation of optimal perturbations (splittings) of Runge-Kutta methods
- Additive linear multistep methods
- More accurate calculation of imaginary stability intervals
- Rational coefficients for more of the built-in RK methods
- Faster computation of stability polynomials
- More general deferred correction methods
- Fixed major bug in deferred correction method construction
- Continuous integration via Travis-CI
- Added information on citing nodepy
- Corrections to the documentation
- Updates for compatitibility with sympy 0.7.6
- Fixed bug involving non-existence of alphahat attribute
- minor bug fixes



What's new in Version 0.5
==========================
*Released: Nov. 4, 2013*

Version 0.5 is a relatively small update, which includes the following:

* More Runge-Kutta methods available in rk.loadRKM(), including the 8(7) Prince-Dormand pair
* Lots of functionality and improvements for studying internal stability of RK methods
* Shu-Osher arrays used to construct an RK method are now stored and (by default) used for timestepping
* Ability to compute the effective order of a RK method
* More accurate computation of stability region intervals
* Use exact arithmetic (sympy) in many more functions
* Generation of Fortran code for order conditions
* Refactoring of how embedded Runge-Kutta pairs and low-storage methods are represented internally
* Plotting functions return a figure handle
* Better pretty-printing of RK methods with exact coefficients
* Updates for compatibility with sympy 0.7.3
* Improved reducibility for RK pairs
* More initial value problems
* Several bug fixes
* Automated testing with Travis

What's new in Version 0.4
==========================
*Released: Aug. 28, 2012*

Version 0.4 of NodePy inclues numerous bug fixes and new features.
The most significant new feature is the use of exact arithmetic for
construction and analysis of many methods, using SymPy.  Because exact
arithmetic can be slow, NodePy automatically switches to floating point
arithmetic for some operations, such as numerical integration of initial value
problems.  If you find operations that seem excessively slow let me know.
You can always revert to floating-point representation of a method by
using method.__num__().

Other new features and fixes include:

    * Improvements to linear multistep methods:
        * Stability region plotting
        * Zero-stability
        * `A(\alpha)`-stability angles
    * Automatic selection of plotting region for stability region plots
    * Code base now hosted on Github (github.com/ketch/nodepy)
    * Documentation corrections
    * Use MathJax (instead of jsMath) in docs
    * Much greater docstring coverage
    * Many more examples in docs (can be run as doctests)
        * For example, 95 doctests covering 25 items in runge_kutta_method.py
    * Extrapolation methods based on GBS (midpoint method) -- thanks to Umair bin Waheed
    * Construction of simple linear finite difference matrices
    * Analysis of the potential for parallelism in Runge-Kutta methods
        * Number of sequentially-dependent stages
        * Plotting of stage dependency graph
    * Automatic reduction of reducible Runge-Kutta methods
    * A heuristic method for possibly-optimal splittings of Runge-Kutta methods
      into upwind/downwind parts
    * Fix bugs in computation of stability intervals
    * Fix bugs in stability region plotting
    * New examples in nodepy/examples/
    * Spectral difference matrices for linear advection -- thanks to Matteo Parsani


