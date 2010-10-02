================
NodePy Overview
================
NodePy (Numerical ODEs in Python) is a Python package for designing,
analyzing, and testing numerical methods for initial value ODEs.

Classes of Numerical ODE Solvers
================================

NodePy includes classes for the following types of methods:
    - :ref:`Linear Multistep Methods <create_lmm>`
    - :ref:`Runge-Kutta Methods <create_rkm>`
    - :ref:`Two-step Runge-Kutta Methods <create_tsrkm>`
    - :ref:`Low-storage Runge-Kutta Methods <create_lsrkm>`


Analysis of Methods
===================

NodePy includes functions for analyzing many properties, including:
    - Stability:
        - Absolute stability (e.g., plot the region of absolute stability)
        - Strong stability preservation
    - Accuracy
        - Order of accuracy
        - Error coefficients
        - Generation of Python and MATLAB code for checking order conditions
          using either the Butcher's approach or Albrecht's approach.


Testing Methods
======================

NodePy includes implementation of the actual time-stepping algorithms
for the various classes of methods.  A wide range of initial value
ODEs can be loaded, including the DETEST suite of problems.
Arbitrary initial value problems
can be instantiated and solved simply by calling a method with the
initial value problem as argument.  For methods with error estimates,
adaptive time-stepping can be used based on a specified error tolerance.
NodePy also includes automated functions for convergence testing.

In the future, NodePy will also support solving semi-discretizations
of initial boundary value PDEs.


Miscellaneous Features
======================

    - Plot and compute products on rooted trees
    - Plot the order star of a Runge-Kutta method
