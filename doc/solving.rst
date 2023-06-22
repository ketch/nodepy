Testing Methods: Solving Initial Value Problems
===============================================

In addition to directly analyzing solver properties, NodePy also facilitates
the testing of solvers through application to problems of interest.
Furthermore, NodePy includes routines for automatically running sets of
tests to compare the practical convergence or efficiency of various solvers
for given problem(s).


.. contents::


.. _ivp:

Initial Value Problems
======================

The principal objects in NodePy are ODE solvers.  The object
upon which a solver acts is an initial value problem.  Mathematically,
an initial value problem (IVP) consists of one or more ordinary
differential equations and an initial condition:

    \\begin{align*}
    u'(t) & = F(u) & \\
    u(0) & = u_0.
    \\end{align*}

In NodePy,
an initial value problem is an object with the following properties:

    * *rhs()*: The right-hand-side function; i.e. F where `u(t)'=F(u)`.
    * *u0*:  The initial condition.
    * *T*:   The (default) final time of solution.

Optionally an IVP may possess the following:
    * *exact()*: a function that takes one argument `t` and returns the exact
      solution (Should we make this a function of `u_0` as well?)
    * *dt0*: The default initial timestep when a variable step size integrator
      is used.
    * Any other problem-specific parameters.

The module ivp contains functions for loading a variety of initial
value problems.  For instance, the van der Pol oscillator problem
can be loaded as follows::

    >> from NodePy import ivp
    >> myivp = ivp.load_ivp('vdp')


Instantiation
-------------

.. automethod:: nodepy.ivp.detest


Solving Initial Value Problems
==============================

Any ODE solver object in NodePy can be used to solve an initial value
problem simply by calling the solver with an initial value problem object
as argument::

    >> t,u = rk44(my_ivp)


.. automethod:: nodepy.ode_solver.ODESolver.__call__


Convergence Testing
-------------------

.. automethod:: nodepy.convergence.ctest

Performance testing with automatic step-size control
----------------------------------------------------

.. automethod:: nodepy.convergence.ptest

