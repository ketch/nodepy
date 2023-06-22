Planned Future Development
==========================

Development of NodePy is based on research needs.  The following
is a list of capabilities or features that would naturally fit
into the package but have not yet been implemented.

Multistep methods
-----------------

* Time-stepping for multistep methods
* Selection of startup method
* Variable step size multistep methods
* Properties of particular multistep + startup method combinations
* Adaptive step size and order selection

Runge-Kutta Methods
-------------------

* Time stepping for implicit methods
* Interpolants (dense output)
* Adaptive order (extrapolation and deferred correction)

PDEs
----

Many common semi-discretizations of PDEs will be implemented as
`ivp` objects.  Initially this will be implemented purely in Python and
limited to simple 1D PDEs (e.g. advection, diffusion), since
time-stepping for multi-dimensional or nonlinear PDEs will be too
slow in Python.  Eventually we plan to support wrapped Fortran and C
semi-discretizations.

Miscellaneous
-------------

 * Additional classes of multi-stage, multistep methods.
 * Analysis of geometric integrators.
 * Partitioned rooted trees, additive and partitioned RK methods.
 * Unit tests.
 * An automatically-generated encyclopedia of solvers.
