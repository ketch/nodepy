================================
Planned Future Development
================================

It is expected that new classes of methods will frequently be
implemented in NodePy, according to the research interests of
NodePy developers.
Several major new areas of functionality are also planned.

Short-term plans
================================

The following functions are planned for the next release (0.4):

    * Reducibility detection and automatic reduction for Runge-Kutta Methods
    * IVPs corresponding to finite difference semi-discretizations of simple PDEs


Long-term plans
================================

Items in *italics* are planned for implementation by version 1.0 at the latest.


Multistep methods
---------------------------

Much of the current functionality in NodePy is limited to one-step
methods.  We plan to implement these features also for multistep
methods, and also to implement functionality that is specific to
multistep methods:

    * Time-stepping for multistep methods
    * Selection of startup methods
    * Variable step size multistep methods
    * Adaptive step size and order selection

Runge-Kutta Methods
---------------------------
Additional planned functionality for Runge-Kutta methods includes:

    *Time stepping for implicit methods*
    * Interpolants (dense output)


PDEs
---------------------------

Many common semi-discretizations of PDEs will be implemented as
ivp objects.  Initially this will be implemented purely in Python and
limited to simple 1D PDEs (e.g. advection, diffusion), since 
time-stepping for multi-dimensional or nonlinear PDEs will be too
slow in Python.  Eventually we plan to support wrapped Fortran and C
semi-discretizations.

General Linear Methods
---------------------------
Additional classes of multi-stage, multistep methods will be supported
in the future.

Symplectic and Hamiltonian Methods
------------------------------------------------------
Support for analysis of geometric integrators.

Rooted Trees
---------------------------
Partitioned rooted trees and other similar abstract objects will be implemented.

Exact specification of coefficients
------------------------------------------------------

All classes will eventually be overloaded so that exact coefficients
can be given and, wherever possible, properties of the methods will 
be computed exactly.  We plan to use the SAGE Rational class for this.

Unit Testing
---------------------------

An extensive set of unit tests with broad code coverage will be implemented.


Encyclopedia of IVP solvers
---------------------------

We would like to use NodePy to compile an online encyclopedia of solvers.
