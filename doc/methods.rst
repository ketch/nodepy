Classes of ODE solvers
======================

The basic object in NodePy is an ODE solver.  Several types of solvers are
supported, including linear multistep methods, Runge-Kutta methods, and
Two-step Runge-Kutta methods.  For each class, individual methods
may be instantiated by specifying their coefficients.
Many convenience functions are also provided for loading common methods
or common families of methods.  The Runge-Kutta method class also supports
some special low-storage RK method classes, integral deferred correction
methods, and extrapolation methods.

.. toctree::

    rkm
    lmm
    tsrkm
