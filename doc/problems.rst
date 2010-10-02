==============================
Initial Value Problems
==============================

The principal objects in NodePy are ODE solvers.  The principal object
upon which a solver acts is an initial value problem.  Mathematically,
an initial value problem (IVP) consists of one or more ordinary 
differential equations and an initial condition:

    \\begin{align*}
    u'(t) & = F(u) & u(0) & = u_0.
    \\end{align*}


In NodePy, 
an initial value problem is an object with the following properties:

    * rhs(): The right-hand-side function; i.e. F where $u'=F(u)$.
    * u0:  The initial condition.
    * T:   The (default) final time of solution.

Optionally an IVP may possess the following:
    * exact(): a function that takes one argument (t) and returns
            the exact solution (Should we make this a function of
            u0 as well?)
    * dt0: The default initial timestep when a variable step size 
            integrator is used.
    * Any other problem-specific parameters.

The module ivp contains functions for loading a variety of initial
value problems.  For instance, the van der Pol oscillator problem
can be loaded as follows::

    >> from NodePy import ivp
    >> myivp = ivp.load_ivp('vdp')
