================================
Classes of ODE solvers
================================

.. contents::

The basic object in NodePy is an ODE solver.  Several types of solvers are
supported, including linear multistep methods, Runge-Kutta methods, and
Two-step Runge-Kutta methods.  For each class, quite general methods are
supported and may be instantiated by specifying their coefficients.  
Many convenience functions are also provided for loading common methods
or common families of methods.  The Runge-Kutta method class also supports
some special low-storage RK method classes, integral deferred correction 
methods, and extrapolation methods.

.. _create_rkm:

Runge-Kutta methods
============================

A Runge-Kutta method is a one-step method that computes the next time
step solution as follows:

    \\begin{align*}
    y_i = & u^{n} + \\Delta t \\sum_{j=1}^{s} + a_{ij} f(y_j)) & (1\\le j \\le s) \\\\
    u^{n+1} = & u^{n} + \\Delta t \\sum_{j=1}^{s} b_j f(y_j).
    \\end{align*}

The simplest way to load a Runge-Kutta method is using the 
loadRKM function::

    >> from nodepy import runge_kutta_method as rk
    >> import numpy as np
    >> rk44=rk.loadRKM('RK44')
    Classical RK4

     0.000 |
     0.500 |  0.500
     0.500 |  0.000  0.500
     1.000 |  0.000  0.000  1.000
    _______|________________________________
           |  0.167  0.333  0.333  0.167

Many well-known methods are available through the loadRKM() function.
Additionally, several classes of methods are available through the
following functions:

  * Optimal strong stability preserving methods: SSPRK2(s), SSPRK3(s), SSPIRK2(), etc.
  * Integral deferred correction methods: 
    :mod:`DC(s) <nodepy.runge_kutta_method.DC>`

  * Extrapolation methods: 
    :mod:`extrap(s) <nodepy.runge_kutta_method.extrap>`
  * Runge-Kutta Chebyshev methods: RKC1(s), RKC2(s)

See the documentation of these functions for more details.

More generally, any Runge-Kutta method may be instantiated by providing 
its Butcher coefficients, $A$ and $b$::

    >> A=np.array([[0,0],[0.5,0]])
    >> b=np.array([0,1.])
    >> rk22=rk.RungeKuttaMethod(A,b)

Note that, because NumPy arrays are indexed from zero, the Butcher coefficient
$a_{21}$, for instance, corresponds to my_rk.a[1,0].
The abscissas $c$ are automatically set to the row sums of $A$ (this
implies that every stage has has stage order at least equal to 1).
Alternatively, a method may be specified in Shu-Osher form, by coefficient
arrays $\\alpha,\\beta$::

    >> rk22=rk.RungeKuttaMethod(alpha=alpha,beta=beta)

A separate subclass is provided for explicit Runge-Kutta methods: *ExplicitRungeKuttaMethod*.
If a method is explicit, it is important to instantiate it as an
ExplicitRungeKuttaMethod and not simply a RungeKuttaMethod, since the
latter class has significantly less functionality in NodePy.
Most significantly, time stepping is currently implemented for explicit methods, 
but not for implicit methods.

Composing Runge-Kutta methods
--------------------------------------------

Butcher has developed an elegant theory of the group structure of 
Runge-Kutta methods.  The Runge-Kutta methods form a group under the
operation of composition.  The multiplication operator has been 
overloaded so that multiplying two Runge-Kutta methods gives the
method corresponding to their composition, with equal timesteps.

It is also possible to compose methods with non-equal timesteps using the
compose() function.


.. _create_lmm:

Low-Storage Runge-Kutta methods
-----------------------------------------

  * 2R/3R methods, pairs
  * 2S/2Semb/2S*/3S* methods, pairs

.. automethod:: nodepy.low_storage_rk.LowStorageRungeKuttaMethod.__init__

Embedded Runge-Kutta Pairs
-----------------------------------------
An embedded Runge-Kutta Pair takes the form:

    \\begin{align*}
    y_i = & u^{n} + \\Delta t \\sum_{j=1}^{s} + a_{ij} f(y_j)) & (1\\le j \\le s) \\\\
    u^{n+1} = & u^{n} + \\Delta t \\sum_{j=1}^{s} b_j f(y_j) \\\\
    \\hat{u}^{n+1} = & u^{n} + \\Delta t \\sum_{j=1}^{s} \\hat{b}_j f(y_j).
    \\end{align*}

That is, both methods use the same intermediate stages $y_i$, but different
weights.  Typically the weights $\\hat{b}_j$ are chosen so that $\\hat{u}^{n+1}$
is accurate of order one less than the order of $u^{n+1}$.  Then their
difference can be used as an error estimate.



Linear Multistep methods
==================================
A linear multistep method computes the next solution value from the values
at several previous steps:

    `\alpha_k y_{n+k} + \alpha_{k-1} y_{n+k-1} + ... + \alpha_0 y_n
    = h ( \beta_k f_{n+k} + ... + \beta_0 f_n )`

Note that different conventions for numbering the coefficients exist;
the above form is used in NodePy.
Methods are automatically normalized so that $\\alpha_k=1$.

The follwing functions return linear multistep methods of some 
common types:

  * Adams-Bashforth methods: Adams_Bashforth(k)
  * Adams-Moulton methods: Adams_Moulton(k)
  * backward_difference_formula(k)
  * Optimal explicit SSP methods (elmm_ssp2(k))

In each case, the argument $k$ specifies the number of steps in the method.
Note that it is possible to generate methods for arbitrary $k$, but currently
for large $k$ there are large errors in the coefficients due to roundoff errors.
This begins to be significant at 7 steps.  However, members of these families
with many steps do not have good properties.

More generally, a linear multistep method can be instantiated by specifying
its coefficients $\\alpha,\\beta$::

    >> from nodepy import linear_multistep_method as lmm
    >> my_lmm=lmm.LinearMultistepMethod(alpha,beta)


.. _create_tsrkm:

Two-step Runge-Kutta methods
======================================

Two-step Runge-Kutta methods are a class of multi-stage multistep methods
that use two steps and (potentially) several stages.

.. automethod:: nodepy.twostep_runge_kutta_method.TwoStepRungeKuttaMethod.__init__

.. _create_lsrkm:

`blah <../../source/conf.py>`_
`<index.html>`_
