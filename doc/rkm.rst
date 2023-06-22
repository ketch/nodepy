.. contents::

.. _create_rkm:

Runge-Kutta methods
===================

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
arrays $\alpha,\beta$::

    >> rk22=rk.RungeKuttaMethod(alpha=alpha,beta=beta)

A separate subclass is provided for explicit Runge-Kutta methods: *ExplicitRungeKuttaMethod*.
If a method is explicit, it is important to instantiate it as an
ExplicitRungeKuttaMethod and not simply a RungeKuttaMethod, since the
latter class has significantly less functionality in NodePy.
Most significantly, time stepping is currently implemented for explicit methods,
but not for implicit methods.

.. automodule:: nodepy.runge_kutta_method
   :noindex:

Accuracy
--------

The principal measure of accuracy of a Runge-Kutta method is its
*order of accuracy*.  By comparing the Runge-Kutta solution with
the Taylor series for the exact solution, it can be shown that the
local truncation error for small enough step size $h$ is approximately

Error `\approx Ch^p`,

where $C$ is a constant independent of $h$.
Thus the expected asymptotic rate of convergence for small
step sizes is $p$.  This error corresponds to the lowest-order terms
that do not match those of the exact solution Taylor series.
Typically, a higher order accurate method will provide greater
accuracy than a lower order method even for practical step sizes.

In order to compare two methods with the same order of accuracy
more detailed information about the accuracy of a method may be
obtained by considering the relative size of the constant $C$.
This can be measured in various ways and is referred to as the
principal error norm.

For example::

    >> rk22.order()
    2
    >> rk44.order()
    4
    >> rk44.principal_error_norm()
    0.014504582343198208
    >> ssp104=rk.loadRKM('SSP104')
    >> ssp104.principal_error_norm()
    0.002211223747053554

Since the SSP(10,4) method has smaller principal error norm, we
expect that it will provide better accuracy than the classical 4-stage
Runge-Kutta method for a given step size.  Of course, the SSP method
has 10 stages, so it requires more work per step.  In order to
determine which method is more efficient, we need to compare the relative
accuracy for a fixed amount of work::

    >> rk.relative_accuracy_efficiency(rk44,ssp104)
    1.7161905294239843

This indicates that, for a desired level of error, the SSP(10,4)
method will require about 72\% more work.

.. automethod:: nodepy.runge_kutta_method.RungeKuttaMethod.principal_error_norm
   :noindex:

.. automethod:: nodepy.runge_kutta_method.RungeKuttaMethod.error_metrics
   :noindex:

.. automethod:: nodepy.runge_kutta_method.RungeKuttaMethod.stage_order
   :noindex:

Classical (linear) stability
----------------------------

.. automethod:: nodepy.runge_kutta_method.RungeKuttaMethod.stability_function
   :noindex:

.. automethod:: nodepy.runge_kutta_method.RungeKuttaMethod.plot_stability_region
   :noindex:

Nonlinear stability
-------------------

.. automethod:: nodepy.runge_kutta_method.RungeKuttaMethod.absolute_monotonicity_radius
   :noindex:

.. automethod:: nodepy.runge_kutta_method.RungeKuttaMethod.circle_contractivity_radius
   :noindex:

Reducibility of Runge-Kutta methods
-----------------------------------

Two kinds of reducibility (*DJ-reducibility* and *HS-reducibility*) have
been identified in the literature.  NodePy contains functions for detecting
both and transforming a reducible method to an equivalent irreducible method.
Of course, reducibility is dealt with relative to some numerical tolerance,
since the method coefficients are floating point numbers.

.. automethod:: nodepy.runge_kutta_method.RungeKuttaMethod._dj_reducible_stages
   :noindex:

.. automethod:: nodepy.runge_kutta_method.RungeKuttaMethod.dj_reduce
   :noindex:

.. automethod:: nodepy.runge_kutta_method.RungeKuttaMethod._hs_reducible_stages
   :noindex:

Composing Runge-Kutta methods
-----------------------------

Butcher has developed an elegant theory of the group structure of
Runge-Kutta methods.  The Runge-Kutta methods form a group under the
operation of composition.  The multiplication operator has been
overloaded so that multiplying two Runge-Kutta methods gives the
method corresponding to their composition, with equal timesteps.

It is also possible to compose methods with non-equal timesteps using the
compose() function.

Embedded Runge-Kutta Pairs
==========================

.. autoclass:: nodepy.runge_kutta_method.ExplicitRungeKuttaPair
   :noindex:

Low-Storage Runge-Kutta methods
===============================

.. automodule:: nodepy.low_storage_rk
   :noindex:

2S/3S methods
-------------

.. autoclass:: nodepy.low_storage_rk.TwoSRungeKuttaMethod
   :noindex:

2S/3S embedded pairs
--------------------

.. autoclass:: nodepy.low_storage_rk.TwoSRungeKuttaPair
   :noindex:

2R/3R methods
-------------

.. autoclass:: nodepy.low_storage_rk.TwoRRungeKuttaMethod
   :noindex:

