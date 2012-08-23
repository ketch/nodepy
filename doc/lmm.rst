.. contents::

.. _create_lmm:

Linear Multistep methods
==================================
A linear multistep method computes the next solution value from the values
at several previous steps:

    `\alpha_k y_{n+k} + \alpha_{k-1} y_{n+k-1} + ... + \alpha_0 y_n
    = h ( \beta_k f_{n+k} + ... + \beta_0 f_n )`

Note that different conventions for numbering the coefficients exist;
the above form is used in NodePy.
Methods are automatically normalized so that $\\alpha_k=1$.

.. automodule:: nodepy.linear_multistep_method
   :noindex:

------------------------
Instantiation
------------------------
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


Adams-Bashforth Methods
------------------------

.. automethod:: nodepy.linear_multistep_method.Adams_Bashforth
   :noindex:

Adams-Moulton Methods
------------------------

.. automethod:: nodepy.linear_multistep_method.Adams_Moulton
   :noindex:

Backward-difference formulas
------------------------------

.. automethod:: nodepy.linear_multistep_method.backward_difference_formula
   :noindex:

Optimal Explicit SSP methods
------------------------------

.. automethod:: nodepy.linear_multistep_method.elmm_ssp2
   :noindex:

------------------------
Stability
------------------------

Characteristic Polynomials
------------------------------
.. automethod:: nodepy.linear_multistep_method.LinearMultistepMethod.characteristic_polynomials
   :noindex:

Plotting The Stability Region
------------------------------
.. automethod:: nodepy.linear_multistep_method.LinearMultistepMethod.plot_stability_region
   :noindex:
