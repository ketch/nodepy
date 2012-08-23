==============================
Analyzing Stability Properties
==============================

Plotting the region of absolute stability
=========================================

Region of absolute stability for the optimal SSP 10-stage, 4th order
Runge-Kutta method:

.. plot::
   :include-source:

   from nodepy.runge_kutta_method import *
   ssp104=loadRKM('SSP104')
   ssp104.plot_stability_region(bounds=[-15,1,-10,10])

.. automethod:: nodepy.runge_kutta_method.RungeKuttaMethod.plot_stability_region
   :noindex:

Region of absolute stability for the 3-step Adams-Moulton method:

.. plot::
   :include-source:

   from nodepy.linear_multistep_method import *
   am3=Adams_Moulton(3)
   am3.plot_stability_region()

.. automethod:: nodepy.linear_multistep_method.LinearMultistepMethod.plot_stability_region
   :noindex:



Plotting the order star
=========================================

Order star for the optimal SSP 10-stage, 4th order
Runge-Kutta method:

.. plot::
   :include-source:

   from nodepy.runge_kutta_method import *
   ssp104=loadRKM('SSP104')
   ssp104.plot_order_star()

.. automethod:: nodepy.runge_kutta_method.RungeKuttaMethod.plot_stability_region
   :noindex:
