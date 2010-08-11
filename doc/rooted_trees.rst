============
Rooted Trees
============

Rooted trees provide a way of understanding properties of multistage
methods (such as Runge-Kutta methods).

Plotting trees
==============

.. plot::
   :include-source:

   from NodePy.rooted_trees import *
   tree=RootedTree('{T^2{T{T}}{T}}')
   tree.plot()

.. plot::
   :include-source:

   from NodePy.rooted_trees import *
   plot_all_trees(5)



.. automethod:: NodePy.rooted_trees.RootedTree.plot

Computing products on trees
===========================

Using rooted trees to generate order conditions
===============================================
