============
Rooted Trees
============
.. plot::

   from nodepy.rooted_trees import *
   tree=RootedTree('{T^2{T{T}}{T}}')
   tree.plot()


.. autoclass:: nodepy.rooted_trees.RootedTree

Plotting trees
==============

A single tree can be plotted using the plot method of the RootedTree class.
For convenience, the method plot_all_trees() plots the whole forest of rooted
trees of a given order.

.. automethod:: nodepy.rooted_trees.RootedTree.plot

.. plot::
   :include-source:

   from nodepy.rooted_trees import *
   tree=RootedTree('{T^2{T{T}}{T}}')
   tree.plot()

.. plot::
   :include-source:

   from nodepy.rooted_trees import *
   plot_all_trees(5)


Functions on rooted trees
===========================
.. automethod:: nodepy.rooted_trees.RootedTree.order

.. automethod:: nodepy.rooted_trees.RootedTree.density

.. automethod:: nodepy.rooted_trees.RootedTree.symmetry

Computing products on trees
===========================
.. automethod:: nodepy.rooted_trees.RootedTree.Gprod

.. automethod:: nodepy.rooted_trees.RootedTree.lamda

Using rooted trees to generate order conditions
===============================================
