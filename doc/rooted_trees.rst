Rooted Trees
============

.. plot::

   from nodepy.rooted_trees import *
   tree=RootedTree('{T^2{T{T}}{T}}')
   tree.plot()


.. autoclass:: nodepy.rooted_trees.RootedTree
   :noindex:

Plotting trees
==============

A single tree can be plotted using the plot method of the RootedTree class.
For convenience, the method plot_all_trees() plots the whole forest of rooted
trees of a given order.

.. automethod:: nodepy.rooted_trees.RootedTree.plot
   :noindex:

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
=========================

.. automethod:: nodepy.rooted_trees.RootedTree.order
   :noindex:

.. automethod:: nodepy.rooted_trees.RootedTree.density
   :noindex:

.. automethod:: nodepy.rooted_trees.RootedTree.symmetry
   :noindex:

Computing products on trees
===========================

.. automethod:: nodepy.rooted_trees.RootedTree.Gprod
   :noindex:

.. automethod:: nodepy.rooted_trees.RootedTree.lamda
   :noindex:

Using rooted trees to generate order conditions
===============================================
