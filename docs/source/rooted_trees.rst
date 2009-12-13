============
Rooted Trees
============

Rooted trees provide a way of understanding properties of multistage
methods (such as Runge-Kutta methods).

Plotting trees
==============

.. plot::

    >>> from CanoPy.rooted_trees import *
    >>> tree=RootedTree('{T^2{T{T}}{T}}')
    >>> tree.plot()

.. plot::

    >>> from CanoPy.rooted_trees import *
    >>> plot_all_trees(5)



.. automethod:: CanoPy.rooted_trees.RootedTree.plot

Computing products on trees
===========================

Using rooted trees to generate order conditions
===============================================
