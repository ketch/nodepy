title: 'NodePy: A package for the analysis of numerical ODE solvers'
tags:
  - Python
  - numerical analysis
  - differential equations
  - Runge-Kutta method
  - linear multistep method
authors:
  - name: David I. Ketcheson^[Corresponding author.]
    orcid: 0000-0002-1212-126X
    affiliation: 1
  - name: Hendrik Ranocha
    orcid: 0000-0002-3456-2277
    affiliation: 1
  - name: Matteo Parsani
    orcid: 0000-0001-7300-1280
    affiliation: 1
  - name: Yiannis Hadjimichael
    orcid: 0000-0003-3517-8557
    affiliation: 2
affiliations:
 - name: King Abdullah University of Science & Technology
   index: 1
 - name: Eötvös Loránd Tudományegyetem
   index: 2
date: 9 July 2020
bibliography: paper.bib

---

# Summary

Ordinary differential equations are used to model a vast range of physical
and other phenomena.  They also arise in the discretization of partial differential
equations.  In most cases, solutions of differential equations must be approximated
by numerical methods.  The study of the properties of numerical methods for
ODEs comprises an important and large body of knowledge.  `NodePy` is a software
package for designing and studying the properties of numerical ODE solvers.
For the most important classes of methods, `Nodepy` can automatically assess
their stability, accuracy, and many other properties.

# Statement of need 

There are many software packages that *implement* ODE solvers with the purpose
of efficiently providing numerical solutions; in contrast, the purpose of
`NodePy` is to facilitate understanding of the properties of the solver algorithms
themselves.  In this sense, it is a sort of meta-software, consisting of
algorithms whose purpose is to compute properties of other algorithms.
It also serves as a reference, providing precise definitions of many of the
algorithms themselves.

`Nodepy` is written entirely in Python and provides software implementations
of many of the theoretical ideas contained for instance in reference texts
on numerical analysis of ODEs [@].  It also contains implementations of
many theoretical ideas from the numerical analysis of literature.
The implementation focuses on the two most important classes of methods;
namely, Runge-Kutta and linear multistep methods, but includes some
more exotic classes.  `NodePy` provides a means for numerical analysts to
quickly and easily determine the properties of existing methods or of new
methods they may develop.


#Nevertheless, the
#object-oriented framework is designed with other classes in mind
#and already includes (for instance) two-step Runge-Kutta methods.
#For instance, it can generate and check order conditions for Runge-Kutta
#methods of up to order 14; generate extrapolation methods based on a range
#of building-block schemes, up to any order of accuracy; and analyze the internal
#stability of Runge-Kutta methods.

`NodePy` development has been motivated largely by research needs and
it has been used in a number of papers (including some written by non-developers;
e.g. [@jin2019higher,@horvathembedded]) and also as a teaching tool for
graduate-level numerical analysis courses.  It relies on both Sympy and Numpy
in order to provide either exact or floating-point results based on the
nature of the inputs provided.

# Features

`NodePy` includes object-oriented representations of the following classes
of numerical methods:

 - Runge-Kutta methods
   - Explicit and Implicit
   - Embedded pairs
   - Classes of low-storage methods
   - Dense output formulas
   - Perturbed/additive and downwind methods
 - Linear multistep methods
 - Two-step Runge-Kutta methods
 - Additive (IMEX) linear multistep methods

The framework is designed to include general linear methods and even more
exotic classes.  Additionally, `NodePy` includes functionality for generating
representations of many specific methods, including:

 - Dozens of specific Runge-Kutta methods and pairs
 - General extrapolation methods, of any order of accuracy, based on a variety
   of building-block schemes and optionally including an error estimator
 - Deferred correction methods
 - Optimal SSP Runge-Kutta methods
 - Adams-Bashforth, Adams-Moulton, and BDF methods of any order

For all of these methods, `NodePy` provides methods and functions to compute many
of their properties -- too many to list here.  The theory on which most of these
properties are based is outlined in standard references 
[@hairer1993,@Hairer:ODEs2,@Hairer:ODEs2].  Many other properties are based on
recent research; usually the method docstring includes a reference to the relevant
paper.  Implementations of the methods themselves are also included as a convenience,
though they are not the primary purpose and are not expected to be efficient since
they are coded in pure Python.  Additional intermediate objects, such as the
absolute stability function of a method, are given their own software representation
and corresponding methods.

Additional features are provided to facilitate the analysis and testing of
these numerical methods.  This includes a range of initial value problems
for testing, such as the stiff and non-stiff DETEST suites, and a few simple
PDE semi-discretizations.  Also included is a library for dealing with rooted
trees, which are a class of graphs that play a key role in the theory of Runge-Kutta
methods.

# Related research and software

NodePy development has proceeded in close connection to the RK-Opt package.
Whereas NodePy is focused in the analysis of numerical methods, RK-Opt is focused 
more on their design through the use of numerical optimization to search
for optimal coefficients tailored to specific desired properties.
A common workflow involves generating new methods with RK-Opt and then studying
their properties in more detail using NodePy.

Some of the research projects that have made use of NodePy (most of which have led
to its further development) include development of:

 - Strong stability preserving (SSP) Runge-Kutta methods 
   [@2008_explicit_ssp, @2009_implicit_ssp, @2013_effective_order_ssp]
 - SSP general linear methods [@2011_tsrk, @2017_msrk]
 - Low-storage Runge-Kutta methods [@2010_LSRK]
 - Additive and downwind SSP Runge-Kutta methods [@2011_dwssp, @2018_perturbations]
 - High-order parallel extrapolation and deferred correction methods [@2014_internal_stability]
 - SSP linear multistep methods [@2016_ssp_lmm_vss, @2018_sspalmm]
 - Dense output formulas for Runge-Kutta methods [@2017_dense]
 - Internal stability theory for Runge-Kutta methods [@2014_internal_stability]
 - Embedded pairs for Runge-Kutta methods [@horvathembedded]
 
Additional recent applications include [@norton2015structure, @jin2019higher].
As can be seen from this list, applications have mostly stemmed from the
work of the main developer's research group, but have recently begun to expand
beyond that.

# Acknowledgements

Much of the initial NodePy development was performed by D. Ketcheson while
he was supported by a DOE Computational Science Graduate Fellowship.  Development
has also been supported by funding from King Abdullah University of Science and Technology.

# References
