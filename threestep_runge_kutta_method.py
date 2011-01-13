"""
Class for three-step Runge-Kutta methods, and various functions related to them.
This still contains errors, particularly in the order conditions!

AUTHOR: David Ketcheson (06-21-2009)

EXAMPLES:

REFERENCES:
    [hairer1993]

.. warning::

    The order condition functions here have not been carefully tested
    and may not generate correct order conditions!
"""
from __future__ import division
from general_linear_method import GeneralLinearMethod
import numpy as np
import rooted_trees as tt
from sympy import Symbol
from strmanip import *

#=====================================================
class ThreeStepRungeKuttaMethod(GeneralLinearMethod):
#=====================================================
    """ General class for Three-step Runge-Kutta Methods """
    def __init__(self,D,theta,A,b):
        r"""
            Initialize a 3-step Runge-Kutta method.  The representation
            uses the form and notation of [Ketcheson2009]_.
        """
        self.D,self.theta,self.A,self.b=D,theta,A,b

    def order(self,tol=1.e-13):
        r""" 
            Return the order of a Three-step Runge-Kutta method.
            Computed by computing the elementary weights corresponding
            to the appropriate rooted trees.
        """
        p=0
        while True:
            z=self.orderConditions(p+1)
            if np.any(abs(z)>tol): return p
            p=p+1

    def orderConditions(self,p):
        r"""
            Evaluate the order conditions corresponding to rooted trees
            of order $p$.

            **Output**:
                - A vector $z$ of residuals (the amount by which each
                  order condition is violated)
        """
        from numpy import dot
        D,theta,A,b=self.D,self.theta,self.A,self.b
        l=np.array(range(0,3))
        e=np.ones(A.shape[1])
        c=dot(A,e)-dot(D,l)
        code=ThSRKOrderConditions(p)
        z=np.zeros(len(code))
        for i in range(len(code)):
            exec('z[i]='+code[i])
            #print p,z
        return z

#================================================================
# Functions for analyzing Three-step Runge-Kutta order conditions
#================================================================

def ThSRKOrderConditions(p,ind='all'):
    forest=tt.list_trees(p)
    code=[]
    for tree in forest:
        code.append(thsrk_elementary_weight_str_matlab(tree)+'-'+str(tree.Emap()))
        code[-1]=code[-1].replace('--','')
        code[-1]=code[-1].replace('1 ','e ')
        code[-1]=code[-1].replace('1)','e)')
    return code

def thsrk_elementary_weight(tree):
    """
        Constructs Butcher's elementary weights 
        for Three-step Runge-Kutta methods.
    """
    b,theta=Symbol('b',False),Symbol('theta',False)
    ew=b*tree.Gprod(tt.ThSRKeta,tt.Dmap)+theta2*tree.Emap(-1)+theta3*tree.Emap(-2)
    return ew

def thsrk_elementary_weight_str(tree):
    """
        Constructs Butcher's elementary weights 
        for Two-step Runge-Kutta methods
        as numpy-executable strings
    """
    ewstr='dot(b,'+tree.Gprod_str(ThSRKeta_str,tt.Dmap_str)+')+('+str(tree.Emap(-1))+')*theta2+('+str(tree.Emap(-2))+')*theta3'
    ewstr=mysimp(ewstr)
    return ewstr

def thsrk_elementary_weight_str_matlab(tree):
    """
        Constructs Butcher's elementary weights 
        for Two-step Runge-Kutta methods
        as matlab-executable strings
    """
    ewstr="b'*("+tree.Gprod_str(tt.ThSRKeta_str_matlab,tt.Dmap_str)+')+('+str(tree.Emap(-1))+')*theta2+('+str(tree.Emap(-2))+')*theta3'
    ewstr=mysimp(ewstr)
    ewstr=ewstr.replace('**','.^')
    return ewstr

def ThSRKeta(tree):
    from rooted_trees import Dprod
    from sympy import symbols
    raise Exception('This function does not work correctly; use the _str version')
    if tree=='':  return 1
    if tree=='T': return symbols('c',commutative=False)
    return symbols('d2',commutative=False)*tree.Emap(-1)+symbols('d3',commutative=False)*tree.Emap(-2)+symbols('A',commutative=False)*Dprod(tree,ThSRKeta)

def ThSRKeta_str(tree):
    """
    Computes eta(t) for Two-step Runge-Kutta methods -- Python string
    """
    from rooted_trees import Dprod_str
    if tree=='':  return 'e'
    if tree=='T': return 'c'
    return '(d2*'+str(tree.Emap(-1))+'+(d3*'+str(tree.Emap(-2))+'+dot(A,'+Dprod_str(tree,ThSRKeta_str)+'))'

def ThSRKeta_str_matlab(tree):
    """
    Computes eta(t) for Two-step Runge-Kutta methods -- Matlab string
    """
    from rooted_trees import Dprod_str
    if tree=='':  return 'e'
    if tree=='T': return 'c'
    return "("+str(tree.Emap(-1))+")*d2+("+str(tree.Emap(-2))+")*d3+(A*"+Dprod_str(tree,ThSRKeta_str_matlab)+')'


#================================================================
