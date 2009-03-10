"""
Class for two-step Runge-Kutta methods, and various functions related to them.

AUTHOR: David Ketcheson (08-30-2008)

EXAMPLES:

REFERENCES:
    [jackiewicz1995,butcher1997,hairer1997]
"""
from __future__ import division
from general_linear_method import GeneralLinearMethod
import numpy as np
import rooted_trees as tt
from sympy import Symbol
from strmanip import *

#=====================================================
class TwoStepRungeKuttaMethod(GeneralLinearMethod):
#=====================================================
    """ General class for Two-step Runge-Kutta Methods """
    def __init__(self,d,theta,Ahat,A,bhat,b):
        self.d,self.theta,self.Ahat,self.A,self.bhat,self.b=d,theta,Ahat,A,bhat,b

    def order(self,tol=1.e-14):
        """ Returns the order of a Two-step Runge-Kutta method """
        p=0
        while True:
            z=self.orderConditions(p+1)
            if np.any(abs(z)>tol): return p
            p=p+1

    def orderConditions(self,p):
        from numpy import dot
        d,theta,Ahat,A,bhat,b=self.d,self.theta,self.Ahat,self.A,self.bhat,self.b
        e=np.ones(len(d))
        c=dot(Ahat+A,e)-d
        code=TSRKOrderConditions(p)
        z=np.zeros(len(code))
        for i in range(len(code)):
            exec('z[i]='+code[i])
            print p,z
        return z

#================================================================
# Functions for analyzing Two-step Runge-Kutta order conditions
#================================================================

def TSRKOrderConditions(p,ind='all'):
    forest=tt.recursive_trees(p)
    code=[]
    for tree in forest:
        code.append(tsrk_elementary_weight_str(tree)+'-'+str(tree.Emap()))
        code[-1]=code[-1].replace('--','')
        code[-1]=code[-1].replace('1 ','e ')
        code[-1]=code[-1].replace('1)','e)')
    return code

def tsrk_elementary_weight(tree):
    """
        Constructs Butcher's elementary weights 
        for Two-step Runge-Kutta methods
    """
    bhat,b,theta=Symbol('bhat',False),Symbol('b',False),Symbol('theta',False)
    ew=bhat*tree.Gprod(tt.Emap,tt.Gprod,betaargs='TSRKeta,Dmap',alphaargs='-1')+b*tree.Gprod(tt.TSRKeta,tt.Dmap)+theta*tree.Emap(-1)
    return ew

def tsrk_elementary_weight_str(tree):
    """
        Constructs Butcher's elementary weights 
        for Two-step Runge-Kutta methods
        as numpy-executable strings
    """
    ewstr='dot(bhat,'+tree.Gprod_str(tt.Emap_str,tt.Gprod_str,betaargs='TSRKeta_str,Dmap_str',alphaargs='-1')+')+dot(b,'+tree.Gprod_str(tt.TSRKeta_str,tt.Dmap_str)+')+theta*'+str(tree.Emap(-1))
    ewstr=mysimp(ewstr)
    return ewstr

#================================================================

def loadTSRK(which='All'):
    TSRK={}
    #================================================
    d=np.array([-113./88,-103./88])
    theta=-4483./8011
    Ahat=np.array([[1435./352,-479./352],[1917./352,-217./352]])
    A=np.eye(2)
    bhat=np.array([180991./96132,-17777./32044])
    b=np.array([-44709./32044,48803./96132])
    TSRK['order4']=TwoStepRungeKuttaMethod(d,theta,Ahat,A,bhat,b)
    #================================================
    d=np.array([-0.210299,-0.0995138])
    theta=-0.186912
    Ahat=np.array([[1.97944,0.0387917],[2.5617,2.3738]])
    A=np.zeros([2,2])
    bhat=np.array([1.45338,0.248242])
    b=np.array([-0.812426,-0.0761097])
    TSRK['order5']=TwoStepRungeKuttaMethod(d,theta,Ahat,A,bhat,b)
    if which=='All': return TSRK
    else: return TSRK[which]
