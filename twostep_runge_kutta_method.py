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
    def __init__(self,d,theta,Ahat,A,bhat,b,name='Two-step Runge-Kutta Method'):
        r"""
            Initialize a 2-step Runge-Kutta method.  The representation
            uses the form and notation of [Jackiewicz1995]_.

            \\begin{align*}
            y^n_j = & d_j u^{n-1} + (1-d_j)u^n + \\Delta t \\sum_{k=1}^{s}
            (\\hat{a}_{jk} f(y_k^{n-1}) + a_{jk} f(y_k^n)) & (1\\le j \\le s) \\\\
            u^{n+1} = & \\theta u^{n-1} + (1-\\theta)u^n + \\Delta t \\sum_{j=1}^{s}(\\hat{b}_j f(y_j^{n-1}) + b_j f(y_j^n))
            \\end{align*}

        """
        self.d,self.theta,self.Ahat,self.A,self.bhat,self.b=d,theta,Ahat,A,bhat,b
        self.name=name

    def order(self,tol=1.e-13):
        r""" 
            Return the order of a Two-step Runge-Kutta method.
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
        d,theta,Ahat,A,bhat,b=self.d,self.theta,self.Ahat,self.A,self.bhat,self.b
        e=np.ones(len(d))
        c=dot(Ahat+A,e)-d
        code=TSRKOrderConditions(p)
        z=np.zeros(len(code))
        for i in range(len(code)):
            exec('z[i]='+code[i])
            #print p,z
        return z


    def stability_matrix(self,z):
        r""" 
            Constructs the stability matrix of a two-step Runge-Kutta method.
            Right now just for a specific value of z.
            We ought to use Sage to do it symbolically.

            **Output**:
                M -- stability matrix evaluated at z
        """
        D=np.vstack([1.-self.d,self.d]).T #This may need to be reversed l/r
        thet=np.hstack([1.-self.theta,self.theta])
        M1=np.linalg.solve(np.eye(len(self.b))-z*self.A,D+z*self.Ahat)
        L1=thet+z*self.bhat+z*np.dot(self.b,M1)
        M=np.vstack([L1,[1.,0.]])
        return M

    def plot_stability_region(self,N=50,bounds=[-10,1,-5,5],
                    color='r',filled=True,scaled=False):
        r""" 
            Plot the region of absolute stability
            of a Two-step Runge-Kutta method, i.e. the set

            `\{ z \in C : M(z) is power bounded \}`

            where $M(z)$ is the stability matrix of the method.

            **Input**: (all optional)
                - N       -- Number of gridpoints to use in each direction
                - bounds  -- limits of plotting region
                - color   -- color to use for this plot
                - filled  -- if true, stability region is filled in (solid); otherwise it is outlined
        """
        import pylab as pl

        x=np.linspace(bounds[0],bounds[1],N)
        y=np.linspace(bounds[2],bounds[3],N)
        X=np.tile(x,(N,1))
        Y=np.tile(y[:,np.newaxis],(1,N))
        Z=X+Y*1j
        R=Z*0
        for i,xx in enumerate(x):
            for j,yy in enumerate(y):
                M=self.stability_matrix(xx+yy*1j)
                R[j,i]=max(abs(np.linalg.eigvals(M)))
                #print xx,yy,R[i,j]
        if filled:
            pl.contourf(X,Y,R,[0,1],colors=color)
        else:
            pl.contour(X,Y,R,[0,1],colors=color)
        pl.title('Absolute Stability Region for '+self.name)
        pl.hold(True)
        pl.plot([0,0],[bounds[2],bounds[3]],'--k',linewidth=2)
        pl.plot([bounds[0],bounds[1]],[0,0],'--k',linewidth=2)
        pl.axis('Image')
        pl.hold(False)


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
    r"""
        Load two particular TSRK methods (From [Jackiewicz1995]_).
    """
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


