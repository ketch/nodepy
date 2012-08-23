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
from strmanip import *

#=====================================================
class TwoStepRungeKuttaMethod(GeneralLinearMethod):
#=====================================================
    """ General class for Two-step Runge-Kutta Methods 
        The representation
        uses the form and notation of [Jackiewicz1995]_.

        `\\begin{align*}
        y^n_j = & d_j u^{n-1} + (1-d_j)u^n + \\Delta t \\sum_{k=1}^{s}
        (\\hat{a}_{jk} f(y_k^{n-1}) + a_{jk} f(y_k^n)) & (1\\le j \\le s) \\\\
        u^{n+1} = & \\theta u^{n-1} + (1-\\theta)u^n + \\Delta t \\sum_{j=1}^{s}(\\hat{b}_j f(y_j^{n-1}) + b_j f(y_j^n))
        \\end{align*}`
    """
    def __init__(self,d,theta,A,b,Ahat=None,bhat=None,type='Type II',name='Two-step Runge-Kutta Method'):
        r"""
            Initialize a 2-step Runge-Kutta method."""
        self.s = max(np.shape(b))
        self.d,self.theta,self.A,self.b = d,theta,A,b
        self.Ahat,self.bhat=Ahat,bhat
        #if type=='General': self.Ahat,self.bhat=Ahat,bhat
        #elif type=='Type I':
        #    self.Ahat = np.zeros([self.s,self.s])
        #    self.bhat = np.zeros([self.s,1])
        #elif type=='Type II':
        #    self.Ahat = np.zeros([self.s,self.s])
        #    self.Ahat[:,0]=Ahat
        #    self.bhat = np.zeros([self.s,1])
        #    self.bhat[0]=bhat
        #else: raise TwoStepRungeKuttaError('Unrecognized type')
        self.name=name
        self.type=type

    def order(self,tol=1.e-13):
        r""" 
            Return the order of a Two-step Runge-Kutta method.
            Computed by computing the elementary weights corresponding
            to the appropriate rooted trees.
        """
        p=0
        while True:
            z=self.order_conditions(p+1)
            if np.any(abs(z)>tol): return p
            p=p+1

    def order_conditions(self,p):
        r"""
            Evaluate the order conditions corresponding to rooted trees
            of order $p$.

            **Output**:
                - A vector $z$ of residuals (the amount by which each
                  order condition is violated)

            TODO: Implement simple order conditions (a la Albrecht)
                    for Type I & II TSRKs
        """
        from numpy import dot
        d,theta,Ahat,A,bhat,b=self.d,self.theta,self.Ahat,self.A,self.bhat,self.b
        e=np.ones([len(b),1])
        b=b.T; bhat=bhat.T
        c=dot(Ahat+A,e)-d
        code=TSRKOrderConditions(p)
        z=np.zeros(len(code))
        for i in range(len(code)):
            exec('yy='+code[i])
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

            WARNING: This only works for Type I & Type II methods
            right now!!!
        """
        D=np.hstack([1.-self.d,self.d]) 
        thet=np.hstack([1.-self.theta,self.theta])
        A,b=self.A,self.b
        if self.type=='Type II':
            ahat = np.zeros([self.s,1]); ahat[:,0] = self.Ahat[:,0]
            bh = np.zeros([1,1]); bh[0,0]=self.bhat[0]
            A=np.hstack([ahat,self.A])
            A=np.vstack([np.zeros([1,self.s+1]),A])
            b =  np.vstack([bh,self.b])

        M1=np.linalg.solve(np.eye(self.s)-z*self.A,D)
        L1=thet+z*np.dot(self.b.T,M1)
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

    def absolute_monotonicity_radius(self,acc=1.e-10,rmax=200,
                    tol=3.e-16):
        r""" 
            Returns the radius of absolute monotonicity
            of a TSRK method.
        """
        from utils import bisect
        r=bisect(0,rmax,acc,tol,self.is_absolutely_monotonic)
        return r

    def spijker_form(self):
        r""" Returns arrays $S,T$ such that the TSRK can be written
            $$ w = S x + T f(w),$$
            and such that $\[S \ \ T\]$ has no two rows equal.
        """
        s=self.s
        if self.type=='General':
            z0=np.zeros([s,1])
            z00=np.zeros([1,1])
            T3 = np.hstack([self.Ahat,z0,self.A,z0])
            T4 = np.hstack([self.bhat.T,z00,self.b.T,z00])
            T = np.vstack([np.zeros([s+1,2*s+2]),T3,T4])
            S1 = np.hstack([np.zeros([s+1,1]),np.eye(s+1)])
            S2 = np.hstack([self.d,np.zeros([s,s]),1-self.d])
            S3 = np.hstack([[[self.theta]],np.zeros([1,s]),[[1-self.theta]]])
            S = np.vstack([S1,S2,S3])

        elif self.type=='Type I':
            K = np.vstack([self.A,self.b.T])
            T2 = np.hstack([np.zeros([s+1,1]),K,np.zeros([s+1,1])])
            T =  np.vstack([np.zeros([1,s+1]),T2])
            S1 = np.vstack([np.zeros([1,1]),self.d,np.array([[self.theta]])])
            S2 = np.vstack([np.zeros([1,1]),1-self.d,np.array([[1-self.theta]])])
            S = np.hstack([S1,S2])

        elif self.type=='Type II':
            S0 = np.array([1,0])
            S1 = np.hstack([self.d,1-self.d])
            S2 = np.array([self.theta,1-self.theta])
            S =  np.vstack([S0,S1,S2])
            ahat = np.zeros([s,1])
            ahat[:,0] = self.Ahat[:,0]
            bh = np.zeros([1,1])
            bh[0,0]=self.bhat[0]
            T0 = np.zeros([1,s+2])
            T1 =  np.hstack([ahat,self.A,np.zeros([s,1])])
            T2 =  np.hstack([bh,self.b.T,np.zeros([1,1])])
            T = np.vstack([T0,T1,T2])
            
        return S,T

    def is_absolutely_monotonic(self,r,tol):
        r""" Returns 1 if the TSRK method is absolutely monotonic
            at $z=-r$.

            The method is absolutely monotonic if $(I+rT)^{-1}$ exists
            and
            $$(I+rT)^{-1}T \\ge 0$$
            $$(I+rT)^{-1}S \\ge 0$$

            The inequalities are interpreted componentwise.

            **References**:
                #. [spijker2007]
        """
        S,T = self.spijker_form()
        m=np.shape(T)[0]
        X=np.eye(m)+r*T
        if abs(np.linalg.det(X))<tol: return 0
        P = np.linalg.solve(X,T)
        R = np.linalg.solve(X,S)
        if P.min()<-tol or R.min()<-tol:
            return 0
        else:
            return 1
        # Need an exception here if rhi==rmax



#================================================================
# Functions for analyzing Two-step Runge-Kutta order conditions
#================================================================

def TSRKOrderConditions(p,ind='all'):
    forest=tt.list_trees(p)
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
    from sympy import Symbol
    bhat,b,theta=Symbol('bhat',False),Symbol('b',False),Symbol('theta',False)
    ew=bhat*tree.Gprod(tt.Emap,tt.Gprod,betaargs=[TSRKeta,Dmap],alphaargs=[-1])+b*tree.Gprod(TSRKeta,tt.Dmap)+theta*tree.Emap(-1)
    return ew

def tsrk_elementary_weight_str(tree):
    """
        Constructs Butcher's elementary weights 
        for Two-step Runge-Kutta methods
        as numpy-executable strings
    """
    from rooted_trees import Dmap_str
    ewstr='dot(bhat,'+tree.Gprod_str(tt.Emap_str,tt.Gprod_str,betaargs=[TSRKeta_str,Dmap_str],alphaargs=[-1])+')+dot(b,'+tree.Gprod_str(TSRKeta_str,tt.Dmap_str)+')+theta*'+str(tree.Emap(-1))
    ewstr=mysimp(ewstr)
    return ewstr

def TSRKeta(tree):
    from rooted_trees import Dprod
    from sympy import symbols
    raise Exception('This function does not work correctly; use the _str version')
    if tree=='':  return 1
    if tree=='T': return symbols('c',commutative=False)
    return symbols('d',commutative=False)*tree.Emap(-1)+symbols('Ahat',commutative=False)*tree.Gprod(Emap,Dprod,betaargs=[TSRKeta],alphaargs=[-1])+symbols('A',commutative=False)*Dprod(tree,TSRKeta)

def TSRKeta_str(tree):
    """
    Computes eta(t) for Two-step Runge-Kutta methods
    """
    from rooted_trees import Dprod_str, Emap_str
    if tree=='':  return 'e'
    if tree=='T': return 'c'
    return '(d*'+str(tree.Emap(-1))+'+dot(Ahat,'+tree.Gprod_str(Emap_str,Dprod_str,betaargs=[TSRKeta_str],alphaargs=[-1])+')'+'+dot(A,'+Dprod_str(tree,TSRKeta_str)+'))'


#================================================================

def loadTSRK(which='All'):
    r"""
        Load two particular TSRK methods (From [Jackiewicz1995]_).
    """
    TSRK={}
    #================================================
    d=np.array([[-113./88,-103./88]]).T
    theta=-4483./8011
    Ahat=np.array([[1435./352,-479./352],[1917./352,-217./352]])
    A=np.eye(2)
    bhat=np.array([[180991./96132,-17777./32044]]).T
    b=np.array([[-44709./32044,48803./96132]]).T
    TSRK['order4']=TwoStepRungeKuttaMethod(d,theta,A,b,Ahat,bhat)
    #================================================
    d=np.array([-0.210299,-0.0995138])
    theta=-0.186912
    Ahat=np.array([[1.97944,0.0387917],[2.5617,2.3738]])
    A=np.zeros([2,2])
    bhat=np.array([1.45338,0.248242])
    b=np.array([-0.812426,-0.0761097])
    TSRK['order5']=TwoStepRungeKuttaMethod(d,theta,A,b,Ahat,bhat)
    if which=='All': return TSRK
    else: return TSRK[which]


#=====================================================
class TwoStepRungeKuttaError(Exception):
#=====================================================
    """
        Exception class for Two-step Runge Kutta methods.
    """
    def __init__(self,msg='Two-step Runge Kutta Error'):
        self.msg=msg

    def __str__(self):
        return self.msg
#=====================================================

def load_type2_TSRK(s,p,type='Type II'):
    r"""
    Load a TSRK method from its coefficients in an ASCII file
    (usually from MATLAB).  The coefficients are stored in the 
    following order: [s d theta A b Ahat bhat].
    """
    path='/Users/ketch/Research/Projects/MRK/methods/TSRK/'
    file=path+'explicitzr'+str(s)+str(p)+'.tsrk'
    f=open(file,'r')
    coeff=[]
    for line in f:
        for word in line.split():
            coeff.append(float(word))

    s=int(coeff[0])
    A=np.zeros([s,s])
    Ahat=np.zeros([s,s])
    b=np.zeros([s,1])
    bhat=np.zeros([s,1])

    d=np.array(coeff[1:s+1],ndmin=2).T
    theta = coeff[s+1]
    for row in range(s):
        A[row,:] = coeff[s+2+s*row:s+2+s*(row+1)]
    b = np.array(coeff[s**2+s+2:s**2+2*s+2],ndmin=2).T
    for row in range(s):
        Ahat[row,:] = coeff[s**2+2*s+2+s*row:s**2+2*s+2+s*(row+1)]
    bhat = np.array(coeff[2*s**2+2*s+2:2*s**2+3*s+2],ndmin=2).T
    return TwoStepRungeKuttaMethod(d,theta,A,b,Ahat,bhat,type=type)
