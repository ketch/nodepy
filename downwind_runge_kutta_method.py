"""
Class for Downwind Runge-Kutta methods, and various functions related to them.

AUTHOR: David Ketcheson (03-10-2010)

EXAMPLES:

REFERENCES:
    [higueras2005]_
"""
from __future__ import division
from general_linear_method import GeneralLinearMethod
import runge_kutta_method as rk
import utils
import numpy as np

#=====================================================
class DownwindRungeKuttaMethod(GeneralLinearMethod):
#=====================================================
    """ General class for Downwind Runge-Kutta Methods """
    def __init__(self,A=None,At=None,b=None,bt=None,alpha=None,beta=None,
            alphat=None,betat=None,
            name='downwind Runge-Kutta Method',description=''):
        r"""
            Initialize a downwind Runge-Kutta method.  The representation
            uses the form and notation of [Ketcheson2010]_.

            \\begin{align*}
            y^n_i = & u^{n-1} + \\Delta t \\sum_{j=1}^{s}
            (a_{ij} f(y_j^{n-1}) + \\tilde{a}_{ij} \\tilde{f}(y_j^n)) & (1\\le j \\le s) \\\\
            \\end{align*}

        """
        butchform=[x is not None for x in [A,At,b,bt]]
        SOform=[x is not None for x in [alpha,alphat,beta,betat]]
        if not ( ( all(butchform) and not (True in SOform) ) or
                    ( (not (True in butchform)) and all(SOform) ) ):
            raise rk.RungeKuttaError("""To initialize a Runge-Kutta method,
                you must provide either Butcher arrays or Shu-Osher arrays,
                but not both.""")
        if A is not None: #Initialize with Butcher arrays
            # Check that number of stages is consistent
            m=np.size(A,0) # Number of stages
            if m>1:
                if not np.all([np.size(A,1),np.size(b)]==[m,m]):
                   raise rk.RungeKuttaError(
                    'Inconsistent dimensions of Butcher arrays')
            else:
                if not np.size(b)==1:
                    raise rk.RungeKuttaError(
                     'Inconsistent dimensions of Butcher arrays')
        elif alpha is not None: #Initialize with Shu-Osher arrays
            A,At,b,bt=downwind_shu_osher_to_butcher(alpha,alphat,beta,betat)
        # Set Butcher arrays
        if len(np.shape(A))==2: 
          self.A=A
          self.At=At
        else: 
          self.A =np.array([A ]) #Fix for 1-stage methods
          self.At=np.array([At])
        self.b=b;
        self.bt=bt
        self.c=np.sum(self.A,1)-np.sum(self.At,1)
        self.name=name
        self.info=description
        self.underlying_method=rk.RungeKuttaMethod(self.A-self.At,self.b-self.bt)

    def __repr__(self): 
        """
        Pretty-prints the Butcher array in the form:
          |   |
        c | A | At
        ___________
          | b | bt
        """
        s=self.name+'\n'+self.info+'\n'
        for i in range(len(self)):
            s+='%6.3f |' % self.c[i]
            for j in range(len(self)):
                s+=' %6.3f' % self.A[i,j]
            s+=' | '    
            for jj in range(len(self)):
                s+=' %6.3f' % self.At[i,jj]    
            s+='\n'
        s+='_______|'+('_______'*2*len(self))+('_____')+'\n'
        s+= '       |'
        for j in range(len(self)):
            s+=' %6.3f' % self.b[j]
        s+=' | '
        for jj in range(len(self)):
            s+=' %6.3f' % self.bt[jj]
        return s
 
    def __len__(self):
        """
            The length of the method is the number of stages.
        """
        return np.size(self.A,0) 

    def order(self,tol=1.e-13):
        r""" 
            Return the order of a Downwind Runge-Kutta method.
        """
        return self.underlying_method.order(tol)

    def absolute_monotonicity_radius(self,acc=1.e-10,rmax=200,
                    tol=3.e-16):
        r""" 
            Returns the radius of absolute monotonicity
            of a Runge-Kutta method.
        """
        from utils import bisect
        
        r=bisect(0,rmax,acc,tol,self.is_absolutely_monotonic)
        return r

    def is_absolutely_monotonic(self,r,tol):
        r""" Returns 1 if the downwind Runge-Kutta method is 
            absolutely monotonic at $z=-r$.

            The method is absolutely monotonic if $(I+rK+rKt)^{-1}$ exists
            and
            $$(I+rK+rKt)^{-1}K \\ge 0$$
            $$(I+rK+rKt)^{-1}Kt \\ge 0$$
            $$(I+rK+rKt)^{-1} e_m \\ge 0$$

            where $e_m$ is the m-by-1 vector of ones and
                  K=[ K  0
                     b^T 0].

            The inequalities are interpreted componentwise.

        """
        m=len(self)
        K  =np.hstack([np.vstack([self.A  ,self.b ]),np.zeros([m+1,1])])
        Kt =np.hstack([np.vstack([self.At ,self.bt]),np.zeros([m+1,1])])
        X=np.eye(len(self)+1) + r*(K+Kt)
        beta =r*np.linalg.solve(X, K)
        betat=r*np.linalg.solve(X,Kt)
        ech=np.linalg.solve(X,np.ones(m+1))
        if min(beta.min(),betat.min(),ech.min())<-tol:
            return 0
        else:
            return 1
        # Need an exception here if rhi==rmax

#================================================================

def loadDWRK(which='All'):
    r"""
        Load some particular DWRK method.
    """
    DWRK={}
    if which=='All': return TSRK
    else: return TSRK[which]

def opt_dwrk(r):
    #a11=(r**2-2*r-2)/(2.*r)
    at12=(r**2-4*r+2)/(2.*r)
    #a21=(r-2.)/2.
    alpha=np.array([[2./r,0],[1.,0],[0,1]])
    alphat=np.array([[0.,(r**2-4*r+2)/(r**2-2*r)],[0,0],[0,0]])
    beta=alpha/r
    betat=alphat/r
    return DownwindRungeKuttaMethod(alpha=alpha,beta=beta,alphat=alphat,betat=betat)


def downwind_shu_osher_to_butcher(alpha,alphat,beta,betat):
    r""" Accepts a Shu-Osher representation of a downwind Runge-Kutta
        method and returns the Butcher coefficients 

        \\begin{align*}
        A  = & (I-\\alpha_0-\\alphat_0)^{-1} \\beta_0 \\\\
        At = & (I-\\alpha_0-\\alphat_0)^{-1} \\betat_0 \\\\
        b = & \\beta_1 + (\\alpha_1 + \\alphat_1) * A
        \\end{align*}

        **References**:  
             #. [gottlieb2009]_
    """
    m=np.size(alpha,1)
    if not np.all([np.size(alpha,0),np.size(beta,0),
                    np.size(beta,1)]==[m+1,m+1,m]):
        raise RungeKuttaError(
             'Inconsistent dimensions of Shu-Osher arrays')
    X=np.eye(m)-alpha[0:m,:]-alphat[0:m,:]
    A=np.linalg.solve(X,beta[0:m,:])
    At=np.linalg.solve(X,betat[0:m,:])
    b=beta[m,:]+np.dot(alpha[m,:]+alphat[m,:],A)
    bt=betat[m,:]+np.dot(alpha[m,:]+alphat[m,:],At)
    return A,At,b,bt


