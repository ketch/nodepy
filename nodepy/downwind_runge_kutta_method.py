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
import snp

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
        A,b,alpha,beta=snp.normalize(A,b,alpha,beta)

        butchform=[x is not None for x in [A,At,b,bt]]
        SOform=[x is not None for x in [alpha,alphat,beta,betat]]
        if not ( ( all(butchform) and not (True in SOform) ) or
                    ( (not (True in butchform)) and all(SOform) ) ):
            raise Exception("""To initialize a Runge-Kutta method,
                you must provide either Butcher arrays or Shu-Osher arrays,
                but not both.""")
        if A is not None: #Initialize with Butcher arrays
            # Check that number of stages is consistent
            m=np.size(A,0) # Number of stages
            if m>1:
                if not np.all([np.size(A,1),np.size(b)]==[m,m]):
                   raise Exception('Inconsistent dimensions of Butcher arrays')
            else:
                if not np.size(b)==1:
                    raise Exception('Inconsistent dimensions of Butcher arrays')
        elif alpha is not None: #Initialize with Shu-Osher arrays
            A,At,b,bt=downwind_shu_osher_to_butcher(alpha,alphat,beta,betat)
        # Set Butcher arrays
        if len(np.shape(A))==2: 
          self.A=A
          self.At=At
        else: 
          self.A =snp.array([A ]) #Fix for 1-stage methods
          self.At=snp.array([At])
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
        from utils import shortstring
        c = [shortstring(ci) for ci in self.c]
        clenmax = max([len(ci) for ci in c])
        A = [shortstring(ai) for ai in self.A.reshape(-1)]
        alenmax = max([len(ai) for ai in A])
        b = [shortstring(bi) for bi in self.b]
        blenmax = max([len(bi) for bi in b])
        At = [shortstring(ai) for ai in self.At.reshape(-1)]
        atlenmax = max([len(ai) for ai in At])
        bt = [shortstring(bi) for bi in self.bt]
        btlenmax = max([len(bi) for bi in bt])
        colmax=max(alenmax,blenmax)
        colmax2 = max(atlenmax, btlenmax)
        colmax = max(colmax,colmax2)


        s=self.name+'\n'+self.info+'\n'
        for i in range(len(self)):
            s+=c[i]+' '*(clenmax-len(c[i])+1)+'| '
            for j in range(len(self)):
                ss=shortstring(self.A[i,j])
                s+=ss.ljust(colmax+1)
            s+=' | '    
            for j in range(len(self)):
                ss=shortstring(self.At[i,j])
                s+=ss.ljust(colmax+1)
            s+='\n'
        s+='_'*(clenmax+1)+'|'+('_______'*2*len(self))+('_____')+'\n'
        s+= ' '*(clenmax+1)+'|'
        for j in range(len(self)):
            s+=' '*(colmax-len(b[j])+1)+b[j]
        s+=' | '
        for j in range(len(self)):
            s+=' '*(colmax-len(bt[j])+1)+bt[j]
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
    at12=(r**2-4*r+2)/(2*r)
    #a21=(r-2.)/2.
    alpha=np.array([[2/r,0],[1,0],[0,1]])
    alphat=np.array([[0,(r**2-4*r+2)/(r**2-2*r)],[0,0],[0,0]])
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
        raise Exception('Inconsistent dimensions of Shu-Osher arrays')
    X=snp.eye(m)-alpha[0:m,:]-alphat[0:m,:]
    A =snp.solve(X, beta[0:m,:])
    At=snp.solve(X,betat[0:m,:])
    b=beta[m,:]+np.dot(alpha[m,:]+alphat[m,:],A)
    bt=betat[m,:]+np.dot(alpha[m,:]+alphat[m,:],At)
    return A,At,b,bt


