"""
Class for Runge-Kutta methods, and various functions related to them.

**Author**: David Ketcheson (08-29-2008)

**Examples**::

    >>> import sys
    >>> mypath= '/Users/ketch/Panos'
    >>> if mypath not in sys.path: sys.path.append(mypath)
    >>> from general_linear_method import *
    >>> from runge_kutta_method import *

* Load a method::

    >>> ssp104=loadRKM('SSP104')

* Check its order of accuracy::

    >>> ssp104.order()
    4

* Find its radius of absolute monotonicity::

    >>> ssp104.absolute_monotonicity_radius()
    5.9999999999490683

* Load a dictionary with many methods::

    >>> RK=loadRKM()
    >>> RK.keys()
    ['GL3', 'FE11', 'BuRK65', 'SSP104', 'RK44', 'GL2', 'SSP22', 'SSP33', 'Mid22']

    >>> RK['Mid22']
        0.0 |
        0.5 | 0.5
        ____|________
            | 0.0 1.0


**TODO**:
    - Add functions to generate optimal implicit SSP families
"""
from __future__ import division
from general_linear_method import GeneralLinearMethod
import numpy as np
import pylab as pl
import rooted_trees as rt
from sympy import Symbol, factorial

#=====================================================
# TODO:
#     - Implement GLMs (Butcher's form? Spijker's form?)
#     - Unit testing for everything
#=====================================================


#=====================================================
class RungeKuttaMethod(GeneralLinearMethod):
#=====================================================
    r""" 
        General class for implicit and explicit Runge-Kutta Methods.
        The method is defined by its Butcher array ($A,b,c$).
        It is assumed everywhere that  `c_i=\sum_j A_{ij}`.

        **References**:  
            #. [butcher2003]_
            #. [hairer1993]_
    """
    def __init__(self,A=None,b=None,alpha=None,beta=None,
            name='Runge-Kutta Method',description=''):
        r"""
            A Runge-Kutta Method is initialized by providing either:
                #. Butcher arrays $A$ and $b$ with valid and consistent 
                   dimensions; or
                #. Shu-Osher arrays `\alpha` and `\beta` with valid and
                   consistent dimensions 

            but not both.

            The Butcher arrays are used as the primary representation of
            the method.  If Shu-Osher arrays are provided instead, the
            Butcher arrays are computed by :ref:`shu_osher_to_butcher`.
        """
        # Here there is a danger that one could change A
        # and c would never be updated
        # Maybe A,b, and c should be accessible through a setter function?
        a1,a2=A is not None, b is not None
        a3,a4=alpha is not None, beta is not None
        if not ( ( (a1 and a2) and not (a3 or a4) ) or
                    ( (a3 and a4) and not (a1 or a2) ) ):
            raise RungeKuttaError("""To initialize a Runge-Kutta method,
                you must provide either Butcher arrays or Shu-Osher arrays,
                but not both.""")
        if A is not None: #Initialize with Butcher arrays
            # Check that number of stages is consistent
            m=np.size(A,0) # Number of stages
            if m>1:
                if not np.all([np.size(A,1),np.size(b)]==[m,m]):
                   raise RungeKuttaError(
                    'Inconsistent dimensions of Butcher arrays')
            else:
                if not np.size(b)==1:
                    raise RungeKuttaError(
                     'Inconsistent dimensions of Butcher arrays')
        if alpha is not None: #Initialize with Shu-Osher arrays
            self.alpha=alpha
            self.beta=beta
            A,b=shu_osher_to_butcher(alpha,beta)
        # Set Butcher arrays
        if len(np.shape(A))==2: self.A=A
        else: self.A=np.array([A]) #Fix for 1-stage methods
        self.b=b
        self.c=np.sum(self.A,1)
        self.name=name
        self.info=description

    def __repr__(self): 
        """
        Pretty-prints the Butcher array in the form:
          |
        c | A
        ______
          | b
        """
        s=self.name+'\n'+self.info+'\n'
        for i in range(len(self)):
            s+='%6.3f |' % self.c[i]
            for j in range(len(self)):
                s+=' %6.3f' % self.A[i,j]
            s+='\n'
        s+='_______|'+('_______'*len(self))+'\n'
        s+= '       |'
        for j in range(len(self)):
            s+=' %6.3f' % self.b[j]
        return s

    def __eq__(self,rkm):
        """
            Methods considered equal if their Butcher arrays are

            TODO: Instead check whether methods have the same elementary weights
        """
        K1=np.vstack([self.A,self.b])
        K2=np.vstack([rkm.A,rkm.b])
        if shape(K1)!=shape(K2):
            return False
        else:
            return np.vstack([self.A,self.b])==np.vstack([rkm.A,rkm.b])

    def __len__(self):
        """
            The length of the method is the number of stages.
        """
        return np.size(self.A,0) 

    def __mul__(self,RK2,h1=1,h2=1):
        """ Multiplication is interpreted as composition:
            RK1*RK2 gives the method obtained by applying
            RK2, followed by RK1, each with half the timestep.

            **Output**:
                The method
                     c_2 | A_2  0
                   1+c_1 | b_2 A_1
                   _____________
                         | b_2 b_1

                but with everything divided by two.
                The b_2 matrix block consists of m_1 (row) copies of b_2.


            TODO: Think about whether this is the right thing to return.
        """
        f1=h1/(h1+h2)
        f2=h2/(h1+h2)
        A1=self.A
        A2=RK2.A
        A=np.vstack([
            np.hstack([A2*f2,np.zeros([np.size(A2,0),np.size(A1,1)])]),
            np.hstack([np.tile(RK2.b*f2,(len(self),1)),A1*f1])]).squeeze()
        b=np.hstack([RK2.b*f2,self.b*f1]).squeeze()
        return RungeKuttaMethod(A,b)

    def remove_stage(self,stage):
        """ Eliminate a stage of a Runge-Kutta method.
            Typically used to reduce reducible methods.
        """
        stage=stage-1 #account for different indexing
        A=np.delete(np.delete(self.A,stage,1),stage,0)
        b=np.delete(self.b,stage)
        c=np.delete(self.c,stage)
        self.A=A
        self.b=b
        self.c=c

    def error_coefficient(self,tree):
        from numpy import dot
        code=elementary_weight_str(tree)
        b=self.b
        A=self.A
        c=self.c
        exec('coeff=('+code+'-1./'+str(tree.density())+')')
        return coeff/tree.symmetry()

    def error_coeffs(self,p):
        forest=rt.recursive_trees(p)
        err_coeffs=[]
        for tree in forest:
            err_coeffs.append(self.error_coefficient(tree))
        return err_coeffs

    def error_metrics(self):
        q=self.order(1.e-13)
        tau_1=self.error_coeffs(q+1)
        tau_2=self.error_coeffs(q+2)
        print tau_1
        A_qp1=np.sqrt(float(np.sum(np.array(tau_1)**2)))
        A_qp1_max=max([abs(tau) for tau in tau_1])
        A_qp2=np.sqrt(float(np.sum(np.array(tau_2)**2)))
        A_qp2_max=max([abs(tau) for tau in tau_2])
        return A_qp1, A_qp1_max, A_qp2, A_qp2_max

    def principal_error_norm(self):
        p=self.order(1.e-13)
        forest=rt.recursive_trees(p+1)
        errs=[]
        for tree in forest:
            errs.append(self.error_coefficient(tree))
        return np.sqrt(float(np.sum(np.array(errs)**2)))
#        return max([abs(err) for err in errs])

    def order(self,tol=1.e-14):
        """ 
            Returns the order of a Runge-Kutta method.
        """
        p=0
        while True:
            z=self.order_conditions(p+1)
            if np.any(abs(z)>tol): return p
            p=p+1

    def order_conditions(self,p):
        """
            Generates and evaluates code to test whether a method
            satisfies the order conditions of order p (only).

            Currently uses Albrecht's recursion to generate the
            order conditions.  This is fast and requires less code
            than if they were hard-coded up to some high order,
            although it still only works up to some fixed order
            (need more recursion loops to go higher).

            Other possibilities, already in place, are:
                #. Use hard code, generated once and for all
                   by Albrecht's recursion or another method.
                   Advantages: fastest
                   Disadvantages: Less satisfying

                #. Use Butcher's recursive product on trees.
                   Advantages: Most satisfying, no maximum order
                   Disadvantages: way too slow for high order

            TODO: Decide on something and fill in this docstring.
        """
        A,b,c=self.A,self.b,self.c
        C=np.diag(c)
        code=runge_kutta_order_conditions(p)
        z=np.zeros(len(code)+1)
        tau=np.zeros([p,len(self)])
        for j in range(1,p):
            tau[j,:]=(c**j/j-np.dot(A,c**(j-1)))/factorial(j-1)
        for i in range(len(code)):
            exec('z[i]='+code[i])
        z[-1]=np.dot(b,c**(p-1))-1./p
        return z


    def propagation_matrix(self,L,dt):
        """
            Returns the solution propagation matrix for the linear 
            autonomous system with RHS equal to the matrix L, i.e. 
            it returns the matrix G such that when the Runge-Kutta
            method is applied to the system 
            $u'(t)=Lu$
            with stepsize dt, the numerical solution is given by
            $u^{n+1} = G u^n$.

            **Input**:
                - self -- a Runge-Kutta method
                - L    -- the RHS of the ODE system
                - dt   -- the timestep

            The formula for $G$ is (if $L$ is a scalar):
            $G = 1 + b^T L (I-A L)^{-1} e$

            where $A$ and $b$ are the Butcher arrays and $e$ is the vector
            of ones.  If $L$ is a matrix, all quantities above are 
            replaced by their Kronecker product with the identity
            matrix of size $m$, where $m$ is the number of stages of
            the Runge-Kutta method.
        """
        neq=np.size(L,0)
        nstage=len(self)
        I =np.identity(nstage)
        I2=np.identity(neq)
        Z=np.kron(I,dt*L)
        X=np.kron(I,I2)-np.dot(np.kron(self.A,I2),Z)
        Xinv=np.linalg.inv(X)
        e=np.kron(np.ones(nstage)[:,np.newaxis],I2)
        G=I2 + np.dot(np.kron(self.b[:,np.newaxis],I2).T,np.dot(Z,np.dot(Xinv,e)))

        return G,Xinv

    def standard_shu_osher_form(self,r=None):
        r"""
            Gives a Shu-Osher form in which the SSP coefficient is
            evident (i.e., in which $\\alpha_{ij},\\beta_{ij} \\ge 0$ and
            $\\alpha_{ij}/\\beta_{ij}=c$ for every $\\beta_{ij}\\ne 0$).

            **Input**: 
                - A RungeKuttaMethod
            **Output**: 
                - alpha, beta -- Shu-Osher arrays

            The 'optimal' Shu-Osher arrays are given by
            
            $$\\alpha= K(I+cA)^{-1}$$
            $$\\beta = c \\alpha$$

            where K=[ A
                     b^T].

            **References**: 
                #. [higueras2005]_

        """
        if r is None: r=self.absolute_monotonicity_radius()
        K=np.vstack([self.A,self.b])
        K=np.hstack([K,np.zeros([s+1,1])])
        X=np.eye(s+1)+r*K
        beta=np.linalg.solve(X,K)
        beta=beta[:,:-1]
        alpha=r*beta
        for i in range(1,len(self)+1):
            alpha[i,0]=1.-np.sum(alpha[i,1:])
        return alpha, beta

    def stage_order(self,tol=1.e-14):
        r""" 
            Returns the stage order of a Runge-Kutta method.

            The stage order is the minimum over all stages of the
            order of accuracy of that stage.  It can be shown to be
            equal to the largest integer k such that the simplifying
            assumptions $B(\\xi)$ and $C(\\xi)$ are satisfied for
            $1 \\le \\xi \\le k$.

            **References**:
                #. Dekker and Verwer
                #. [butcher2003]_
        """
        k,B,C=0,0.,0.
        while np.all(abs(B)<tol) and np.all(abs(C)<tol):
            k=k+1
            B=np.dot(self.b,self.c**(k-1))-1./k
            C=np.dot(self.A,self.c**(k-1))-self.c**k/k
        return k-1

    def circle_contractivity_radius(self,acc=1.e-13,rmax=1000):
        r""" 
            Returns the radius of circle contractivity
            of a Runge-Kutta method.
        """
        from utils import bisect

        tol=1.e-14
        r=bisect(0,rmax,acc,tol,self.is_circle_contractive)
        return r

    def absolute_monotonicity_radius(self,acc=1.e-10,rmax=200,
                    tol=3.e-16):
        r""" 
            Returns the radius of absolute monotonicity
            of a Runge-Kutta method.
        """
        from utils import bisect

        r=bisect(0,rmax,acc,tol,self.is_absolutely_monotonic)
        return r

    def linear_monotonicity_radius(self,acc=1.e-10,tol=1.e-15):
      r"""
          Computes Horvath's monotonicity radius of the stability
          function.  Not sure yet if this is M_lin of the method.
      """
      from utils import bisect

      p,q=self.stability_function()
      #First check extreme cases
      if p.order>q.order: return 0
      phi = lambda z: p(z)/q(z)
      #Get the negative real zeroes of the derivative of p/q:
      phip=p.deriv()*q-q.deriv()*p
      zeroes=[z for z in phip.r if np.isreal(z) and z<0]
      #Find the extremum of phi on (-inf,0):
      if len(zeroes)>0:
          zmax=max(abs(phi(zeroes)))
          rlo=max(zeroes)
          if p.order==q.order: 
              zmax=max(zmax, abs(p[0]/q[0]))
      else:
          if p.order<q.order: return -np.inf
          if p.order==q.order: 
              zmax=abs(p[0]/q[0])
              rlo=-10000

      phim=lambda z,tol: p(z)/q(z)<zmax

      r=bisect(rlo,0,acc,tol,phim)
      return float(np.real(r))

    def stability_function(self):
        r""" 
            Constructs the numerator and denominator of the 
            stability function of a Runge-Kutta method.

            **Output**:
                - p -- Numpy poly representing the numerator
                - q -- Numpy poly representing the denominator

            Uses the formula $\\phi(z)=p(z)/q(z)$, where

            $$p(z)=\\det(I - z A + z e b^T)$$
            $$q(z)=\\det(I - z A)$$
        """
        p1=np.poly(self.A-np.tile(self.b,(len(self),1)))
        q1=np.poly(self.A)
        p=np.poly1d(p1[::-1])    # Numerator
        q=np.poly1d(q1[::-1])    # Denominator
        return p,q

    def plot_stability_region(self,N=200,bounds=[-10,1,-5,5],
                    color='r',filled=True,scaled=False):
        r""" 
            Plot the region of absolute stability
            of a Runge-Kutta method, i.e. the set

            `\{ z \in C : |R (z)|\le 1 \}`

            where $R(z)$ is the stability function of the method.

            **Input**: (all optional)
                - N       -- Number of gridpoints to use in each direction
                - bounds  -- limits of plotting region
                - color   -- color to use for this plot
                - filled  -- if true, stability region is filled in (solid); otherwise it is outlined
        """
        p,q=self.stability_function()
        m=len(p)
        x=np.linspace(bounds[0],bounds[1],N)
        y=np.linspace(bounds[2],bounds[3],N)
        X=np.tile(x,(N,1))
        Y=np.tile(y[:,np.newaxis],(1,N))
        Z=X+Y*1j
        if scaled: R=np.abs(p(Z*m)/q(Z*m))
        else: R=np.abs(p(Z)/q(Z))
        #pl.clf()
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
        #pl.show()

    def plot_order_star(self,N=200,bounds=[-5,5,-5,5],
                    color='r',filled=True):
        r""" Plot the order star of a Runge-Kutta method,
            i.e. the set
            
            $$ \{ z \in C : |R(z)/exp(z)|\le 1 \} $$

            where $R(z)$ is the stability function of the method.

            **Input**: (all optional)
                - N       -- Number of gridpoints to use in each direction
                - bounds  -- limits of plotting region
                - color   -- color to use for this plot
                - filled  -- if true, order star is filled in (solid); otherwise it is outlined
        """
        p,q=self.stability_function()
        x=np.linspace(bounds[0],bounds[1],N)
        y=np.linspace(bounds[2],bounds[3],N)
        X=np.tile(x,(N,1))
        Y=np.tile(y[:,np.newaxis],(1,N))
        Z=X+Y*1j
        R=np.abs(p(Z)/q(Z)/np.exp(Z))
        pl.clf()
        if filled:
            pl.contourf(X,Y,R,[0,1],colors=color)
        else:
            pl.contour(X,Y,R,[0,1],colors=color)
        pl.title('Order star for '+self.name)
        pl.hold(True)
        pl.plot([0,0],[bounds[2],bounds[3]],'--k')
        pl.plot([bounds[0],bounds[1]],[0,0],'--k')
        pl.axis('Image')
        pl.hold(False)

    def is_circle_contractive(self,r,tol):
        r""" Returns 1 if the Runge-Kutta method has radius of circle
            contractivity at least $r$.
            
            **References**:
                #. Dekker and Verwer
        """
        B=np.diag(self.b)
        M=np.dot(B,self.A)+np.dot(self.A.T,B)-np.outer(self.b,self.b)
        X=M+B/r
        v,d=np.linalg.eig(X)
        if v.min()>-tol:
            return 1
        else:
            return 0

    def is_absolutely_monotonic(self,r,tol):
        r""" Returns 1 if the Runge-Kutta method is absolutely monotonic
            at $z=-r$.

            The method is absolutely monotonic if $(I+rA)^{-1}$ exists
            and
            $$K(I+rA)^{-1} \\ge 0$$
            $$(I+rA)^{-1} e_m \\ge 0$$

            where $e_m$ is the m-by-1 vector of ones and
                  K=[ A
                     b^T].

            The inequalities are interpreted componentwise.

            **References**:
                #. [kraaijevanger1991]
        """
        s=len(self)
        K=np.vstack([self.A,self.b])
        K=np.hstack([K,np.zeros([s+1,1])])
        X=np.eye(s+1)+r*K
        if abs(np.linalg.det(X))<tol: return 0
        beta_r=np.linalg.solve(X,K)
        v_r_sum = np.dot(np.eye(s+1)-r*beta_r,np.ones([s+1,1]))
        if beta_r.min()<-tol or v_r_sum.min()<-tol:
            return 0
        else:
            return 1
        # Need an exception here if rhi==rmax

    def canonical_shu_osher_form(self,r):
        r""" Return d,P where P isthe matrix P=r(I+rK)^{-1}K 
             and d is the vector d=(I+rK)^{-1}e=(I-P)e
        """
        s=len(self)
        K=np.vstack([self.A,self.b])
        K=np.hstack([K,np.zeros([s+1,1])])
        I=np.eye(s+1)
        P=np.dot(r*np.linalg.inv(I+r*K),K)
        d=(I-P).sum(1)
        return d,P

    def split(self,r,tol=1.e-15):
        s=len(self)
        I=np.eye(s+1)
        d,P=self.canonical_shu_osher_form(r)
        alpha=P*(P>0)
        alphatilde=-P*(P<0)
        x=np.dot(np.linalg.inv(I-alphatilde),2*alphatilde)
        M=np.linalg.inv(I+x)
        #Equivalently:
        M=np.dot(np.linalg.inv(I+alphatilde),I-alphatilde)
        alphanew=np.dot(M,alpha)
        dnew=np.dot(M,d)
        print np.dot(M-I,alphatilde)
        alphatildenew=np.dot(M,x-alphatilde)
        return dnew, alphanew, alphatildenew

    def is_splittable(self,r,tol=1.e-15):
        d,alpha,alphatilde=self.split(r,tol=tol)
        if max(abs(self.A[0,:]))<tol: alpha[1:,0]+=d[1:]/2.
        if alpha.min()>=-tol and d.min()>=-tol: return True
        else: return False

    def optimal_perturbed_splitting(self,acc=1.e-12,rmax=50.01,tol=1.e-13):
        r"""
            Return the optimal downwind splitting of the method
            along with the optimal downwind SSP coefficient.
        """
        from utils import bisect

        r=bisect(0,rmax,acc,tol,self.is_splittable)
        d,alpha,alphatilde=self.split(r,tol=tol)
        return r,d,alpha,alphatilde

    def is_explicit(self):
        return False

#=====================================================
class ExplicitRungeKuttaMethod(RungeKuttaMethod):
#=====================================================
    r"""
        Class for explicit Runge-Kutta methods.  Mostly identical
        to RungeKuttaMethod.
    """
    def __repr__(self): 
        """
        Pretty-prints the Butcher array in the form:
          |
        c | A
        ______
          | b
        """
        s=self.name+'\n'+self.info+'\n'
        for i in range(len(self)):
            s+='%6.3f |' % self.c[i]
            for j in range(i):
                s+=' %6.3f' % self.A[i,j]
            s+='\n'
        s+='_______|'+('________'*len(self))+'\n'
        s+= '       |'
        for j in range(len(self)):
            s+=' %6.3f' % self.b[j]
        return s

    def linear_absolute_monotonicity_radius(self,acc=1.e-10,rmax=50,
                                            tol=3.e-16):
        """ 
            Returns the radius of absolute monotonicity
            of the stability function of a Runge-Kutta method.
        """
        from utils import bisect
        p,q=self.stability_function()
        if q.order!=0 or q[0]!=1:
            print q
            print 'Not yet implemented for rational functions'
            return 0
        else:
            r=bisect(0,rmax,acc,tol,is_absolutely_monotonic_poly,p)
        return r

    def __step__(self,f,t,u,dt,x=None):
        """
            Take a time step on the ODE u'=f(t,u).

            **Input**:
                - f  -- function being integrated
                - t  -- array of previous solution times
                - u  -- array of previous solution steps (u[i,:] is the solution at time t[i])
                - dt -- length of time step to take

            **Output**:
                - unew -- approximate solution at time t[-1]+dt

            The implementation here is wasteful in terms of storage.
        """
        m=len(self)
        y=[u[-1]+0] # by adding zero we get a copy; is there a better way?
        if x is not None: fy=[f(t[-1],y[0],x)]
        else: fy=[f(t[-1],y[0])]
        for i in range(1,m):
            y.append(u[-1]+0)
            for j in range(i):
               y[i]+=self.A[i,j]*dt*fy[j]
            if x is not None: fy.append(f(t[-1]+self.c[i]*dt,y[i],x))
            else: fy.append(f(t[-1]+self.c[i]*dt,y[i]))
        if m==1: i=0 #fix just for one-stage methods
        if x is not None: fy[i]=f(t[-1]+self.c[i]*dt,y[-1],x)
        else: fy[i]=f(t[-1]+self.c[i]*dt,y[-1])
        unew=u[-1]+sum([self.b[j]*dt*fy[j] for j in range(m)])
        return unew

    def imaginary_stability_interval(self,tol=1.e-7,max=100.,eps=1.e-6):
        p,q=self.stability_function()
        zhi=max
        zlo=0.
        #Use bisection to get an upper bound:
        while (zhi-zlo)>tol:
            z=0.5*(zhi+zlo)
            mag=abs(p(1j*z))
            if (mag-1.)>eps: zhi=z
            else: zlo=z
                
        #Now check more carefully:
        zz=np.linspace(0.,z,z/0.01)
        vals=np.array(map(p,zz*1j))
        notok=np.where(vals>1.+eps)[0]
        if len(notok)==0: return z
        else: return zz[min(notok)]

    def real_stability_interval(self,tol=1.e-7,max=100.,eps=1.e-6):
        p,q=self.stability_function()
        zhi=max
        zlo=0.
        #Use bisection to get an upper bound:
        while (zhi-zlo)>tol:
            z=0.5*(zhi+zlo)
            mag=abs(p(-z))
            if (mag-1.)>eps: zhi=z
            else: zlo=z
                
        #Now check more carefully:
        zz=np.linspace(0.,z,z/0.01)
        vals=np.array(map(p,-zz))
        notok=np.where(vals>1.+eps)[0]
        if len(notok)==0: return z
        else: return zz[min(notok)]

    def is_explicit(self):
        return True

#=====================================================
#End of ExplicitRungeKuttaMethod class
#=====================================================


#=====================================================
class ExplicitRungeKuttaPair(ExplicitRungeKuttaMethod):
#=====================================================
    """
        Class for embedded Runge-Kutta pairs.  These consist of
        two methods with identical coefficients $a_{ij}$
        but different coefficients $b_j$ such that the methods
        have different orders of accuracy.  Typically the
        higher order accurate method is used to advance
        the solution, while the lower order method is
        used to obtain an error estimate.
    """

    def embedded_order(self,tol=1.e-14):
        """ 
            Returns the order of the embedded method of a Runge-Kutta pair.
        """
        p=0
        while True:
            z=self.embedded_order_conditions(p+1)
            if np.any(abs(z)>tol): return p
            p=p+1

    def embedded_order_conditions(self,p):
        """
            Generates and evaluates code to test whether a method
            satisfies the order conditions of order p (only).

            Currently uses Albrecht's recursion to generate the
            order conditions.  This is fast and requires less code
            than if they were hard-coded up to some high order,
            although it still only works up to some fixed order
            (need more recursion loops to go higher).

            Other possibilities, already in place, are:
                #. Use hard code, generated once and for all
                   by Albrecht's recursion or another method.
                   Advantages: fastest
                   Disadvantages: Less satisfying

                #. Use Butcher's recursive product on trees.
                   Advantages: Most satisfying, no maximum order
                   Disadvantages: way too slow for high order

            TODO: Decide on something and fill in this docstring.
        """
        A,b,c=self.A,self.bhat,self.c
        D=np.diag(c)
        code=runge_kutta_order_conditions(p)
        z=np.zeros(len(code)+1)
        tau=np.zeros([p,len(self)])
        for j in range(1,p):
            tau[j,:]=(c**j/j-np.dot(A,c**(j-1)))/factorial(j-1)
        for i in range(len(code)):
            exec('z[i]='+code[i])
        z[-1]=np.dot(b,c**(p-1))-1./p
        return z

    def error_coefficient(self,tree,method='principal'):
        from numpy import dot
        code=elementary_weight_str(tree)
        b=self.b
        if method=='embedded':
            b=self.bhat
        A=self.A
        c=self.c
        exec('coeff=('+code+'-1./'+str(tree.density())+')')
        return coeff/tree.symmetry()

    def error_coeffs(self,p,method='principal'):
        forest=rt.recursive_trees(p)
        err_coeffs=[]
        for tree in forest:
            err_coeffs.append(self.error_coefficient(tree,method=method))
        return err_coeffs

    def error_metrics(self):
        r"""Return full set of error metrics
            See Kennedy et. al. 2000 p. 181"""
        q=self.order(1.e-13)
        p=self.embmeth.order(1.e-13)

        tau_qp1=self.error_coeffs(q+1)
        tau_qp2=self.error_coeffs(q+2)
        tau_pp2=self.error_coeffs(p+2)

        tau_pp1_hat=self.error_coeffs(p+1,method='embedded')
        tau_pp2_hat=self.error_coeffs(p+2,method='embedded')

        A_qp1=    np.sqrt(float(np.sum(np.array(tau_qp1)**2)))
        A_qp1_max=    max([abs(tau) for tau in tau_qp1])
        A_qp2=    np.sqrt(float(np.sum(np.array(tau_qp2)**2)))
        A_qp2_max=    max([abs(tau) for tau in tau_qp2])

        A_pp1_hat=np.sqrt(float(np.sum(np.array(tau_pp1_hat)**2)))
        A_pp1_hat_max=max([abs(tau) for tau in tau_pp1_hat])

        A_pp2=    np.sqrt(float(np.sum(np.array(tau_pp2)**2)))
        A_pp2_hat=np.sqrt(float(np.sum(np.array(tau_pp2_hat)**2)))
        A_pp2_max=    max([abs(tau) for tau in tau_pp2])
        A_pp2_hat_max=max([abs(tau) for tau in tau_pp2_hat])

        B_pp2=    A_pp2_hat    /A_pp1_hat
        B_pp2_max=A_pp2_hat_max/A_pp1_hat_max

        tau2diff=np.array(tau_pp2_hat)-np.array(tau_pp2)
        C_pp2=    np.sqrt(float(np.sum(tau2diff**2)))/A_pp1_hat
        C_pp2_max=max([abs(tau) for tau in tau2diff])/A_pp1_hat_max

        D=max(np.max(self.A),np.max(self.b),np.max(self.bhat),np.max(self.c))

        E_pp2=    A_pp2    /A_pp1_hat
        E_pp2_max=A_pp2_max/A_pp1_hat_max

        return A_qp1, A_qp1_max, A_qp2, A_qp2_max, A_pp1_hat, A_pp1_hat_max, B_pp2, B_pp2_max, C_pp2, C_pp2_max, D, E_pp2, E_pp2_max

    def principal_error_norm(self):
        p=self.order(1.e-13)
        forest=rt.recursive_trees(p+1)
        errs=[]
        for tree in forest:
            errs.append(self.error_coefficient(tree))
        return np.sqrt(float(np.sum(np.array(errs)**2)))
#        return max([abs(err) for err in errs])



#=====================================================
#End of ExplicitRungeKuttaPair class
#=====================================================

#=====================================================
class RungeKuttaError(Exception):
#=====================================================
    """
        Exception class for Runge Kutta methods.
    """
    def __init__(self,msg='Runge Kutta Error'):
        self.msg=msg

    def __str__(self):
        return self.msg
#=====================================================

#=====================================================
#Functions for generating order conditions
#=====================================================
def elementary_weight(tree):
    """
        Constructs Butcher's elementary weights 
        for a Runge-Kutta method

        Currently doesn't work because of SAGE bug.

        **References**:
            [butcher2003]_
    """
    print 'Non-commutativity not working!'
    b=Symbol('b',False)
    ew=b*tree.Gprod(rt.RKeta,rt.Dmap)
    return ew

def elementary_weight_str(tree,style='python'):
    from strmanip import collect_powers, mysimp
    """
        Constructs Butcher's elementary weights for a Runge-Kutta method
        as strings suitable for numpy execution.
    """
    ewstr='dot(b,'+tree.Gprod_str(rt.RKeta_str,rt.Dmap_str)+')'
    ewstr=ewstr.replace('1*','')
    ewstr=collect_powers(ewstr,'c')
    ewstr=mysimp(ewstr)
    if style=='matlab': ewstr=python_to_matlab(ewstr)
    return ewstr

def discrete_adjoint(meth):
    """
        Returns the discrete adjoint of a Runge-Kutta method
    """
    A=np.zeros([len(meth),len(meth)])
    b=meth.b
    for i in range(len(meth)):
        for j in range(len(meth)):
            #A[i,j]=meth.A[j,i]*b[j]/b[i]
            A[i,j]=(b[i]*b[j]-meth.A[j,i]*b[j])/b[i]
    return RungeKuttaMethod(A,b)

def is_absolutely_monotonic_poly(r,tol,p):
    """ 
        Returns 1 if the polynomial p is absolutely monotonic
        at z=-r.
    """
    postest=np.arange(p.order+1)<-1
    for i in range(p.order+1):
        pdiff=p.deriv(i)
        postest[i]=pdiff(-r)>-tol
    if np.all(postest):
        return 1
    else:
        return 0

def shu_osher_change_alpha_ij(alpha,beta,i,j,val):
    """
        **Input**: 
            - alpha, beta: Shu-Osher arrays 
            - i,j: indices 
            - val -- real number

        **Output**: Shu-Osher arrays alph, bet with alph[i,j]=alpha[i,j]+val.
    """
    alph=alpha+0.
    bet=beta+0.
    alph[i,j]=alph[i,j]+val
    alph[i,0:]-=val*alph[j,0:]
    bet[i,0:] -=val* bet[j,0:]
    return alph,bet

def shu_osher_zero_alpha_ij(alpha,beta,i,j):
    """
        **Input**: Shu-Osher arrays alpha, beta
                indices i,j

        **Output**: Shu-Osher arrays alph, bet with alph[i,j]=0.
    """
    return shu_osher_change_alpha_ij(alpha,beta,i,j,-alpha[i,j])

def shu_osher_zero_beta_ij(alpha,beta,i,j):
    """
        **Input**: 
            - Shu-Osher arrays alpha, beta
            - indices i,j

        **Output**: 
            - Shu-Osher arrays alph, bet with bet[i,j]=0.
    """
    t=-beta[i,j]/beta[j,j]
    return shu_osher_change_alpha_ij(alpha,beta,i,j,-t)


def shu_osher_to_butcher(alpha,beta):
    r""" Accepts a Shu-Osher representation of an explicit Runge-Kutta
        and returns the Butcher coefficients 

        \\begin{align*}
        A = & (I-\\alpha_0)^{-1} \\beta_0 \\\\
        b = & \\beta_1 + \\alpha_1
        \\end{align*}

        **References**:  
             #. [gottlieb2009]_
    """
    m=np.size(alpha,1)
    if not np.all([np.size(alpha,0),np.size(beta,0),
                    np.size(beta,1)]==[m+1,m+1,m]):
        raise RungeKuttaError(
             'Inconsistent dimensions of Shu-Osher arrays')
    X=np.eye(m)-alpha[0:m,:]
    A=np.linalg.solve(X,beta[0:m,:])
    b=beta[m,:]+np.dot(alpha[m,:],A)
    return A,b

def loadRKM(which='All'):
    from numpy import sqrt
    """ 
        Load a set of standard Runge-Kutta methods for testing.
        The following methods are included:

        Explicit:

        'FE':         Forward Euler
        'RK44':       Classical 4-stage 4th-order
        'MTE22':      Minimal truncation error 2-stage 2nd-order
        'Heun33':     Third-order method of Heun
        'SSP22':      Trapezoidal rule 2nd-order
        'DP5':        Dormand-Prince 5th-order
        'Fehlberg45': 5th-order part of Fehlberg's pair
        'Lambert65':

        Implicit:

        'BE':         Backward Euler
        'GL2':        2-stage Gauss-Legendre
        'GL3':        3-stage Gauss-Legendre

        Also various Lobatto and Radau methods.
    """
    RK={}
    #================================================
    A=np.array([1])
    b=np.array([1])
    RK['BE']=RungeKuttaMethod(A,b,name='Implicit Euler')

    #================================================
    A=np.array([0])
    b=np.array([1])
    RK['FE']=ExplicitRungeKuttaMethod(A,b,name='Forward Euler')

    #================================================
    alpha=np.array([[0,0],[1.,0],[0.261583187659478,0.738416812340522]])
    beta=np.array([[0,0],[0.822875655532364,0],[-0.215250437021539,0.607625218510713]])
    RK['SSP22star']=ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name='SSPRK22star',
                description=
                "The optimal 2-stage, 2nd order downwind SSP Runge-Kutta method with one star")

    #================================================
    A=np.array([[1.,-sqrt(5),sqrt(5),-1.],[1.,3.,(10-7.*sqrt(5))/5.,sqrt(5)/5.],[1.,(10.+7*sqrt(5))/12.,3.,-sqrt(5)/5.],[1.,5.,5.,1.]])/12.
    b=np.array([1.,5.,5.,1.])/12.
    RK['LobattoIIIC4']=RungeKuttaMethod(A,b,name='LobattoIIIC4',
                description="The LobattoIIIC method with 4 stages")

    #================================================
    A=np.array([[1./6,-1./3,1./6],[1./6,5./12,-1./12],[1./6,2./3,1./6]])
    b=np.array([1./6,2./3,1./6])
    RK['LobattoIIIC3']=RungeKuttaMethod(A,b,name='LobattoIIIC3',
                description="The LobattoIIIC method with 3 stages")

    #================================================
    A=np.array([[1./2,-1./2],[1./2,1./2]])
    b=np.array([1./2,1./2])
    RK['LobattoIIIC2']=RungeKuttaMethod(A,b,name='LobattoIIIC2',
                description="The LobattoIIIC method with 2 stages")

    #================================================
    A=np.array([[0.,0.],[1./2,1./2]])
    b=np.array([1./2,1./2])
    RK['LobattoIIIA2']=RungeKuttaMethod(A,b,name='LobattoIIIA2',
                description="The LobattoIIIA method with 2 stages")

    #================================================
    A=np.array([[5./12,-1./12],[3./4,1./4]])
    b=np.array([3./4,1./4])
    RK['RadauIIA2']=RungeKuttaMethod(A,b,name='RadauIIA2',
                description="The RadauIIA method with 2 stages")

    #================================================
    A=np.array([[(88-7*np.sqrt(6))/360,(296-169*np.sqrt(6))/1800,(-2+3*np.sqrt(6))/225],
                [(296+169*np.sqrt(6))/1800,(88+7*np.sqrt(6))/360,(-2-3*np.sqrt(6))/225],
                [(16-np.sqrt(6))/36,(16+np.sqrt(6))/36,1/9]]);
    b=np.array([(16-np.sqrt(6))/36,(16+np.sqrt(6))/36,1/9]);
    RK['RadauIIA3']=RungeKuttaMethod(A,b,name='RadauIIA3',
                description="The RadauIIA method with 3 stages")

    #================================================
    A=np.array([[0,0],[1.,0]])
    b=np.array([1./2,1./2])
    RK['SSP22']=ExplicitRungeKuttaMethod(A,b,name='SSPRK22',
                description=
                "The optimal 2-stage, 2nd order SSP Runge-Kutta method")

    #================================================
    A=np.array([[0,0,0],[1.,0,0],[1./4,1./4,0]])
    b=np.array([1./6,1./6,2./3])
    RK['SSP33']=ExplicitRungeKuttaMethod(A,b,name='SSPRK33',
                description=
                "The optimal 3-stage, 3rd order SSP Runge-Kutta method")

    #================================================
    A=np.array([[0,0,0],[1./3,0,0],[0.,2./3,0]])
    b=np.array([1./4,0.,3./4])
    RK['Heun33']=ExplicitRungeKuttaMethod(A,b,name='Heun33',
                description= "Heun's 3-stage, 3rd order")

    #================================================
    A=np.array([[0,0,0],[1./3,0,0],[0.,1.,0]])
    b=np.array([1./2,0.,1./2])
    RK['NSSP32']=ExplicitRungeKuttaMethod(A,b,name='NSSPRK32',
                description= "Wang and Spiteri NSSP32")

    #================================================
    A=np.array([[0,0,0],[-4./9,0,0],[7./6,-1./2,0]])
    b=np.array([1./4,0.,3./4])
    RK['NSSP33']=ExplicitRungeKuttaMethod(A,b,name='NSSPRK33',
                description= "Wang and Spiteri NSSP33")

    #================================================
    m=10
    r=6.
    alpha=np.diag(np.ones(m),-1)
    alpha[5,4]=2./5
    alpha[m,m-1]=3./5
    alpha[m,4]=9./25
    alpha=alpha[:,:m]
    beta=alpha/r
    RK['SSP104']=ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,
                    name='SSPRK10,4',description=
                    "The optimal ten-stage, fourth order Runge-Kutta method")
    #================================================
    alpha=np.zeros([7,6])
    beta=np.zeros([7,6])
    alpha[1,0]=1.
    alpha[2,0:2]=[3./4,1./4]
    alpha[3,0:3]=[3./8,1./8,1./2]
    alpha[4,0:4]=[1./4,1./8,1./8,1./2]
    alpha[5,0:5]=[89537./2880000,407023./2880000,1511./12000,87./200,4./15]
    alpha[6,:]  =[4./9,1./15,0.,8./45,0.,14./45]
    beta[1,0]=1./2
    beta[2,0:2]=[0.,1./8]
    beta[3,0:3]=[-1./8,-1./16,1./2]
    beta[4,0:4]=[-5./64,-13./64,1./8,9./16]
    beta[5,0:5]=[2276219./40320000,407023./672000,1511./2800,-261./140,8./7]
    beta[6,:]  =[0.,-8./45,0.,2./3,0.,7./90]
    RK['Lambert65']=ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,
                        name='Lambert65',description='From Shu-Osher paper')
    #================================================
    A=np.array([[0,0],[2./3,0]])
    b=np.array([1./4,3./4])
    RK['MTE22']=ExplicitRungeKuttaMethod(A,b,name='Minimal Truncation Error 22')

    #================================================
    A=np.array([[0,0],[1./2,0]])
    b=np.array([0.,1.])
    RK['Mid22']=ExplicitRungeKuttaMethod(A,b,name='Midpoint Runge-Kutta')

    #================================================
    A=np.array([[0,0,0,0],[1./2,0,0,0],[0,1./2,0,0],[0,0,1,0]])
    b=np.array([1./6,1./3,1./3,1./6])
    RK['RK44']=ExplicitRungeKuttaMethod(A,b,name='Classical RK4')

    #================================================
    A=np.array([[0,0,0,0,0,0],[1/4.,0,0,0,0,0],[1/8.,1/8.,0,0,0,0],
         [0,0,1/2.,0,0,0],[3/16.,-3/8.,3/8.,9/16.,0,0],
         [-3/7.,8/7.,6/7.,-12/7.,8/7.,0]])
    b=np.array([7/90.,0,16/45.,2/15.,16/45.,7/90.])
    RK['BuRK65']=ExplicitRungeKuttaMethod(A,b,name="Butcher's RK65")

    #================================================
    A=np.array([[1/4.,1/4.-np.sqrt(3.)/6.],[1/4.+np.sqrt(3.)/6.,1/4.]])
    b=np.array([1/2.,1/2.])
    RK['GL2']=RungeKuttaMethod(A,b,name="Gauss-Legendre RK24")

    #================================================
    A=np.array([[5/36.,(80-24*np.sqrt(15.))/360.,(50-12*np.sqrt(15.))/360.],
         [(50+15*np.sqrt(15.))/360.,2/9.,(50-15*np.sqrt(15.))/360.],
         [(50+12*np.sqrt(15.))/360.,(80+24*np.sqrt(15.))/360.,5/36.]])
    b=np.array([5/18.,4/9.,5/18.])
    RK['GL3']=RungeKuttaMethod(A,b,name="Gauss-Legendre RK36")
    #================================================
    A=np.array([[0,0,0,0,0,0],[1./4,0,0,0,0,0],[3./32,9./32,0,0,0,0],
        [1932./2197,-7200/2197,7296./2197,0,0,0],
        [439./216,-8.,3680./513,-845./4104,0,0],
        [-8./27,2.,-3544./2565,1859./4104,-11./40,0]])
    b=np.array([16./135,0,6656./12825,28561./56430,-9./50,2./55])
    RK['Fehlberg45']=ExplicitRungeKuttaMethod(A,b,name='Fehlberg RK45')
    #================================================
    A=np.array([[0,0,0,0,0,0,0],[1./5,0,0,0,0,0,0],[3./40,9./40,0,0,0,0,0],
        [44./45,-56/15,32./9,0,0,0,0],
        [19372./6561,-25360./2187,64448./6561,-212./729,0,0,0],
        [9017./3168,-355./33,46732./5247,49./176,-5103./18656,0,0],
        [35./384,0.,500./1113,125./192,-2187./6784,11./84,0]])
    b=np.array([35./384,0.,500./1113,125./192,-2187./6784,11./84,0])
    RK['DP5']=ExplicitRungeKuttaMethod(A,b,name='Dormand-Prince RK75')


    if which=='All':
        return RK
    else:
        return RK[which]

def SSPRK2(m):
    """ Construct the optimal m-stage, second order SSP 
        Explicit Runge-Kutta method (m>=2).

        **Input**: m -- number of stages
        **Output**: A ExplicitRungeKuttaMethod

        **Examples**::
            
            Load the 4-stage method:
            >>> SSP42=SSPRK2(4)
            >>> SSP42

            SSPRK42

             0.000 |  0.000  0.000  0.000  0.000
             0.333 |  0.333  0.000  0.000  0.000
             0.667 |  0.333  0.333  0.000  0.000
             1.000 |  0.333  0.333  0.333  0.000
            _______|____________________________
                   |  0.250  0.250  0.250  0.250

            >>> SSP42.absolute_monotonicity_radius()
            2.9999999999745341

        **References**: 
            #. [ketcheson2008]_
    """
    assert m>=2, "SSPRKm2 methods must have m>=2"
    r=m-1.
    alpha=np.vstack([np.zeros(m),np.eye(m)])
    alpha[m,m-1]=(m-1.)/m
    beta=alpha/r
    alpha[m,0]=1./m
    name='SSPRK'+str(m)+'2'
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name)

def SSPIRK1(m):
    """ Construct the m-stage, first order unconditionally SSP 
        Implicit Runge-Kutta method with smallest 
        coefficient of z^2 (in the stability polynomial)

        **Input**: m -- number of stages
        **Output**: A RungeKuttaMethod

        **Examples**::
            
            Load the 4-stage method:
            >>> ISSP41=SSPIRK1(4)
            >>> ISSP41

            SSPIRK41

             0.250 |  0.250  0.000  0.000  0.000
             0.500 |  0.250  0.250  0.000  0.000
             0.750 |  0.250  0.250  0.250  0.000
             1.000 |  0.250  0.250  0.250  0.250
            _______|____________________________
                   |  0.250  0.250  0.250  0.250
    """
    A=np.tri(m)/m
    b=np.ones(m)/m
    name='SSPIRK'+str(m)+'1'
    return RungeKuttaMethod(A,b,name=name)


def SSPIRK2(m):
    """ Construct the optimal m-stage, second order SSP 
        Implicit Runge-Kutta method (m>=2).

        **Input**: m -- number of stages
        **Output**: A RungeKuttaMethod

        **Examples**::
            
            Load the 4-stage method:
            >>> ISSP42=SSPIRK2(4)
            >>> ISSP42

            SSPIRK42

             0.000 |  0.000  0.000  0.000  0.000
             0.333 |  0.333  0.000  0.000  0.000
             0.667 |  0.333  0.333  0.000  0.000
             1.000 |  0.333  0.333  0.333  0.000
            _______|____________________________
                   |  0.250  0.250  0.250  0.250

            >>> ISSP42.absolute_monotonicity_radius()
            2.9999999999745341

        **References**:
            #. [ketcheson2009]_
    """
    r=2.*m
    alpha=np.vstack([np.zeros(m),np.eye(m)])
    beta=alpha/r
    for i in range(m): beta[i,i]=1./r
    name='SSPIRK'+str(m)+'2'
    return RungeKuttaMethod(alpha=alpha,beta=beta,name=name)


def SSPIRK3(m):
    """ Construct the optimal m-stage, third order SSP 
        Implicit Runge-Kutta method (m>=2).

        **Input**: m -- number of stages
        **Output**: A RungeKuttaMethod

        **Examples**::
            
            Load the 4-stage method:
            >>> ISSP43=SSPIRK3(4)
            >>> ISSP43

            SSPIRK43

             0.113 |  0.113  0.000  0.000  0.000
             0.371 |  0.258  0.113  0.000  0.000
             0.629 |  0.258  0.258  0.113  0.000
             0.887 |  0.258  0.258  0.258  0.113
            _______|____________________________
                   |  0.250  0.250  0.250  0.250

            >>> ISSP43.absolute_monotonicity_radius()
            6.8729833461475209

        **References**:
            #. [ketcheson2009]_
    """
    r=m-1+np.sqrt(m**2-1)
    alpha=np.vstack([np.zeros(m),np.eye(m)])
    alpha[-1,-1]=((m+1)*r)/(m*(r+2))
    beta=alpha/r
    for i in range(m): beta[i,i]=1./2*(1-np.sqrt((m-1.)/(m+1.)))
    name='SSPIRK'+str(m)+'3'
    return RungeKuttaMethod(alpha=alpha,beta=beta,name=name)

def RK22_family(gamma):
    """ 
        Construct a 2-stage second order Runge-Kutta method 

        **Input**: w -- family parameter
        **Output**: An ExplicitRungeKuttaMethod

    """
    A=np.array([[0,0],[1./(2.*gamma),0]])
    b=np.array([1.-gamma,gamma])
    return ExplicitRungeKuttaMethod(A,b)

def RK44_family(w):
    """ 
        Construct a 4-stage fourth order Runge-Kutta method 

        **Input**: w -- family parameter
        **Output**: An ExplicitRungeKuttaMethod

    """
    A=np.array([[0,0,0,0],[1./2,0,0,0],[1./2-1./(6*w),1./(6*w),0,0],
                [0,1.-3.*w,3.*w,0]])
    b=np.array([1./6,2./3-w,w,1./6])
    return ExplicitRungeKuttaMethod(A,b)


def SSPRK3(m):
    """ 
        Construct the optimal m-stage third order SSP
        Runge-Kutta method (m=n**2, n>=2) 

        **Input**: m -- number of stages
        **Output**: A RungeKuttaMethod

        **Examples**::

            Load the 4-stage method:
            >>> SSP43=SSPRK3(4)

            Runge-Kutta Method

            0.000 |  0.000  0.000  0.000  0.000
            0.500 |  0.500  0.000  0.000  0.000
            1.000 |  0.500  0.500  0.000  0.000
            0.500 |  0.167  0.167  0.167  0.000
           _______|____________________________
                  |  0.167  0.167  0.167  0.500


        **References**: 
            #. [ketcheson2008]_

    """
    n=np.sqrt(m)
    assert n==round(n), "SSPRKm3 methods must have m=n^2"
    assert m>=4, "SSPRKm3 methods must have m>=4"
    r=float(m-n)
    alpha=np.vstack([np.zeros(m),np.eye(m)])
    alpha[n*(n+1)/2,n*(n+1)/2-1]=(n-1.)/(2*n-1.)
    beta=alpha/r
    alpha[n*(n+1)/2,(n-1)*(n-2)/2]=n/(2*n-1.)
    name='SSPRK'+str(m)+'3'
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name)


if __name__== "__main__":
    RK=loadRKM()


def RKC1(m,eps=0):
    """ Construct the m-stage, first order 
        Explicit Runge-Kutta-Chebyshev methods of Verwer (m>=1).

        **Input**: m -- number of stages
        **Output**: A ExplicitRungeKuttaMethod

        **Examples**::
            
            Load the 4-stage method:
            >>> RKC41=RKC1(4)
            >>> RKC41

            RKC41

             0.000 |  0.000  0.000  0.000  0.000
             0.333 |  0.333  0.000  0.000  0.000
             0.667 |  0.333  0.333  0.000  0.000
             1.000 |  0.333  0.333  0.333  0.000
            _______|____________________________
                   |  0.250  0.250  0.250  0.250

        **References**: 
            #. [verwer2004]_
    """

    import scipy.special.orthogonal as orth

    Tm=orth.chebyt(m)
    w0=1.+eps/m**2
    w1=Tm(w0)/Tm.deriv()(w0)

    alpha=np.zeros([m+1,m])
    beta=np.zeros([m+1,m])

    b=np.zeros(m+1)
    a=np.zeros(m+1)
    mu=np.zeros(m+1)
    nu=np.zeros(m+1)
    mut=np.zeros(m+1)
    gamt=np.zeros(m+1)

    b[0]=1.
    b[1]=1./w0
    mut[1] = b[1]*w1
    alpha[1,0]=1.
    beta[1,0]=mut[1]

    for j in range(2,m+1):
      b[j] = 1./orth.eval_chebyt(j,w0)
      a[j] = 1.-b[j]*orth.eval_chebyt(j,w0)
      mu[j]= 2.*b[j]*w0/b[j-1]
      nu[j]= -b[j]/b[j-2]
      mut[j] = mu[j]*w1/w0
      gamt[j] = -a[j-1]*mut[j]

      alpha[j,0]=1.-mu[j]-nu[j]
      alpha[j,j-1]=mu[j]
      alpha[j,j-2]=nu[j]
      beta[j,j-1]=mut[j]
      beta[j,0]=gamt[j]

    name='RKC'+str(m)+'1'
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name)


def RKC2(m,eps=0):
    """ Construct the m-stage, second order 
        Explicit Runge-Kutta-Chebyshev methods of Verwer (m>=2).

        **Input**: m -- number of stages
        **Output**: A ExplicitRungeKuttaMethod

        **Examples**::
            
            Load the 4-stage method:
            >>> RKC42=RKC2(4)
            >>> RKC42

            RKC42

             0.000 |  0.000  0.000  0.000  0.000
             0.333 |  0.333  0.000  0.000  0.000
             0.667 |  0.333  0.333  0.000  0.000
             1.000 |  0.333  0.333  0.333  0.000
            _______|____________________________
                   |  0.250  0.250  0.250  0.250

        **References**: 
            #. [verwer2004]_
    """

    import scipy.special.orthogonal as orth

    Tm=orth.chebyt(m)
    w0=1.+eps/m**2
    w1=Tm.deriv()(w0)/Tm.deriv(2)(w0)

    alpha=np.zeros([m+1,m])
    beta=np.zeros([m+1,m])

    b=np.zeros(m+1)
    a=np.zeros(m+1)
    mu=np.zeros(m+1)
    nu=np.zeros(m+1)
    mut=np.zeros(m+1)
    gamt=np.zeros(m+1)

    T2=orth.chebyt(2)
    b[0]=T2.deriv(2)(w0)/(T2.deriv()(w0))**2
    b[1]=1./w0
    mut[1] = b[1]*w1
    alpha[1,0]=1.
    beta[1,0]=mut[1]

    for j in range(2,m+1):
      Tj=orth.chebyt(j)
      b[j] = Tj.deriv(2)(w0)/(Tj.deriv()(w0))**2
      a[j] = 1.-b[j]*orth.eval_chebyt(j,w0)
      mu[j]= 2.*b[j]*w0/b[j-1]
      nu[j]= -b[j]/b[j-2]
      mut[j] = mu[j]*w1/w0
      gamt[j] = -a[j-1]*mut[j]

      alpha[j,0]=1.-mu[j]-nu[j]
      alpha[j,j-1]=mu[j]
      alpha[j,j-2]=nu[j]
      beta[j,j-1]=mut[j]
      beta[j,0]=gamt[j]

    name='RKC'+str(m)+'1'
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name)

def dcweights(x):
    """
      Takes a set of abscissae x and an index i, and returns
      the quadrature weights for the interval [x_i,x_{i+1}].
      Used in construction of deferred correction methods.
    """

    #Form the vanderMonde matrix:
    A=np.vander(x).T
    A=A[::-1,:]
    F=0*A
    n=np.arange(len(x))+1
    for i in range(len(x)-1):
        a=x[i]; b=x[i+1];
        f=(b**n-a**n)/n
        F[:,i]=f
    w=np.linalg.solve(A,F)

    return w

def DC(s,theta=0.,grid='eq'):
    """ Construct deferred correction methods.
        For now, based on explicit Euler and equispaced points.
        TODO: generalize base method and grid.

        **Input**: s -- number of grid points & number of correction iterations

        **Output**: A ExplicitRungeKuttaMethod

        **Examples**::
            
        **References**: 
            #. [dutt2000]_
            #. [gottlieb2009]_
    """

    # Choose the grid:
    if grid=='eq':
        t=np.linspace(0.,1.,s+1) # Equispaced
    elif grid=='cheb':
        t=0.5*(np.cos(np.arange(0,s+1)*np.pi/s)+1.)  #Chebyshev
        t=t[::-1]
    dt=np.diff(t)

    m=s
    alpha=np.zeros([s**2+m+1,s**2+m])
    beta=np.zeros([s**2+m+1,s**2+m])

    w=dcweights(t)       #Get the quadrature weights for our grid
                         #w[i,j] is the weight of node i for the integral
                         #over [x_j,x_j+1]

    #first iteration (k=1)
    for i in range(1,s+1):
        alpha[i,i-1]=1.
        beta[i,i-1]=dt[i-1]

    #subsequent iterations:
    for k in range(1,s+1):
        beta[s*k+1,0]=w[0,0]
        for i in range(1,s+1):
            alpha[s*k+1,0]=1.
            beta[s*k+1,s*(k-1)+i]=w[i,0]

        for m in range(1,s):
            alpha[s*k+m+1,s*k+m] = 1.
            beta[s*k+m+1,s*k+m] = theta
            beta[s*k+m+1,0]=w[0,m]
            for i in range(1,s+1):
                beta[s*k+m+1,s*(k-1)+i]=w[i,m]
                if i==m:
                    beta[s*k+m+1,s*(k-1)+i]-=theta

    name='DC'+str(s)*2
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name)


def extrap(s,seq='harmonic'):
    """ Construct extrapolation methods.
        For now, based on explicit Euler, but allowing arbitrary sequences.
        TODO: 
            - generalize base method
            - Eliminate the unnecessary stages, and make the construction
                more numerically stable

        **Input**: s -- number of grid points & number of correction iterations

        **Output**: A ExplicitRungeKuttaMethod

        **Examples**::
            
        **References**: 
            #. [Hairer]_ chapter II.9
    """

    if seq=='harmonic': N=np.arange(s)+1; 
    elif seq=='Romberg': N=np.arange(s)+1;  N=2**(N-1)

    J=np.cumsum(N)+1

    nrs = J[-1]

    alpha=np.zeros([nrs+s*(s-1)/2.,nrs+s*(s-1)/2.-1])
    beta=np.zeros([nrs+s*(s-1)/2.,nrs+s*(s-1)/2.-1])


    alpha[1,0]=1.
    beta[1,0]=1./N[0]

    for j in range(1,len(N)):
        #Form T_j1:
        alpha[J[j-1],0] = 1.
        beta[J[j-1],0]=1./N[j]

        for i in range(1,N[j]):
            alpha[J[j-1]+i,J[j-1]+i-1]=1.
            beta[J[j-1]+i,J[j-1]+i-1]=1./N[j]
    
    #We have formed the T_j1
    #Now form the rest
    #
    #Really there are no more "stages", and we could form T_ss directly
    #but we need to work out the formula
    #This is a numerically unstable alternative (fails for s>5)

    for j in range(1,s):
        #form T_{j+1,2}:
        alpha[nrs-1+j,J[j]-1]=1.+1./(N[j]/N[j-1]-1.)
        alpha[nrs-1+j,J[j-1]-1]=-1./(N[j]/N[j-1]-1.)

    #Now form all the rest, up to T_ss

    nsd = nrs-1+s
    for k in range(2,s):
        for ind,j in enumerate(range(k,s)):
            #form T_{j+1,k+1}:
            alpha[nsd+ind,nsd-(s-k)+ind] = 1+1./(N[j]/N[j-k]-1.)
            alpha[nsd+ind,nsd-(s-k)+ind-1] = -1./(N[j]/N[j-k]-1.)
        nsd += s-k

    print alpha
    print beta

    name='extrap'+str(s)
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name)

def SSPIRK1(m):
    """ Construct the m-stage, first order unconditionally SSP 
        Implicit Runge-Kutta method with smallest 
        coefficient of z^2 (in the stability polynomial)

        **Input**: m -- number of stages
        **Output**: A RungeKuttaMethod

        **Examples**::
            
            Load the 4-stage method:
            >>> ISSP41=SSPIRK1(4)
            >>> ISSP41

            SSPIRK41

             0.250 |  0.250  0.000  0.000  0.000
             0.500 |  0.250  0.250  0.000  0.000
             0.750 |  0.250  0.250  0.250  0.000
             1.000 |  0.250  0.250  0.250  0.250
            _______|____________________________
                   |  0.250  0.250  0.250  0.250
    """
    A=np.tri(m)/m
    b=np.ones(m)/m
    name='SSPIRK'+str(m)+'1'
    return RungeKuttaMethod(A,b,name=name)





def rk_order_conditions_hardcoded(self,p,tol):
    """ 
        Returns a vector that is identically zero if the
        Runge-Kutta method satisfies the conditions of order p (only) 

        This function involves explicitly coded order conditions up to
        order six.  It is deprecated for now.
    """
    print 'rk_order_conditions_hardcoded: This function is deprecated!'
    A=self.A
    b=self.b
    c=self.c
    if p==1:
        z=sum(b)-1.
    if p==2:
        z=sum(np.dot(b,c))-1/2.
    if p==3:
        z=np.zeros(2)
        z[0]=0.5*np.dot(c**2,b)-1/6.
        z[1]=np.dot(b,np.dot(A,c))-1/6.
    if p==4:
        z=np.zeros(4)
        z[0]=1/6.*np.dot(b,c**3)-1/24.
        z[1]=np.dot(b*c,np.dot(A,c))-1/8.
        z[2]=1/2.*np.dot(b,np.dot(A,c**2))-1/24.
        z[3]=np.dot(b,np.dot(np.dot(A,A),c))-1/24.
    if p==5:
        z=np.zeros(9)
        z[0]=1/24.*np.dot(c**4,b)                        -1/120.
        z[1]=1/2. *np.dot(b*c**2,np.dot(A,c))               -1/20.
        z[2]=1/2. *np.dot(b,np.dot(A,c)**2)                 -1/40.
        z[3]=1/2. *np.dot(b*c,np.dot(A,c**2))               -1/30.
        z[4]=1/6. *np.dot(b,np.dot(A,c**3))                 -1/120.
        z[5]=      np.dot(b*c,np.dot(np.dot(A,A),c))           -1/30.
        z[6]=      np.dot(b,np.dot(A,np.dot(np.diag(c),np.dot(A,c))))-1/40.
        z[7]=1/2. *np.dot(b,np.dot(np.dot(A,A),c**2))          -1/120.
        z[8]=      np.dot(b,np.dot(np.dot(A,np.dot(A,A)),c))      -1/120.
    if p==6:
        z=np.zeros(20)
        z[0]=1/120.*np.dot(c**5,b)-1/720.
        z[1]=1/6.*np.dot(b,np.dot(np.diag(c**3),np.dot(A,c)))-1/72.
        z[2]=1/2.*np.dot(b,np.dot(np.diag(c),np.dot(A,c)**2))-1/48.
        z[3]=1/4.*np.dot(b,np.dot(np.diag(c**2),np.dot(A,c**2)))-1/72.
        z[4]=1/2.*np.dot(b,np.dot(A,c**2)*np.dot(A,c))-1/72.
        z[5]=1/6.*np.dot(b,np.dot(np.diag(c),np.dot(A,c**3)))-1/144.
        z[6]=1/24.*np.dot(b,np.dot(A,c**4))-1/720.
        z[7]=1/2.*np.dot(b,np.dot(np.diag(c**2),np.dot(np.dot(A,A),c)))-1/72.
        z[8]=np.dot(b,(np.dot(np.dot(A,A),c)*np.dot(A,c)))-1/72.
        z[9]=np.dot(b,np.dot(np.diag(c),np.dot(A,np.dot(np.diag(c),np.dot(A,c)))))-1/48.
        z[10]=1/2.*np.dot(b,np.dot(A,np.dot(np.diag(c**2),np.dot(A,c))))-1/120.
        z[11]=1/2.*np.dot(b,np.dot(A,np.dot(A,c)**2))-1/240.
        z[12]=1/2.*np.dot(b,np.dot(np.diag(c),np.dot(A,np.dot(A,c**2))))-1/144.
        z[13]=1/2.*np.dot(b,np.dot(A,np.dot(np.diag(c),np.dot(A,c**2))))-1/180.
        z[14]=1/6.*np.dot(b,np.dot(A,np.dot(A,c**3)))-1/720.
        z[15]=np.dot(b,np.dot(np.diag(c),np.dot(A,np.dot(A,np.dot(A,c)))))-1/144.
        z[16]=np.dot(b,np.dot(A,np.dot(np.diag(c),np.dot(A,np.dot(A,c)))))-1/180.
        z[17]=np.dot(b,np.dot(A,np.dot(A,np.dot(np.diag(c),np.dot(A,c)))))-1/240.
        z[18]=1/2.*np.dot(b,np.dot(A,np.dot(A,np.dot(A,c**2))))-1/720.
        z[19]=np.dot(b,np.dot(A,np.dot(A,np.dot(A,np.dot(A,c)))))-1/720.
    #Need exception for p>6 or p<1
    if p>6:
        z=1
    return z


def runge_kutta_order_conditions(p,ind='all'):
    """ 
        This is the current method of producing the code on-the-fly
        to test order conditions for RK methods.  May be deprecated
        soon.
    """
    strings=rt.recursiveVectors(p,'all')
    code=[]
    for oc in strings:
         code.append(RKOCstr2code(oc))
    return code

def RKOCstr2code(ocstr):
    """ 
        Converts output of runge_kutta_order_conditions() to
        numpy-executable code.
    """
    factors=ocstr.split(',')
    occode='np.dot(b,'
    for factor in factors[0:len(factors)-1]:
         occode=occode+'np.dot('+factor+','
    occode=occode+factors[len(factors)-1]
    occode=occode.replace(']',',:]')
    occode=occode+')'*len(factors)
    return occode

def compose(RK1,RK2,h1=1,h2=1):
    """ Multiplication is interpreted as composition:
    RK1*RK2 gives the method obtained by applying
    RK2, followed by RK1, each with half the timestep.

    **Output**:
        The method
             c_2 | A_2  0
           1+c_1 | b_2 A_1
           _____________
                 | b_2 b_1

        but with everything divided by two.
        The b_2 matrix block consists of m_1 (row) copies of b_2.


    TODO: Think about whether this is the right thing to return.
    """
    f1=h1/(h1+h2)
    f2=h2/(h1+h2)
    A=np.vstack([
    np.hstack([RK2.A*f2,np.zeros([np.size(RK2.A,0),np.size(RK1.A,1)])]),
        np.hstack([np.tile(RK2.b*f2,(len(RK1),1)),RK1.A*f1])]).squeeze()
    b=np.hstack([RK2.b*f2,RK1.b*f1]).squeeze()
    return RungeKuttaMethod(A,b)

def plot_rational_stability_region(p,q,N=200,bounds=[-10,1,-5,5],
                          color='r',filled=True,scaled=False):
    r""" 
        Plot the region of absolute stability
        of a rational function i.e. the set

        `\{ z \in C : |R (z)|\le 1 \}`

            where $R(z)=p(z)/q(z)$ is the rational function.

            **Input**: (all optional)
                - N       -- Number of gridpoints to use in each direction
                - bounds  -- limits of plotting region
                - color   -- color to use for this plot
                - filled  -- if true, stability region is filled in (solid); otherwise it is outlined
    """
    m=len(p)
    print m
    x=np.linspace(bounds[0],bounds[1],N)
    y=np.linspace(bounds[2],bounds[3],N)
    X=np.tile(x,(N,1))
    Y=np.tile(y[:,np.newaxis],(1,N))
    Z=X+Y*1j
    if not scaled: R=np.abs(p(Z)/q(Z))
    else: R=np.abs(p(Z*m)/q(Z*m))
    #pl.clf()
    if filled:
        pl.contourf(X,Y,R,[0,1],colors=color)
    else:
        pl.contour(X,Y,R,[0,1],colors=color)
    pl.hold(True)
    pl.plot([0,0],[bounds[2],bounds[3]],'--k',linewidth=2)
    pl.plot([bounds[0],bounds[1]],[0,0],'--k',linewidth=2)
    pl.axis('Image')
    pl.hold(False)
    #pl.show()

def python_to_matlab(code):
    r"""
        Convert python code string (order condition) to matlab code string
        Doesn't really work yet.  We need to do more parsing.
    """
    print code
    outline=code
    outline=outline.replace("**",".^")
    outline=outline.replace("*",".*")
    outline=outline.replace("dot(b,","b'*(")
    outline=outline.replace("dot(bhat,","bhat'*(")
    outline=outline.replace("dot(Ahat,","Ahat*(")
    outline=outline.replace("dot(A,","(A*(")
    outline=outline.replace("( c)","c")
    outline=outline.replace("-0","")
    print outline
    print '******************'
    return outline
