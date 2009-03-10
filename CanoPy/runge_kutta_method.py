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
import pdb

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
            #. *J. C. Butcher, "Numerical Methods for Ordinary Differential Equations"*
            #. *Hairer & Wanner, "Solving Ordinary Differential Equations I: Nonstiff Problems" Chapter II*
    """
    def __init__(self,A=None,b=None,alpha=None,beta=None,
            name='Runge-Kutta Method',description=''):
        r"""
            A Runge-Kutta Method is initialized by providing either:
                #. Butcher arrays $A$ and $b$ with valid and consistent 
                   dimensions; or
                #. Shu-Osher arrays `\alpha` and `\beta` with valid and
                   consistent dimensions but not both.
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
            A,b=shu_osher_to_butcher(alpha,beta)
        # Set Butcher arrays
        self.A=A
        self.b=b
        if len(self)>1:
            self.c=np.sum(A,1)
        else:
            self.c=A
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

    def __mul__(self,RK2):
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
        A=np.vstack([
            hstack([RK2.A,np.zeros([np.size(RK2.A,0),np.size(self.A,1)])]),
            hstack([np.tile(RK2.b,(len(self),1)),self.A])])
        b=hstack([RK2.b,self.b])
        return RungeKuttaMethod(A=A/2.,b=b/2.)

    def error_coefficient(self,tree):
        from numpy import dot
        code=elementary_weight_str(tree)
        b=self.b
        A=self.A
        c=self.c
        exec('coeff=('+code+'-1./'+str(tree.density())+')')
        return coeff/tree.symmetry()

    def principal_error_norm(self):
        p=self.order(1.e-13)
        forest=rt.recursive_trees(p+1)
        errs=[]
        for tree in forest:
            errs.append(self.error_coefficient(tree))
        return np.sqrt(np.sum(np.array(errs)**2))
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
            1. Use hard code, generated once and for all
                by Albrecht's recursion or another method.
                Advantages: fastest
                Disadvantages: Less satisfying
            2. Use Butcher's recursive product on trees.
                Advantages: Most satisfying, no maximum order
                Disadvantages: way too slow for high order

            TODO: Decide on something and fill in this docstring.
        """
        A,b,c=self.A,self.b,self.c
        D=np.diag(c)
        code=runge_kutta_order_conditions(p)
        z=np.zeros(len(code)+1)
        gamma=np.zeros([p,len(self)])
        for j in range(1,p):
            gamma[j,:]=(c**j/j-np.dot(A,c**(j-1)))/factorial(j-1)
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

    def standard_shu_osher_form(self):
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
                #. *I. Higueras, "Representations of Runge-Kutta methods and strong stability preserving methods", SINUM 43 pp. 924-948 (2005)*

        """
        r=self.absolute_monotonicity_radius()
        K=np.vstack([self.A,self.b])
        X=np.eye(len(self))+r*self.A
        beta=np.linalg.solve(X.T,K.T).T
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
                #. Butcher
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
        tol=1.e-14
        r=bisect(0,rmax,acc,tol,self.is_circle_contractive)
        return r

    def absolute_monotonicity_radius(self,acc=1.e-10,rmax=50,
                    tol=3.e-16):
        r""" 
            Returns the radius of absolute monotonicity
            of a Runge-Kutta method.
        """
        r=bisect(0,rmax,acc,tol,self.is_absolutely_monotonic)
        return r

    def stability_function(self):
        r""" 
            Constructs the numerator and denominator of the 
            stability function of a Runge-Kutta method.

            **Output**:
                - p -- Numpy poly representing the numerator
                * q -- Numpy poly representing the denominator

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
                    color='r',filled=True):
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
        x=np.linspace(bounds[0],bounds[1],N)
        y=np.linspace(bounds[2],bounds[3],N)
        X=np.tile(x,(N,1))
        Y=np.tile(y[:,np.newaxis],(1,N))
        Z=X+Y*1j
        R=np.abs(p(Z)/q(Z))
        pl.clf()
        if filled:
            pl.contourf(X,Y,R,[0,1],colors=color)
        else:
            pl.contour(X,Y,R,[0,1],colors=color)
        pl.title('Absolute Stability Region for '+self.name)
        pl.hold(True)
        pl.plot([0,0],[bounds[2],bounds[3]],'--k')
        pl.plot([bounds[0],bounds[1]],[0,0],'--k')
        pl.axis('Image')
        pl.hold(False)
        pl.show()

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
        pl.title('Absolute Stability Region for '+self.name)
        pl.hold(True)
        pl.plot([0,0],[bounds[2],bounds[3]],'--k')
        pl.plot([bounds[0],bounds[1]],[0,0],'--k')
        pl.axis('Image')
        pl.hold(False)
        pl.show()

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
               $$rK(I+rA)^{-1} e_m \\le e_{m+1}$$

            where $e_m$ is the m-by-1 vector of ones and
                  K=[ A
                     b^T].
            The inequalities are interpreted componentwise.

            **References**:
                #. JFBM Kraaijevanger, "Contractivity of Runge-Kutta Methods", BIT 1991
        """
        K=np.vstack([self.A,self.b])
        X=np.eye(len(self))+r*self.A
        beta=np.linalg.solve(X.T,K.T).T
        ech=r*K*np.linalg.solve(X,np.ones(len(self)))
        if beta.min()<-tol or ech.max()-1>tol:
            return 0
        else:
            return 1
        # Need an exception here if rhi==rmax


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
        p,q=self.stability_function()
        if q.order!=0 or q[0]!=1:
            print 'Not yet implemented for rational functions'
            return 0
        else:
            r=bisect(0,rmax,acc,tol,is_absolutely_monotonic_poly,p)
        return r

    def __step__(self,f,t,u,dt):
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
        fy=[f(t[-1],y[0])]
        for i in range(1,m):
            y.append(u[-1]+0)
            for j in range(i):
               y[i]+=self.A[i,j]*dt*fy[j]
            fy.append(f(t[-1]+self.c[i]*dt,y[i]))
        fy[i]=f(t[-1]+self.c[i]*dt,y[-1])
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
            Butcher
    """
    print 'Non-commutativity not working!'
    b=Symbol('b',False)
    ew=b*tree.Gprod(rt.RKeta,rt.Dmap)
    return ew

def elementary_weight_str(tree):
    from strmanip import collect_powers
    """
        Constructs Butcher's elementary weights for a Runge-Kutta method
        as strings suitable for numpy execution.
    """
    ewstr='dot(b,'+tree.Gprod_str(rt.RKeta_str,rt.Dmap_str)+')'
    ewstr=ewstr.replace('1*','')
    ewstr=collect_powers(ewstr,'c')
    return ewstr


def bisect(rlo, rhi, acc, tol, fun, params=None):
    """ 
        Performs a bisection search.

        **Input**:
            - fun -- a function such that fun(r)==True iff x_0>r, where x_0 is the value to be found.
    """
    while rhi-rlo>acc:
        r=0.5*(rhi+rlo)
        if params: isvalid=fun(r,tol,params)
        else: isvalid=fun(r,tol)
        if isvalid:
            rlo=r
        else:
            rhi=r
    return rlo

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
    return shu_osher_change_alpha_ij(alpha,beta,i,j,-alpha[i,j])


def shu_osher_to_butcher(alpha,beta):
    r""" Accepts a Shu-Osher representation of an explicit Runge-Kutta
        and returns the Butcher coefficients 

        \\begin{align*}
        A = & (I-\\alpha_0)^{-1} \\beta_0 \\\\
        b = & \\beta_1 + \\alpha_1
        \\end{align*}

        **References**:  Gottlieb, Ketcheson, & Shu, "High Order Strong 
             Stability Preserving Time Discretizations", J. Sci. Comput. 2008
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
    """ 
        Load a set of standard Runge-Kutta methods for testing.

        TODO: 
            - Others?
    """
    RK={}
    #================================================
    A=np.array([0])
    b=np.array([1])
    RK['FE11']=ExplicitRungeKuttaMethod(A,b,name='Forward Euler')

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

        **References**: D.I. Ketcheson, "Highly efficient strong stability 
                preserving Runge-Kutta methods with low-storage
                implementations", SISC 2008 [Ketcheson2008]
    """
    assert m>=2, "SSPRKm2 methods must have m>=2"
    r=m-1.
    alpha=np.vstack([np.zeros(m),np.eye(m)])
    alpha[m,m-1]=(m-1.)/m
    beta=alpha/r
    alpha[m,0]=1./m
    name='SSPRK'+str(m)+'2'
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name)

def SSPIRK2(m):
    """ Construct the optimal m-stage, second order SSP 
        Implicit Runge-Kutta method (m>=2).

        **Input**: m -- number of stages
        **Output**: A RungeKuttaMethod

        **Examples**::
            
            Load the 4-stage method:
            >>> ISSP42=SSPIRK2(4)
            >>> ISSP42

            SSPRK42

             0.000 |  0.000  0.000  0.000  0.000
             0.333 |  0.333  0.000  0.000  0.000
             0.667 |  0.333  0.333  0.000  0.000
             1.000 |  0.333  0.333  0.333  0.000
            _______|____________________________
                   |  0.250  0.250  0.250  0.250

            >>> ISSP42.absolute_monotonicity_radius()
            2.9999999999745341

        **References**: D.I. Ketcheson,
    """
    r=2.*m
    alpha=np.vstack([np.zeros(m),np.eye(m)])
    beta=alpha/r
    for i in range(m): beta[i,i]=1./r
    name='SSPIRK'+str(m)+'2'
    return RungeKuttaMethod(alpha=alpha,beta=beta,name=name)

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


        **References**: D.I. Ketcheson, "Highly efficient strong stability 
                preserving Runge-Kutta methods with low-storage
                implementations", SISC 2008 [Ketcheson2008]

    """
    n=np.sqrt(m)
    assert n==round(n), "SSPRKm3 methods must have m=n^2"
    assert m>=4, "SSPRKm3 methods must have m>=4"
    r=float(m-n)
    alpha=np.vstack([np.zeros(m),np.eye(m)])
    alpha[n*(n+1)/2,n*(n+1)/2-1]=(n-1.)/(2*n-1.)
    beta=alpha/r
    alpha[n*(n+1)/2,(n-1)*(n-2)/2]=n/(2*n-1.)
    return RungeKuttaMethod(alpha=alpha,beta=beta)


if __name__== "__main__":
    RK=loadRKM()




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


