"""
**Examples**::

    >>> from nodepy.runge_kutta_method import *

* Load a method::

    >>> ssp104=loadRKM('SSP104')

* Check its order of accuracy::

    >>> ssp104.order()
    4

* Find its radius of absolute monotonicity::

    >>> ssp104.absolute_monotonicity_radius()
    5.999999999949068

* Load a dictionary with many methods::

    >>> RK=loadRKM()
    >>> RK.keys()
    ['BE', 'SSP75', 'Lambert65', 'Fehlberg45', 'FE', 'SSP33', 'MTE22', 'SSP95', 'RK44', 'SSP22star', 'RadauIIA3', 'RadauIIA2', 'BS5', 'Heun33', 'SSP22', 'DP5', 'LobattoIIIC4', 'NSSP33', 'NSSP32', 'SSP85', 'BuRK65', 'DP8', 'SSP104', 'LobattoIIIA2', 'GL2', 'GL3', 'LobattoIIIC3', 'LobattoIIIC2', 'Mid22']

    >>> RK['Mid22']
    Midpoint Runge-Kutta
    <BLANKLINE>
     0   |
     1/2 |  1/2
    _____|__________
         |    0    1

**References**:  
    #. [butcher2003]_
    #. [hairer1993]_
"""
from __future__ import division
from general_linear_method import GeneralLinearMethod
import numpy as np
import snp
import matplotlib.pyplot as pl
import sympy


#=====================================================
class RungeKuttaMethod(GeneralLinearMethod):
#=====================================================
    r""" 
        General class for implicit and explicit Runge-Kutta Methods.
        The method is defined by its Butcher array (`A,b,c`).
        It is assumed everywhere that  `c_i=\sum_j A_{ij}`.
        
        A Runge-Kutta Method is initialized by providing either:
            #. Butcher arrays `A` and `b` with valid and consistent 
               dimensions; or
            #. Shu-Osher arrays `\alpha` and `\beta` with valid and
               consistent dimensions 

        but not both.

        The Butcher arrays are used as the primary representation of
        the method.  If Shu-Osher arrays are provided instead, the
        Butcher arrays are computed by :meth:`shu_osher_to_butcher`.
    """

    #============================================================
    # Private functions
    #============================================================

    def __init__(self,A=None,b=None,alpha=None,beta=None,
            name='Runge-Kutta Method',shortname='RKM',description='',mode='exact'):
        r"""
            Initialize a Runge-Kutta method.  For explicit methods,
            the class ExplicitRungeKuttaMethod should be used instead.

            TODO: make A a property and update c when it is changed
        """
        A,b,alpha,beta=snp.normalize(A,b,alpha,beta)
        # Here there is a danger that one could change A
        # and c would never be updated
        # A,b, and c should be properties
        butcher   = (A is not None) and (b is not None)
        shu_osher = (alpha is not None) and (beta is not None)
        if not (butcher + shu_osher == 1):
            raise Exception("""To initialize a Runge-Kutta method,
                you must provide either Butcher arrays or Shu-Osher arrays,
                but not both.""")
        if butcher:
            # Check that number of stages is consistent
            m=np.size(A,0) # Number of stages
            if m>1:
                if not np.all([np.size(A,1),np.size(b)]==[m,m]):
                    raise Exception(
                     'Inconsistent dimensions of Butcher arrays')
            else:
                if not np.size(b)==1:
                    raise Exception(
                     'Inconsistent dimensions of Butcher arrays')
        elif shu_osher:
            self.alpha=alpha
            self.beta=beta
            A,b=shu_osher_to_butcher(alpha,beta)
        # Set Butcher arrays
        if len(np.shape(A))==2: self.A=A
        else: self.A=np.array([A]) #Fix for 1-stage methods

        self.b=b
        self.c=np.sum(self.A,1) # Assume stage order >= 1

        self.name=name
        self.shortname=shortname
        self.info=description

        if isinstance(self,ExplicitRungeKuttaMethod):
            self.mtype = 'Explicit Runge-Kutta method'
        elif not (self.A.T - np.triu(self.A.T)).any():
            self.mtype = 'Diagonally implicit Runge-Kutta method'
        else:
            self.mtype = 'Implicit Runge-Kutta method'

        if not isinstance(self,ExplicitRungeKuttaMethod):
            if not np.triu(self.A).any():
                print """Warning: this method appears to be explicit, but is
                       being initialized as a RungeKuttaMethod rather than
                       as an ExplicitRungeKuttaMethod."""

    def __num__(self):
        """
        Returns a copy of the method but with floating-point coefficients.
        This is useful whenever we need to operate numerically without
        worrying about the representation of the method.
        """
        import copy
        numself = copy.deepcopy(self)
        if self.A.dtype==object:
            numself.A=np.array(self.A,dtype=np.float64)
            numself.b=np.array(self.b,dtype=np.float64)
            numself.c=np.array(self.c,dtype=np.float64)
        return numself

    def latex(self):
        """A laTeX representation of the Butcher arrays."""
        from sympy.printing import latex
        s= r'\begin{align}'
        s+='\n'
        s+=r'  \begin{array}{c|'
        s+='c'*len(self)
        s+='}\n'
        for i in range(len(self)):
            s+=latex(self.c[i])
            for j in range(len(self)):
                s+=' & '+latex(self.A[i,j])
            s+=r'\\'
            s+='\n'
        s+=r'\hline'
        s+='\n'
        for j in range(len(self)):
            s+=' & '+latex(self.b[j])
        s+='\n'
        s+=r'\end{array}'
        s+=r'\end{align}'
        s=s.replace('- -','')
        return s

    def __repr__(self): 
        """
        Pretty-prints the Butcher array in the form:
          |
        c | A
        ______
          | b
        """
        from utils import shortstring

        A = [shortstring(ai) for ai in self.A.reshape(-1)]
        b = [shortstring(bi) for bi in self.b]
        c = [shortstring(ci) for ci in self.c]
        bhatlenmax = 0
        #if hasattr(self,'bhat'):
        #    bhat = [shortstring(bi) for bi in self.bhat]
        #    bhatlenmax = max([len(bi) for bi in bhat])
        lenmax, colmax = _get_column_widths([A,b,c])
        alenmax, blenmax, clenmax = lenmax

        s=self.name+'\n'+self.info+'\n'
        for i in range(len(self)):
            s+=c[i]+' '*(clenmax-len(c[i])+1)+'| '
            for j in range(len(self)):
                ss=shortstring(self.A[i,j])
                s+=ss.ljust(colmax+1)
            s=s.rstrip()+'\n'
        s+='_'*(clenmax+1)+'|'+('_'*(colmax+1)*len(self))+'\n'
        s+= ' '*(clenmax+1)+'|'
        for j in range(len(self)):
            s+=' '*(colmax-len(b[j])+1)+b[j]
        return s

    def __eq__(self,rkm):
        """
            Methods considered equal if their Butcher arrays are

            TODO: Instead check whether methods have the same elementary weights
                  up to some order.
        """
        K1=np.vstack([self.A,self.b])
        K2=np.vstack([rkm.A,rkm.b])
        if K1.shape!=K2.shape:
            return False
        else:
            return (np.vstack([self.A,self.b])==np.vstack([rkm.A,rkm.b])).all()

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
        """
        return compose(self,RK2,1,1)

    #============================================================
    # Reducibility
    #============================================================

    def _dj_reducible_stages(self,tol=1.e-13):
        """ Determine whether the method is DJ-reducible.

            A method is DJ-reducible if it contains any stage that
            does not influence the output.

            Returns a list of unnecessary stages.  If the method is
            DJ-irreducible, returns an empty list.
        """
        from copy import copy
        b=self.b; A=self.A
        Nset = [j for j in range(len(b)) if abs(b[j])<tol]
        while len(Nset)>0:      #Try successively smaller sets N
            Nsetold=copy(Nset)
            for j in Nset:      #Test whether stage j matters
                remove_j=False
                for i in range(len(self)):
                    if i not in Nset and abs(A[i,j])>tol: #Stage j matters
                        remove_j=True
                        continue       
                    if remove_j: continue
                if remove_j: 
                    Nset.remove(j)
                    continue
            if Nset==Nsetold: return Nset
        if hasattr(self,'embedded_method'):
            Nset2 = self.embedded_method._dj_reducible_stages(tol)
            Nset = [x for x in Nset if x in Nset2]
        return Nset

    def dj_reduce(self,tol=1.e-13):
        """Remove all DJ-reducible stages.

            A method is DJ-reducible if it contains any stage that
            does not influence the output.

            **Examples**::
            
                Construct a reducible method:
                >>> from nodepy import rk
                >>> A=np.array([[0,0],[1,0]])
                >>> b=np.array([1,0])
                >>> rkm = rk.ExplicitRungeKuttaMethod(A,b)

                Check that it is reducible:
                >>> rkm._dj_reducible_stages()
                [1]

                Reduce it by removing stage 1 (the second stage):
                >>> rkm.dj_reduce()
                Runge-Kutta Method
                <BLANKLINE>                
                 0 |
                ___|___
                   |  1
        """
        reducible=True
        while(reducible):
            djs = self._dj_reducible_stages(tol=tol)
            if len(djs)>0: 
                reducible = True
                self._remove_stage(djs[0])
            else: reducible = False
        return self


    def _hs_reducible_stages(self,tol=1.e-13):
        """ 
            Determine whether the method is HS-reducible.
            A Runge-Kutta method is HS-reducible if two
            rows of A are equal.

            If the method is HS-reducible, returns True and a
            pair of equal stages.  If not, returns False and
            the minimum pairwise difference (in the maximum norm) 
            between rows of A.
        """
        m=len(self)
        mindiff=10.
        for i in range(m):
            for j in range(i+1,m):
                dif = np.max(np.abs(self.A[i,:]-self.A[j,:]))
                if dif<tol: return True,[i,j]
                mindiff=min(mindiff,dif)
        return False, mindiff

    def _remove_stage(self,stage):
        """ Eliminate a stage of a Runge-Kutta method.
            Typically used to reduce reducible methods.

            Note that stages in the NumPy arrays are indexed from zero,
            so to remove stage j use _remove_stage(j-1).
        """
        A=np.delete(np.delete(self.A,stage,1),stage,0)
        b=np.delete(self.b,stage)
        c=np.delete(self.c,stage)
        self.A=A
        self.b=b
        self.c=c
        if hasattr(self,'bhat'):
            bhat=np.delete(self.bhat,stage)
            self.bhat=bhat

    #============================================================
    # Accuracy
    #============================================================

    def error_coefficient(self,tree):
        r"""
        Returns the coefficient in the Runge-Kutta method's error expansion
        multiplying a single elementary differential,
        corresponding to a given tree.

           **Examples**::
            
                Construct an RK method and some rooted trees:
                >>> from nodepy import rk, rt
                >>> rk4 = rk.loadRKM('RK44')
                >>> tree4 = rt.list_trees(4)[0]
                >>> tree5 = rt.list_trees(5)[0]

                The method has order 4, so this gives zero:
                >>> rk4.error_coefficient(tree4)
                0

                This is non-zero, as the method doesn't
                satisfy fifth-order conditions:
                >>> rk4.error_coefficient(tree5)
                -1/720
        """
        from numpy import dot
        from sympy import Rational, simplify
        code=elementary_weight_str(tree)
        b=self.b
        A=self.A
        c=self.c
        if A.dtype == object:
            exec('coeff=simplify('+code+'-Rational(1,'+str(tree.density())+'))')
        else:
            exec('coeff=('+code+'-1./'+str(tree.density())+')')
        return coeff/tree.symmetry()

    def error_coeffs(self,p):
        r"""
        Returns the coefficients in the Runge-Kutta method's error expansion
        multiplying all elementary differentials of the given order.
        """
        import rooted_trees as rt
        forest=rt.list_trees(p)
        err_coeffs=[]
        for tree in forest:
            err_coeffs.append(self.error_coefficient(tree))
        return err_coeffs

    def error_metrics(self):
        r"""
        Returns several measures of the accuracy of the Runge-Kutta method.
        In order, they are:

            * `A^{q+1}`: 2-norm of the vector of leading order error coefficients
            * `A^{q+1}_{max}`: Max-norm of the vector of leading order error coefficients
            * `A^{q+2}` : 2-norm of the vector of next order error coefficients
            * `A^{q+2}_{max}`: Max-norm of the vector of next order error coefficients
            * `D`: The largest (in magnitude) coefficient in the Butcher array

            **Examples**::

                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> rk4.error_metrics()
                (0.01450458234319821, 1/120, 0.016035314699606992, 1/144, 1)
                
            Reference: [kennedy2000]_
        """
        q=self.order()
        tau_1=self.error_coeffs(q+1)
        tau_2=self.error_coeffs(q+2)

        A_qp1=np.sqrt(float(np.sum(np.array(tau_1)**2)))
        A_qp1_max=max([abs(tau) for tau in tau_1])
        A_qp2=np.sqrt(float(np.sum(np.array(tau_2)**2)))
        A_qp2_max=max([abs(tau) for tau in tau_2])

        D=max(np.max(np.abs(self.A)),np.max(np.abs(self.b)),np.max(np.abs(self.c)))
        return A_qp1, A_qp1_max, A_qp2, A_qp2_max, D


    def principal_error_norm(self,tol=1.e-13):
        r""" Returns the 2-norm of the vector of leading order error coefficients."""
        import rooted_trees as rt
        p=self.order(tol)
        forest=rt.list_trees(p+1)
        errs=[]
        for tree in forest:
            errs.append(self.error_coefficient(tree))
        return np.sqrt(float(np.sum(np.array(errs)**2)))

    def order(self,tol=1.e-14):
        """ The order of a Runge-Kutta method.

            **Examples**::

                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> rk4.order()
                4

            TODO: Fix this for sympy coefficients
        """
        from sympy import simplify
        p=0
        while True:
            z=self.order_conditions(p+1)
            z = snp.array([simplify(zz) for zz in z])
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

            TODO: Implement the different approaches through optional keyword
        """
        from sympy import factorial,Rational
        A,b,c=self.A,self.b,self.c
        C=snp.diag(c)
        code=runge_kutta_order_conditions(p)
        z=snp.zeros(len(code)+1)
        tau=snp.zeros([p,len(self)])
        for j in range(1,p):
            tau[j,:]=(c**j/j-np.dot(A,c**(j-1)))/factorial(j-1)
        for i in range(len(code)):
            exec('z[i]='+code[i])
        z[-1]=np.dot(b,c**(p-1))-Rational(1,p)
        return z

    def stage_order(self,tol=1.e-14):
        r""" 
            The stage order of a Runge-Kutta method is the minimum, 
            over all stages, of the
            order of accuracy of that stage.  It can be shown to be
            equal to the largest integer k such that the simplifying
            assumptions `B(\\xi)` and `C(\\xi)` are satisfied for
            `1 \\le \\xi \\le k`.

            **Examples**::

                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> rk4.stage_order()
                1
                >>> gl2 = rk.loadRKM('GL2')
                >>> gl2.stage_order()
                2

            **References**:
                #. Dekker and Verwer
                #. [butcher2003]_
        """
        from sympy import simplify
        simp_array = np.vectorize(sympy.simplify)
        k,B,C=0,0.,0.
        while np.all(abs(B)<tol) and np.all(abs(C)<tol):
            k=k+1
            B=simplify(np.dot(self.b,self.c**(k-1)))-1./k
            C=simp_array(np.dot(self.A,self.c**(k-1))-self.c**k/k)
        return k-1


    #============================================================
    # Classical Stability
    #============================================================
    def stability_function(self):
        r""" 
            The stability function of a Runge-Kutta method is
            `\\phi(z)=p(z)/q(z)`, where

            $$p(z)=\\det(I - z A + z e b^T)$$

            $$q(z)=\\det(I - z A)$$

            This function constructs the numerator and denominator of the 
            stability function of a Runge-Kutta method.

            **Output**:
                - p -- Numpy poly representing the numerator
                - q -- Numpy poly representing the denominator

            **Examples**::

                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> p,q = rk4.stability_function()
                >>> print p
                         4          3       2
                0.04167 x + 0.1667 x + 0.5 x + 1 x + 1

        """
        import sympy

        if self.A.dtype==object:
            Asym=sympy.matrices.Matrix(self.A)
            bsym=sympy.matrices.Matrix(np.tile(self.b,(len(self),1)))
            xsym=Asym-bsym
            x=sympy.var('x')
            p1=xsym.charpoly(x).coeffs()
            q1=Asym.charpoly(x).coeffs()
        else:
            p1=np.poly(self.A-np.tile(self.b,(len(self),1)))
            q1=np.poly(self.A)
        p=np.poly1d(p1[::-1])    # Numerator
        q=np.poly1d(q1[::-1])    # Denominator
        return p,q

    def plot_stability_function(self,bounds=[-20,1]):
        p,q=self.stability_function()
        xx=np.arange(bounds[0], bounds[1], 0.01)
        yy=p(xx)/q(xx)
        pl.plot(xx,yy)
        pl.draw()  


    def plot_stability_region(self,N=200,color='r',filled=True,
                              plotroots=False,alpha=1.,scalefac=1.,to_file=False,
                              longtitle=True,fignum=None):
        r""" 
            The region of absolute stability
            of a Runge-Kutta method, is the set

            `\{ z \in C : |\phi (z)|\le 1 \}`

            where `\phi(z)` is the stability function of the method.

            **Input**: (all optional)
                - N       -- Number of gridpoints to use in each direction
                - bounds  -- limits of plotting region
                - color   -- color to use for this plot
                - filled  -- if true, stability region is filled in (solid); otherwise it is outlined
        """
        import stability_function 
        import matplotlib.pyplot as plt

        p,q=self.__num__().stability_function()

        fig = stability_function.plot_stability_region(p,q,N,color,filled,plotroots,
                alpha,scalefac)

        ax = fig.get_axes()
        if longtitle:
            plt.setp(ax,title='Absolute Stability Region for '+self.name)
        else:
            plt.setp(ax,title='Stability region')
        if to_file:
            plt.savefig(to_file, transparent=True, bbox_inches='tight', pad_inches=0.3)
        else:
            plt.draw()
        return fig

    def plot_order_star(self,N=200,bounds=[-5,5,-5,5],
                    color='r',filled=True,plotaxes=True):
        r""" The order star of a Runge-Kutta method is the set
            
            $$ \\{ z \\in C : | \\phi(z)/\\exp(z) | \\le 1 \\} $$

            where `\phi(z)` is the stability function of the method.

            **Input**: (all optional)
                - N       -- Number of gridpoints to use in each direction
                - bounds  -- limits of plotting region
                - color   -- color to use for this plot
                - filled  -- if true, order star is filled in (solid); otherwise it is outlined
        """
        p,q=self.__num__().stability_function()
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
        if plotaxes:
            pl.hold(True)
            pl.plot([0,0],[bounds[2],bounds[3]],'--k')
            pl.plot([bounds[0],bounds[1]],[0,0],'--k')
        pl.axis('Image')
        pl.hold(False)
        
    #============================================================
    # Nonlinear Stability
    #============================================================
    def circle_contractivity_radius(self,acc=1.e-13,rmax=1000):
        r""" 
            Returns the radius of circle contractivity
            of a Runge-Kutta method.
        """
        from utils import bisect

        tol=1.e-14
        r=bisect(0,rmax,acc,tol,self.__num__().is_circle_contractive)
        return r

    def absolute_monotonicity_radius(self,acc=1.e-10,rmax=200,
                    tol=3.e-16):
        r""" 
            Returns the radius of absolute monotonicity
            (also referred to as the radius of contractivity or
            the strong stability preserving coefficient 
            of a Runge-Kutta method.
        """
        from utils import bisect

        r=bisect(0,rmax,acc,tol,self.is_absolutely_monotonic)
        if r>=rmax-acc: return np.inf
        else: return r

    def linear_monotonicity_radius(self,acc=1.e-10,tol=1.e-15,tol2=1e-8):
        r"""
            Computes Horvath's monotonicity radius of the stability
            function.

            TODO: clean this up.
        """

        p,q=self.stability_function()
        for i in range(len(p)+1):
            if abs(p[i])<=tol2: p[i]=0.0
        for i in range(len(q)+1):
            if abs(q[i])<=tol2: q[i]=0.0
        #First check extreme cases
        if p.order>q.order: return 0
        phi = lambda z: p(z)/q(z)
        #Get the negative real zeroes of the derivative of p/q:
        phip=p.deriv()*q-q.deriv()*p
        zeroes=[z for z in phip.r if np.isreal(z) and z<0]      
        #Find the extremum of phi on (-inf,0):
        xmax=-10000
        if phip(0)<0: return 0
        if len(zeroes)>0:
            for i in range(len(zeroes)):
                if p(zeroes[i])/q(zeroes[i])<p(xmax)/q(xmax) and zeroes[i]>xmax:  xmax=zeroes[i]
            zmax=max(abs(phi(zeroes)))
            rlo=max(zeroes)
            if p.order==q.order: 
                zmax=max(zmax, abs(p[len(p)]/q[len(q)]))
        else:
            if p.order<q.order: return -np.inf
            if p.order==q.order: 
                zmax=abs(p[len(p)]/q[len(q)])
                if p[len(p)]/q[len(q)]>=-tol: return -np.inf
                rlo=-10000
        s=p-zmax*q 
        zeroes2=[z for z in s.r if np.isreal(z) and z<0 and z>=xmax]
        if len(zeroes2)>0:
            r=max(zeroes2)
        else: r=0
        return float(np.real(r))


    def is_circle_contractive(self,r,tol):
        r""" Returns 1 if the Runge-Kutta method has radius of circle
            contractivity at least `r`.
            
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

    def denominator_absolute_monotonicity_radius(self,acc=1.e-10,rmax=50,
                                            tol=3.e-16):
        """ 
            Returns the radius of absolute monotonicity
            of the denominator of the stability function of a Runge-Kutta method.
        """
        from utils import bisect
        p,q=self.stability_function()
        r=bisect(0,rmax,acc,tol,is_absolutely_monotonic_poly,q)
        return r

    def numerator_absolute_monotonicity_radius(self,acc=1.e-10,rmax=50,
                                            tol=3.e-16):
        """ 
            Returns the radius of absolute monotonicity
            of the numerator of the stability function of a Runge-Kutta method.
        """
        from utils import bisect
        p,q=self.stability_function()
        r=bisect(0,rmax,acc,tol,is_absolutely_monotonic_poly,p)
        return r


    def is_absolutely_monotonic(self,r,tol):
        r""" Returns 1 if the Runge-Kutta method is absolutely monotonic
            at `z=-r`.

            The method is absolutely monotonic if `(I+rA)^{-1}` exists
            and
            $$K(I+rA)^{-1} \\ge 0$$
            $$(I+rA)^{-1} e_m \\ge 0$$

            where `e_m` is the m-by-1 vector of ones and
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

    #============================================================
    # Representations
    #============================================================
    def optimal_shu_osher_form(self,r=None):
        r"""
            Gives a Shu-Osher form in which the SSP coefficient is
            evident (i.e., in which `\\alpha_{ij},\\beta_{ij} \\ge 0` and
            `\\alpha_{ij}/\\beta_{ij}=c` for every `\\beta_{ij}\\ne 0`).

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
        m=len(self)
        if r is None: r=self.absolute_monotonicity_radius()
        K=np.vstack([self.A,self.b])
        K=np.hstack([K,np.zeros([m+1,1])])
        X=snp.eye(m+1)+r*K
        beta=snp.solve(X,K)
        beta=beta[:,:-1]
        alpha=r*beta
        for i in range(1,len(self)+1):
            alpha[i,0]=1.-np.sum(alpha[i,1:])
        return alpha, beta

    def canonical_shu_osher_form(self,r):
        r""" d,P where P is the matrix `P=r(I+rK)^{-1}K`
             and d is the vector `d=(I+rK)^{-1}e=(I-P)e`
        """
        s=len(self)
        K=np.vstack([self.A,self.b])
        K=np.hstack([K,np.zeros([s+1,1])])
        I=snp.eye(s+1)
        P=r*snp.solve(I+r*K,K)
        d=(I-P).sum(1)
        return d,P

    #==========================================
    #The next three functions are experimental!
    #==========================================

    def ssplit(self,r,P_signs=None,delta=None):
        """Sympy exact version of split()
        
        If P_signs is passed, use that as the sign pattern of the P matrix.
        This is useful if r is symbolic (since then in general the signs of
        elemnts of P are unknown).
        """
        import numpy as np
        s=len(self)
        I=snp.eye(s+1)
        d,P=self.canonical_shu_osher_form(r)

        # Split P into positive and negative parts
        if P_signs is None:
            P_signs = (P>0).astype(int)

        if delta is None:
            delta = np.zeros(P.shape)

        P_plus=P*P_signs + delta
        P_minus=-P*(1-P_signs) + delta

        # Form new coefficients
        M=I+2*P_minus
        alpha=snp.solve(M,P_plus)
        gamma=snp.solve(M,d)
        alphatilde=snp.solve(M,P_minus)

        if self.is_explicit():
            # Assuming gamma is positive, we can redistribute it
            alpha[1:,0]+=gamma[1:]/2
            alphatilde[1:,0]+=gamma[1:]/2
            gamma[1:]=0

        return gamma, alpha, alphatilde

    def split(self,r,tol=1.e-15):
        import numpy as np
        s=len(self)
        I=np.eye(s+1)
        d,P=self.canonical_shu_osher_form(r)

        # Split P into positive and negative parts
        P_plus=P*(P>0).astype(int)
        P_minus=-P*(P<0).astype(int)

        # Form new coefficients
        M=np.linalg.inv(I+2*P_minus)
        alpha=np.dot(M,P_plus)
        gamma=np.dot(M,d)
        alphatilde=np.dot(M,P_minus)

        if self.is_explicit():
            # Assuming gamma is positive, we can redistribute it
            alpha[1:,0]+=gamma[1:]/2.
            alphatilde[1:,0]+=gamma[1:]/2.
            gamma[1:]=0.

        return gamma, alpha, alphatilde


    def is_splittable(self,r,tol=1.e-15):
        d,alpha,alphatilde=self.split(r,tol=tol)
        if alpha.min()>=-tol and d.min()>=-tol and alphatilde.min()>=-tol: return True
        else: return False

    def optimal_perturbed_splitting(self,acc=1.e-12,rmax=50.01,tol=1.e-13):
        r"""
            Return the optimal (possibly?) downwind splitting of the method
            along with the optimal downwind SSP coefficient.
        """
        from utils import bisect

        r=bisect(0,rmax,acc,tol,self.is_splittable)
        d,alpha,alphatilde=self.split(r,tol=tol)
        return r,d,alpha,alphatilde

    #============================================================
    # Miscellaneous
    #============================================================
    def propagation_matrix(self,L,dt):
        """
            Returns the solution propagation matrix for the linear 
            autonomous system with RHS equal to the matrix L, i.e. 
            it returns the matrix G such that when the Runge-Kutta
            method is applied to the system 
            `u'(t)=Lu`
            with stepsize dt, the numerical solution is given by
            `u^{n+1} = G u^n`.

            **Input**:
                - self -- a Runge-Kutta method
                - L    -- the RHS of the ODE system
                - dt   -- the timestep

            The formula for `G` is (if `L` is a scalar):
            `G = 1 + b^T L (I-A L)^{-1} e`

            where `A` and `b` are the Butcher arrays and `e` is the vector
            of ones.  If `L` is a matrix, all quantities above are 
            replaced by their Kronecker product with the identity
            matrix of size `m`, where `m` is the number of stages of
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


    def is_explicit(self):
        return False

    def is_FSAL(self):
        """True if method is "First Same As Last"."""
        if np.all(self.A[-1,:]==self.b): return True
        else: return False

#=====================================================
class ExplicitRungeKuttaMethod(RungeKuttaMethod):
#=====================================================
    r"""
        Class for explicit Runge-Kutta methods.  Mostly identical
        to RungeKuttaMethod, but also includes time-stepping and
        a few other functions.
    """
    def __repr__(self): 
        """
        Pretty-prints the Butcher array in the form:
          |
        c | A
        ______
          | b
        """
        from utils import shortstring

        c = [shortstring(ci) for ci in self.c]
        clenmax = max([len(ci) for ci in c])
        A = [shortstring(ai) for ai in self.A.reshape(-1)]
        alenmax = max([len(ai) for ai in A])
        b = [shortstring(bi) for bi in self.b]
        blenmax = max([len(bi) for bi in b])
        colmax=max(alenmax,blenmax)

        s=self.name+'\n'+self.info+'\n'

        s=self.name+'\n'+self.info+'\n'
        for i in range(len(self)):
            s+=c[i]+' '*(clenmax-len(c[i])+1)+'| '
            for j in range(i):
                ss=shortstring(self.A[i,j])
                s+=ss.ljust(colmax+1)
            s=s.rstrip()+'\n'
        s+='_'*(clenmax+1)+'|'+('_'*(colmax+1)*len(self))+'\n'
        s+= ' '*(clenmax+1)+'|'
        for j in range(len(self)):
            s+=' '*(colmax-len(b[j])+1)+b[j]
        return s

    def __step__(self,f,t,u,dt,x=None,errest=False):
        """
            Take a time step on the ODE u'=f(t,u).

            **Input**:
                - f  -- function being integrated
                - t  -- array of previous solution times
                - u  -- array of previous solution steps (u[i] is the solution at time t[i])
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
        unew=u[-1]+dt*sum([self.b[j]*fy[j] for j in range(m)])
        return unew

    def imaginary_stability_interval(self,tol=1.e-7,zmax=100.,eps=1.e-6):
        r"""
            Length of imaginary axis half-interval contained in the
            method's region of absolute stability.

            **Examples**::

                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> rk4.imaginary_stability_interval()
                2.8284274972975254
        """
        p,q=self.__num__().stability_function()
        zhi=zmax
        zlo=0.
        #Use bisection to get an upper bound:
        while (zhi-zlo)>tol:
            z=0.5*(zhi+zlo)
            mag=abs(p(1j*z))
            if (mag-1.)>eps: zhi=z
            else: zlo=z
                
        #Now check more carefully:
        vals = np.array([abs(p(1j*zz)) for zz in np.linspace(0.,z,z/0.01)])
        unstable_z=np.where(vals>1.+eps)[0]
        if len(unstable_z)==0: return z
        else: 
            zz = np.linspace(0.,z,z/0.01)
            return zz[min(unstable_z)]

    def real_stability_interval(self,tol=1.e-7,zmax=100.,eps=1.e-6):
        r"""
            Length of negative real axis interval contained in the
            method's region of absolute stability.
    
            **Examples**::

                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> rk4.real_stability_interval()
                2.785294223576784
        """

        p,q=self.__num__().stability_function()
        zhi=zmax
        zlo=0.
        #Use bisection to get an upper bound:
        while (zhi-zlo)>tol:
            z=0.5*(zhi+zlo)
            mag=abs(p(-z))
            if (mag-1.)>eps: zhi=z
            else: zlo=z
                
        #Now check more carefully:
        vals = np.array([p(-zz) for zz in np.linspace(0.,z,z/0.01)])
        unstable_z=np.where(vals>1.+eps)[0]
        if len(unstable_z)==0: return z
        else: 
            zz = np.linspace(0.,z,z/0.01)
            return zz[min(unstable_z)]

    def linear_absolute_monotonicity_radius(self,acc=1.e-10,rmax=50,
                                            tol=3.e-16):
        """ 
            Returns the radius of absolute monotonicity
            of the stability function of a Runge-Kutta method.

            TODO: implement this functionality for implicit methods.
        """
        from utils import bisect
        p,q=self.stability_function()
        if q.order!=0 or q[0]!=1:
            raise NotImplementedError('Not yet implemented for rational functions')
        else:
            r=bisect(0,rmax,acc,tol,is_absolutely_monotonic_poly,p)
        return r

    def is_explicit(self):
        return True

    def work_per_step(self):
        "Number of function evaluations required for one step."
        if self.is_FSAL(): return len(self)-1
        else: return len(self)

    def num_seq_dep_stages(self):
        r"""Number of sequentially dependent stages.

        of sequential function evaluations that must be made.

            **Examples**::

                Extrapolation methods are parallelizable:
                >>> from nodepy import rk
                >>> ex4 = rk.extrap(4)
                >>> len(ex4)
                7
                >>> ex4.num_seq_dep_stages()
                4
                
                So are deferred correction methods:
                >>> dc4 = rk.DC(4)
                >>> len(dc4)
                20
                >>> dc4.num_seq_dep_stages()
                8

                Unless `\theta` is non-zero:
                >>> rk.DC(4,theta=1).num_seq_dep_stages()
                20
        """
        n_s = [0]*len(self)
        for i in range(len(self)):
            for j in range(i):
                if self.A[i,j] != 0:
                    n_s[i] = max(n_s[i], n_s[j]+1)

        n = 0
        for i in range(len(self)):
            if self.b[i] != 0:
                n = max(n, n_s[i]+1)
        return n


#=====================================================
#End of ExplicitRungeKuttaMethod class
#=====================================================


#=====================================================
class ExplicitRungeKuttaPair(ExplicitRungeKuttaMethod):
#=====================================================
    r"""

        Class for embedded Runge-Kutta pairs.  These consist of
        two methods with identical coefficients `a_{ij}`
        but different coefficients `b_j` such that the methods
        have different orders of accuracy.  Typically the
        higher order accurate method is used to advance
        the solution, while the lower order method is
        used to obtain an error estimate.

        An embedded Runge-Kutta Pair takes the form:

        \\begin{align*}
        y_i = & u^{n} + \\Delta t \\sum_{j=1}^{s} + a_{ij} f(y_j)) & (1\\le j \\le s) \\\\
        u^{n+1} = & u^{n} + \\Delta t \\sum_{j=1}^{s} b_j f(y_j) \\\\
        \\hat{u}^{n+1} = & u^{n} + \\Delta t \\sum_{j=1}^{s} \\hat{b}_j f(y_j).
        \\end{align*}

        That is, both methods use the same intermediate stages `y_i`, but different
        weights.  Typically the weights `\\hat{b}_j` are chosen so that `\\hat{u}^{n+1}`
        is accurate of order one less than the order of `u^{n+1}`.  Then their
        difference can be used as an error estimate.

        In NodePy, if *rkp* is a Runge-Kutta pair, the principal (usually
        higher-order) method is the one used if accuracy or stability properties
        are queried.  Properties of the embedded (usually lower-order) method can
        be accessed via *rkp.embedded_method*.

        When solving an IVP with an embedded pair, one can specify a desired
        error tolerance.  The step size will be adjusted automatically
        to achieve approximately this tolerance.
    """
    def __init__(self,A=None,b=None,bhat=None,alpha=None,beta=None,
            name='Runge-Kutta Pair',shortname='RKM',description=''):
        r"""
            In addition to the ordinary Runge-Kutta initialization,
            here the embedded coefficients `\hat{b}_j` are set as well.
        """
        super(ExplicitRungeKuttaPair,self).__init__(A,b,alpha,beta,name,shortname,description)
        if bhat.shape != self.b.shape: 
            raise Exception("Dimensions of embedded method don't agree with those of principal method")
        self.bhat=bhat
        self.mtype = 'Explicit embedded Runge-Kutta pair'

    @property
    def embedded_method(self):
        return ExplicitRungeKuttaMethod(self.A,self.bhat)

    def __num__(self):
        """
        Returns a copy of the method but with floating-point coefficients.
        This is useful whenever we need to operate numerically without
        worrying about the representation of the method.
        """
        numself = super(ExplicitRungeKuttaPair,self).__num__()
        if self.A.dtype==object:
            numself.bhat=np.array(self.bhat,dtype=np.float64)
        return numself

    def __repr__(self): 
        """
        Pretty-prints the Butcher array in the form:
          |
        c | A
        ________
          | b
          | bhat
        """
        s = super(ExplicitRungeKuttaPair,self).__repr__()
        from utils import shortstring

        A = [shortstring(ai) for ai in self.A.reshape(-1)]
        b = [shortstring(bi) for bi in self.b]
        bhat = [shortstring(bi) for bi in self.bhat]
        c = [shortstring(ci) for ci in self.c]
        lenmax, colmax = _get_column_widths([A,b,c])
        alenmax, blenmax, clenmax = lenmax
        s+= '\n'+' '*(clenmax+1)+'|'
        for j in range(len(self)):
            s+=' '*(colmax-len(bhat[j])+1)+bhat[j]
        return s


    def __step__(self,f,t,u,dt,x=None,errest=False):
        """
            Take a time step on the ODE u'=f(t,u).
            Just like the corresponding method for RKMs, but
            for RK pairs also computes an error estimate using
            the embedded method.

            **Input**:
                - f  -- function being integrated
                - t  -- array of previous solution times
                - u  -- array of previous solution steps (u[i] is the solution at time t[i])
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
        unew=u[-1]+dt*sum([self.b[j]*fy[j] for j in range(m)])
        if errest:
            uhat=u[-1]+dt*sum([self.bhat[j]*fy[j] for j in range(m)])
            return unew, np.max(np.abs(unew-uhat))
        else: return unew


    def error_metrics(self):
        r"""Return full set of error metrics
            See [kennedy2000]_ p. 181"""
        A_qp1, A_qp1_max, A_qp2, A_qp2_max, D = super(ExplicitRungeKuttaPair).error_metrics()

        p=self.embedded_method.order(1.e-13)
        tau_pp2=self.error_coeffs(p+2)

        tau_pp1_hat=self.embedded_method.error_coeffs(p+1)
        tau_pp2_hat=self.embedded_method.error_coeffs(p+2)

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


    def is_FSAL(self):
        if np.all(self.A[-1,:]==self.b): return True
        elif np.all(self.A[-1,:]==self.bhat): return True
        else: return False

    def plot_stability_region(self,N=200,color='r',filled=True,
                              plotroots=False,alpha=1.,scalefac=1.,to_file=False,
                              longtitle=True):
        import stability_function 
        import matplotlib.pyplot as plt

        p,q=self.__num__().stability_function()

        fig = stability_function.plot_stability_region(p,q,N,color,filled,plotroots,
                alpha,scalefac)

        plt.hold(True)
        p,q = self.embedded_method.__num__().stability_function()
        stability_function.plot_stability_region(p,q,N,color='k',filled=False,
                plotroots=plotroots,alpha=alpha,scalefac=scalefac,fignum=fig.number)

        ax = fig.get_axes()
        if longtitle:
            plt.setp(ax,title='Absolute Stability Region for '+self.name)
        else:
            plt.setp(ax,title='Stability region')
        if to_file:
            plt.savefig(to_file, transparent=True, bbox_inches='tight', pad_inches=0.3)
        else:
            plt.draw()
        return fig


#=====================================================
#End of ExplicitRungeKuttaPair class
#=====================================================

#=====================================================
#Functions for generating order conditions
#=====================================================
def elementary_weight(tree):
    """
        Constructs Butcher's elementary weights 
        for a Runge-Kutta method

        Currently doesn't work right; note that two of the 5th-order
        weights appear identical.  The _str version below works
        correctly and produces NumPy code.  But it would be nice to 
        have this version working so that we could symbolically 
        simplify the expressions.

        In order to do things correctly, we need a symbolic
        system that includes support for either:

            * Two different types of multiplication; or
            * Full tensor expressions

        The latter is now available in Sympy, but I haven't
        tried to use it here.

        **References**:
            [butcher2003]_
    """
    #raise Exception('This function does not work correctly; use the _str version')
    import rooted_trees as rt
    from sympy import symbols
    b=symbols('b',commutative=False)
    ew=b*tree.Gprod(RKeta,rt.Dmap)
    return ew

def elementary_weight_str(tree,style='python'):
    """
        Constructs Butcher's elementary weights for a Runge-Kutta method
        as strings suitable for numpy execution.

        **Examples**:

            >>> from nodepy import rk, rt
            >>> tree = rt.list_trees(5)[0]
            >>> rk.elementary_weight_str(tree)
            'dot(b,dot(A,c**3))'

            >>> rk.elementary_weight_str(tree,style='matlab')
            "b'*((A*c.^3))"
    """
    from strmanip import collect_powers, mysimp
    from rooted_trees import Dmap_str
    ewstr='dot(b,'+tree.Gprod_str(RKeta_str,Dmap_str)+')'
    ewstr=ewstr.replace('1*','')
    ewstr=collect_powers(ewstr,'c')
    ewstr=mysimp(ewstr)
    if style=='matlab': ewstr=python_to_matlab(ewstr)
    return ewstr

def RKeta(tree):
    from sympy.physics.quantum import TensorProduct
    #raise Exception('This function does not work correctly; use the _str version')
    from rooted_trees import Dprod
    from sympy import symbols
    if tree=='':  return symbols('e',commutative=False)
    if tree=='T': return symbols('c',commutative=False)
    return TensorProduct(symbols('A',commutative=False),Dprod(tree,RKeta))

def RKeta_str(tree):
    """
    Computes eta(t) for Runge-Kutta methods
    """
    from rooted_trees import Dprod_str
    if tree=='':  return 'e'
    if tree=='T': return 'c'
    return 'dot(A,'+Dprod_str(tree,RKeta_str)+')'


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
        raise Exception(
             'Inconsistent dimensions of Shu-Osher arrays')
    X=snp.eye(m)-alpha[0:m,:]
    A=snp.solve(X,beta[0:m,:])
    b=beta[m,:]+np.dot(alpha[m,:],A)
    A = snp.simplify(A)
    b = snp.simplify(b)
    return A,b

def loadRKM(which='All'):
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
	'DP8':        Dormand-Prince 8th-order and 7th-order pair
        'Fehlberg45': 5th-order part of Fehlberg's pair
        'Lambert65':

        Implicit:

        'BE':         Backward Euler
        'GL2':        2-stage Gauss-Legendre
        'GL3':        3-stage Gauss-Legendre

        Also various Lobatto and Radau methods.
    """
    from sympy import sqrt, Rational

    RK={}

    half = Rational(1,2)
    one  = Rational(1,1)
    zero = Rational(0,1)

    #================================================
    A=np.array([one])
    b=np.array([one])
    RK['BE']=RungeKuttaMethod(A,b,name='Implicit Euler',shortname='BE')

    #================================================
    A=np.array([zero])
    b=np.array([one])
    RK['FE']=ExplicitRungeKuttaMethod(A,b,name='Forward Euler',shortname='FE')

    #================================================
    alpha=np.array([[0,0],[1.,0],[0.261583187659478,0.738416812340522]])
    beta=np.array([[0,0],[0.822875655532364,0],[-0.215250437021539,0.607625218510713]])
    RK['SSP22star']=ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name='SSPRK22star',
                description=
                "The underlying method of the optimal 2-stage, 2nd order downwind SSP Runge-Kutta method with one star",shortname='SSP22star')

    #================================================
    A=np.array([[one,-sqrt(5),sqrt(5),-one],[one,3*one,(10-7*sqrt(5))/5,sqrt(5)/5],[one,(10+7*sqrt(5))/5,3*one,-sqrt(5)/5],[one,5*one,5*one,one]])/12
    b=np.array([one,5*one,5*one,one])/12
    RK['LobattoIIIC4']=RungeKuttaMethod(A,b,name='LobattoIIIC4',
                description="The LobattoIIIC method with 4 stages",shortname='LobattoIIIC4')

    #================================================
    A=np.array([[one/6,-one/3,one/6],[one/6,5*one/12,-one/12],[one/6,2*one/3,one/6]])
    b=np.array([one/6,2*one/3,one/6])
    RK['LobattoIIIC3']=RungeKuttaMethod(A,b,name='LobattoIIIC3',
                description="The LobattoIIIC method with 3 stages",shortname='LobattoIIC3')

    #================================================
    A=np.array([[half,-half],[half,half]])
    b=np.array([half,half])
    RK['LobattoIIIC2']=RungeKuttaMethod(A,b,name='LobattoIIIC2',
                description="The LobattoIIIC method with 2 stages",shortname='LobattoIIIC2')

    #================================================
    A=np.array([[0,0],[half,half]])
    b=np.array([half,half])
    RK['LobattoIIIA2']=RungeKuttaMethod(A,b,name='LobattoIIIA2',
                description="The LobattoIIIA method with 2 stages",shortname='LobattoIIIA2')

    #================================================
    A=np.array([[5*one/12,-1*one/12],[3*one/4,1*one/4]])
    b=np.array([3*one/4,1*one/4])
    RK['RadauIIA2']=RungeKuttaMethod(A,b,name='RadauIIA2',
                description="The RadauIIA method with 2 stages",shortname='RadauIIA2')

    #================================================
    A=np.array([[(88-7*sqrt(6))/360,(296-169*sqrt(6))/1800,(-2+3*sqrt(6))/225],
                [(296+169*sqrt(6))/1800,(88+7*sqrt(6))/360,(-2-3*sqrt(6))/225],
                [(16-sqrt(6))/36,(16+sqrt(6))/36,one/9]])
    b=np.array([(16-sqrt(6))/36,(16+sqrt(6))/36,one/9])
    RK['RadauIIA3']=RungeKuttaMethod(A,b,name='RadauIIA3',
                description="The RadauIIA method with 3 stages",shortname='RadauIIA3')

    #================================================
    A=np.array([[0,0],[one,0]])
    b=np.array([half,half])
    RK['SSP22']=ExplicitRungeKuttaMethod(A,b,name='SSPRK22',
                description=
                "The optimal 2-stage, 2nd order SSP Runge-Kutta method",shortname='SSPRK22')

    #================================================
    A=np.array([[0,0,0],[one,0,0],[one/4,one/4,0]])
    b=np.array([one/6,one/6,2*one/3])
    RK['SSP33']=ExplicitRungeKuttaMethod(A,b,name='SSPRK33',
                description=
                "The optimal 3-stage, 3rd order SSP Runge-Kutta method",shortname='SSPRK33')

    #================================================
    A=np.array([[0,0,0],[one/3,0,0],[0.,2*one/3,0]])
    b=np.array([one/4,0,3*one/4])
    RK['Heun33']=ExplicitRungeKuttaMethod(A,b,name='Heun33',
                description= "Heun's 3-stage, 3rd order",shortname='Heun33')

    #================================================
    A=np.array([[0,0,0],[one/3,0,0],[0,one,0]])
    b=np.array([one/2,0,one/2])
    RK['NSSP32']=ExplicitRungeKuttaMethod(A,b,name='NSSPRK32',
                description= "Wang and Spiteri NSSP32",shortname='NSSPRK32')

    #================================================
    A=np.array([[0,0,0],[-4*one/9,0,0],[7*one/6,-one/2,0]])
    b=np.array([one/4,0.,3*one/4])
    RK['NSSP33']=ExplicitRungeKuttaMethod(A,b,name='NSSPRK33',
                description= "Wang and Spiteri NSSP33",shortname='NSSPRK33')

    #================================================
    m=10
    r=6*one
    alpha=snp.diag(snp.ones(m),-1)
    alpha[5,4]=2*one/5
    alpha[m,m-1]=3*one/5
    alpha[m,4]=9*one/25
    alpha=alpha[:,:m]
    beta=alpha/r
    RK['SSP104']=ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,
                    name='SSPRK(10,4)',description=
                    "The optimal ten-stage, fourth order SSP Runge-Kutta method",shortname='SSPRK(10,4)')
    #================================================
    alpha=snp.zeros([7,6])
    beta=snp.zeros([7,6])
    alpha[1,0]=one
    alpha[2,0:2]=[3*one/4,1*one/4]
    alpha[3,0:3]=[3*one/8,1*one/8,1*one/2]
    alpha[4,0:4]=[1*one/4,1*one/8,1*one/8,1*one/2]
    alpha[5,0:5]=[89537*one/2880000,407023*one/2880000,1511*one/12000,87*one/200,4*one/15]
    alpha[6,:]  =[4*one/9,1*one/15,zero,8*one/45,zero,14*one/45]
    beta[1,0]=1*one/2
    beta[2,0:2]=[zero,1*one/8]
    beta[3,0:3]=[-1*one/8,-1*one/16,1*one/2]
    beta[4,0:4]=[-5*one/64,-13*one/64,1*one/8,9*one/16]
    beta[5,0:5]=[2276219*one/40320000,407023*one/672000,1511*one/2800,-261*one/140,8*one/7]
    beta[6,:]  =[zero,-8*one/45,zero,2*one/3,zero,7*one/90]
    RK['Lambert65']=ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,
                        name='Lambert65',description='From Shu-Osher paper',shortname='Lambert65')
    #================================================
    A=np.array([[0,0],[2*one/3,0]])
    b=np.array([1*one/4,3*one/4])
    RK['MTE22']=ExplicitRungeKuttaMethod(A,b,name='Minimal Truncation Error 22',shortname='MTE22')

    #================================================
    A=np.array([[0,0],[1*one/2,0]])
    b=np.array([0,one])
    RK['Mid22']=ExplicitRungeKuttaMethod(A,b,name='Midpoint Runge-Kutta',shortname='Mid22')

    #================================================
    A=snp.array([[0,0,0,0],[half,0,0,0],[0,half,0,0],[0,0,one,0]])
    b=snp.array([one/6,one/3,one/3,one/6])
    description='The original four-stage, fourth-order method of Kutta'
    RK['RK44']=ExplicitRungeKuttaMethod(A,b,name='Classical RK4',shortname='RK44',description=description)

    #================================================
    A=np.array([[0,0,0,0,0,0],[one/4,0,0,0,0,0],[one/8,one/8,0,0,0,0],
         [0,0,half,0,0,0],[3*one/16,-3*one/8,3*one/8,9*one/16,0,0],
         [-3*one/7,8*one/7,6*one/7,-12*one/7,8*one/7,0]])
    b=np.array([7*one/90,0,16*one/45,2*one/15,16*one/45,7*one/90])
    RK['BuRK65']=ExplicitRungeKuttaMethod(A,b,name="Butcher's RK65",shortname='BuRK65')

    #================================================
    A=np.array([[one/4,one/4-sqrt(3)/6],[one/4+sqrt(3)/6,one/4]])
    b=np.array([half,half])
    RK['GL2']=RungeKuttaMethod(A,b,name="Gauss-Legendre RK24",shortname='GL2')

    #================================================
    A=np.array([[5*one/36,(80-24*sqrt(15))/360,(50-12*sqrt(15))/360],
         [(50+15*sqrt(15))/360,2*one/9,(50-15*sqrt(15))/360],
         [(50+12*sqrt(15))/360,(80+24*sqrt(15))/360,5*one/36]])
    b=np.array([5*one/18,4*one/9,5*one/18])
    RK['GL3']=RungeKuttaMethod(A,b,name="Gauss-Legendre RK36",shortname='GL3')
    #================================================
    A=np.array([[0,0,0,0,0,0],[one/4,0,0,0,0,0],[3*one/32,9*one/32,0,0,0,0],
        [1932*one/2197,-7200*one/2197,7296*one/2197,0,0,0],
        [439*one/216,-8,3680*one/513,-845*one/4104,0,zero],
        [-8*one/27,2,-3544*one/2565,1859*one/4104,-11*one/40,zero]])
    b=np.array([16*one/135,zero,6656*one/12825,28561*one/56430,-9*one/50,2*one/55])
    bhat=np.array([25*one/216,0,1408*one/2565,2197*one/4104,-1*one/5,zero])
    RK['Fehlberg45']=ExplicitRungeKuttaPair(A,b,bhat,name='Fehlberg RK5(4)6',shortname='Fehlberg45')
    #================================================
    A=np.array([[0,0,0,0,0,0,0],[one/5,0,0,0,0,0,0],[3*one/40,9*one/40,0,0,0,0,0],
        [44*one/45,-56*one/15,32*one/9,0,0,0,0],
        [19372*one/6561,-25360*one/2187,64448*one/6561,-212*one/729,0,0,0],
        [9017*one/3168,-355*one/33,46732*one/5247,49*one/176,-5103*one/18656,0,0],
        [35*one/384,0*one,500*one/1113,125*one/192,-2187*one/6784,11*one/84,0]])
    b=np.array([35*one/384,0*one,500*one/1113,125*one/192,-2187*one/6784,11*one/84,0])
    bhat=np.array([5179*one/57600,0*one,7571*one/16695,393*one/640,-92097*one/339200,187*one/2100,1*one/40])
    RK['DP5']=ExplicitRungeKuttaPair(A,b,bhat,name='Dormand-Prince RK5(4)7',shortname='DP5')
    #================================================
    A=np.array([[0,0,0,0,0,0,0,0],[one/6,0,0,0,0,0,0,0],[2*one/27,4*one/27,0,0,0,0,0,0],
        [183*one/1372,-162*one/343,1053*one/1372,0,0,0,0,0],
        [68*one/297,-4*one/11,42*one/143,1960*one/3861,0,0,0,0],
        [597*one/22528,81*one/352,63099*one/585728,58653*one/366080,4617*one/20480,0,0,0],
        [174197*one/959244,-30942*one/79937,8152137*one/19744439,666106*one/1039181,-29421*one/29068,482048*one/414219,0,0],
        [587*one/8064,0,4440339*one/15491840,24353*one/124800,387*one/44800,2152*one/5985,7267*one/94080,0]])
    b=A[-1,:]
    bhat=np.array([2479*one/34992,0*one,123*one/416,612941*one/3411720,43*one/1440,2272*one/6561,79937*one/1113912,3293*one/556956])
    RK['BS5']=ExplicitRungeKuttaPair(A,b,bhat,name='Bogacki-Shampine RK5(4)8',shortname='BS5')
     #================================================
    A=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],[1./18,0,0,0,0,0,0,0,0,0,0,0,0],[1./48,1./16,0,0,0,0,0,0,0,0,0,0,0],
        [1./32,0,3./32,0,0,0,0,0,0,0,0,0,0],[5./16,0,-75./64,75./64,0,0,0,0,0,0,0,0,0],[3./80,0,0,3./16,3./20,0,0,0,0,0,0,0,0],[29443841./614563906,0,0,77736538./692538347,-28693883./1125000000,23124283./1800000000,0,0,0,0,0,0,0],[16016141./946692911,0,0,61564180./158732637,22789713./633445777,545815736./2771057229,-180193667./1043307555,0,0,0,0,0,0],[39632708./573591083,0,0,-433636366./683701615,-421739975./2616292301,100302831./723423059,790204164./839813087,800635310./3783071287,0,0,0,0,0],[246121993./1340847787,0,0,-37695042795./15268766246,-309121744./1061227803,-12992083./490766935,6005943493./2108947869,393006217./1396673457,123872331./1001029789,0,0,0,0],[-1028468189./846180014,0,0,8478235783./508512852,1311729495./1432422823,-10304129995./1701304382,-48777925059./3047939560,15336726248./1032824649,-45442868181./3398467696,3065993473./597172653,0,0,0],[185892177./718116043,0,0,-3185094517./667107341,-477755414./1098053517,-703635378./230739211,5731566787./1027545527,5232866602./850066563,-4093664535./808688257,3962137247./1805957418,65686358./487910083,0,0],[403863854./491063109,0,0,-5068492393./434740067,-411421997./543043805,652783627./914296604,11173962825./925320556,-13158990841./6184727034,3936647629./1978049680,-160528059./685178525,248638103./1413531060,0,0]])
    b=np.array([14005451./335480064,0.,0.,0.,0.,-59238493./1068277825,181606767./758867731,561292985./797845732,-1041891430./1371343529,760417239./1151165299,118820643./751138087,-528747749./2220607170,1./4])
    bhat=np.array([13451932./455176623,0.,0.,0.,0.,-808719846./976000145,1757004468./5645159321,656045339./265891186,-3867574721./1518517206,465885868./322736535,53011238./667516719,2./45,0.])
    RK['DP8']=ExplicitRungeKuttaPair(A,b,bhat,name='Dormand-Prince RK8(7)13')
    #================================================
    A=np.array([[0,0,0,0,0,0,0], [0.392382208054010,0,0,0,0,0,0],
                [0.310348765296963 ,0.523846724909595 ,0,0,0,0,0],[0.114817342432177 ,0.248293597111781 ,0,0,0,0,0],
                [0.136041285050893 ,0.163250087363657 ,0,0.557898557725281 ,0,0,0],
                [0.135252145083336 ,0.207274083097540 ,-0.180995372278096 ,0.326486467604174 ,0.348595427190109 ,0,0],
                [0.082675687408986 ,0.146472328858960 ,-0.160507707995237 ,0.161924299217425 ,0.028864227879979 ,0.070259587451358 ,0]])
    b=np.array([0.110184169931401 ,0.122082833871843 ,-0.117309105328437 ,0.169714358772186, 0.143346980044187, 0.348926696469455, 0.223054066239366])
    RK['SSP75']=ExplicitRungeKuttaMethod(A,b,name='SSP75',description='From Ruuth-Spiteri paper',shortname='SSP75')
    #================================================
    A=np.array([[0,0,0,0,0,0,0,0],[0.276409720937984 ,0,0,0,0,0,0,0],[0.149896412080489 ,0.289119929124728 ,0,0,0,0,0,0],
                [0.057048148321026 ,0.110034365535150 ,0.202903911101136 ,0,0,0,0,0],
                [0.169059298369086 ,0.326081269617717 ,0.450795162456598 ,0,0,0,0,0],
                [0.061792381825461 ,0.119185034557281 ,0.199236908877949 ,0.521072746262762 ,-0.001094028365068 ,0,0,0],
                [0.111048724765050 ,0.214190579933444 ,0.116299126401843 ,0.223170535417453 ,-0.037093067908355 ,0.228338214162494 ,0,0],
                [0.071096701602448 ,0.137131189752988 ,0.154859800527808 ,0.043090968302309 ,-0.163751550364691 ,0.044088771531945 ,0.102941265156393 ,0]])
    b=np.array([0.107263534301213 ,0.148908166410810 ,0.105268730914375 ,0.124847526215373 ,-0.068303238298102 ,0.127738462988848 ,0.298251879839231 ,0.156024937628252 ])
    RK['SSP85']=ExplicitRungeKuttaMethod(A,b,name='SSP85',description='From Ruuth-Spiteri paper',shortname='SSP85')
    #================================================
    A=np.array([[0,0,0,0,0,0,0,0,0],[0.234806766829933 ,0,0,0,0,0,0,0,0],
                [0.110753442788106 ,0.174968893063956 ,0,0,0,0,0,0,0],
                [0.050146926953296 ,0.079222388746543 ,0.167958236726863 ,0,0,0,0,0,0],
                [0.143763164125647 ,0.227117830897242 ,0.240798769812556 ,0,0,0,0,0,0],
                [0.045536733856107 ,0.071939180543530 ,0.143881583463234 ,0.298694357327376 ,-0.013308014505658,0,0,0,0],
                [0.058996301344129 ,0.093202678681501 ,0.109350748582257 ,0.227009258480886 ,-0.010114159945349 ,0.281923169534861 ,0,0,0],
                [0.114111232336224 ,0.180273547308430 ,0.132484700103381 ,0.107410821979346 ,-0.129172321959971 ,0.133393675559324 ,0.175516798122502 ,0,0],
                [0.096188287148324 ,0.151958780732981 ,0.111675915818310 ,0.090540280530361 ,-0.108883798219725 ,0.112442122530629 ,0.147949153045843 ,0.312685695043563 ,0]])
    b=np.array([0.088934582057735 ,0.102812792947845 ,0.111137942621198 ,0.158704526123705 ,-0.060510182639384 ,0.197095410661808 ,0.071489672566698 ,0.151091084299943 ,0.179244171360452 ])
    RK['SSP95']=ExplicitRungeKuttaMethod(A,b,name='SSP95',description='From Ruuth-Spiteri paper',shortname='SSP95')

    if which=='All':
        return RK
    else:
        return RK[which]

#============================================================
# Generic Families of Runge-Kutta methods
#============================================================

def RK22_family(gamma):
    """ 
        Construct a 2-stage second order Runge-Kutta method 

        **Input**: gamma -- family parameter
        **Output**: An ExplicitRungeKuttaMethod
        **Examples**::

            >>> from nodepy import rk
            >>> rk.RK22_family(-1)
            Runge-Kutta Method
            <BLANKLINE>
             0    |
             -1/2 |  -1/2
            ______|____________
                  |     2    -1
    """
    from sympy import Rational
    one = Rational(1,1)

    A=snp.array([[0,0],[one/(2*gamma),0]])
    b=snp.array([one-gamma,gamma])
    return ExplicitRungeKuttaMethod(A,b)

def RK44_family(w):
    """ 
        Construct a 4-stage fourth order Runge-Kutta method 

        **Input**: w -- family parameter
        **Output**: An ExplicitRungeKuttaMethod

        **Examples**::

            >>> from nodepy import rk
            >>> rk.RK44_family(1)
            Runge-Kutta Method
            <BLANKLINE>
             0   |
             1/2 |  1/2
             1/2 |  1/3   1/6
             1   |  0     -2    3
            _____|________________________
                 |   1/6  -1/3     1   1/6
                       
    """
    from sympy import Rational
    one = Rational(1,1)

    A=snp.array([[0,0,0,0],[one/2,0,0,0],[one/2-one/(6*w),one/(6*w),0,0],
                [0,one-3*w,3*w,0]])
    b=snp.array([one/6,2*one/3-w,w,one/6])
    return ExplicitRungeKuttaMethod(A,b)


#============================================================
# Families of optimal SSP Runge-Kutta methods
#============================================================

def SSPRK2(m):
    """ Construct the optimal m-stage, second order SSP 
        Explicit Runge-Kutta method (m>=2).

        **Input**: m -- number of stages
        **Output**: A ExplicitRungeKuttaMethod

        **Examples**::
            
            Load the 4-stage method:
            >>> SSP42=SSPRK2(4)
            >>> SSP42
            SSPRK(4,2)
            <BLANKLINE>
             0   |
             1/3 |  1/3
             2/3 |  1/3  1/3
             1   |  1/3  1/3  1/3
            _____|____________________
                 |  1/4  1/4  1/4  1/4

            >>> SSP42.absolute_monotonicity_radius()
            2.999999999974534

        **References**: 
            #. [ketcheson2008]_
    """
    from sympy import Rational
    assert m>=2, "SSPRKm2 methods must have m>=2"
    one = Rational(1)
    r=m-one
    alpha=np.vstack([snp.zeros(m),snp.eye(m)])
    alpha[m,m-1]=(m-one)/m
    beta=alpha/r
    alpha[m,0]=one/m
    name='SSPRK('+str(m)+',2)'
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name)


def SSPRK3(m):
    """ 
        Construct the optimal m-stage third order SSP
        Runge-Kutta method (m=n**2, n>=2) 

        **Input**: m -- number of stages
        **Output**: A RungeKuttaMethod

        **Examples**::

            Load the 4-stage method:
            >>> SSP43=SSPRK3(4)
            >>> SSP43
            SSPRK43
            <BLANKLINE>
             0   |
             1/2 |  1/2
             1   |  1/2  1/2
             1/2 |  1/6  1/6  1/6
            _____|____________________
                 |  1/6  1/6  1/6  1/2

            >>> SSP43.absolute_monotonicity_radius()
            1.9999999999527063

        **References**: 
            #. [ketcheson2008]_

    """
    from sympy import sqrt, Rational
    one = Rational(1)

    n = sqrt(m)
    assert n==int(n), "SSPRKm3 methods must have m=n^2"
    assert m>=4, "SSPRKm3 methods must have m>=4"
    r = m - n
    alpha=np.vstack([snp.zeros(m),snp.eye(m)])
    alpha[n*(n+1)/2,n*(n+1)/2-1]=(n-one)/(2*n-one)
    beta=alpha/r
    alpha[n*(n+1)/2,(n-1)*(n-2)/2]=n/(2*n-one)
    name='SSPRK'+str(m)+'3'
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name)



def SSPRKm(m):
    """ Construct the optimal m-stage, linearly mth order SSP 
        Explicit Runge-Kutta method (m>=2).

        **Input**: m -- number of stages
        **Output**: A ExplicitRungeKuttaMethod

        **Examples**::
            
            Load the 4-stage method:
            >>> SSP44=SSPRKm(4)
            >>> SSP44
            SSPRK44
            <BLANKLINE>
             0 |
             1 |  1
             2 |  1     1
             3 |  1     1     1
            ___|________________________
               |   5/8  7/24  1/24  1/24


            >>> SSP44.absolute_monotonicity_radius()
            0.9999999999308784

        **References**: 
            #. [gottlieb2001]_
    """
    from sympy import factorial, Rational

    assert m>=2, "SSPRKm methods must have m>=2"

    alph=snp.zeros([m+1,m+1])
    alph[1,0]=1
    for mm in range(2,m+1):
        for k in range(1,m):
            alph[mm,k]= Rational(alph[mm-1,k-1],k)
            alph[mm,mm-1]=Rational(1,factorial(mm))
            alph[mm,0] = 1-sum(alph[mm,1:])

    alpha=np.vstack([snp.zeros(m),snp.eye(m)])
    alpha[m,m-1]=Rational(1/factorial(m))
    beta=alpha.copy()
    alpha[m,1:m-1]=alph[m,1:m-1]
    alpha[m,0] = 1-sum(alpha[m,1:])
    name='SSPRK'+str(m)*2
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
            <BLANKLINE>
             1/4 |  1/4  0    0    0
             1/2 |  1/4  1/4  0    0
             3/4 |  1/4  1/4  1/4  0
             1   |  1/4  1/4  1/4  1/4
            _____|____________________
                 |  1/4  1/4  1/4  1/4
    """
    A=snp.tri(m)/m
    b=snp.ones(m)/m
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
            <BLANKLINE>
             1/8 |  1/8  0    0    0
             3/8 |  1/4  1/8  0    0
             5/8 |  1/4  1/4  1/8  0
             7/8 |  1/4  1/4  1/4  1/8
            _____|____________________
                 |  1/4  1/4  1/4  1/4

            >>> ISSP42.absolute_monotonicity_radius()
            7.999999999992724

        **References**:
            #. [ketcheson2009]_
    """
    from sympy import Rational
    r=2*m
    alpha=np.vstack([snp.zeros(m),snp.eye(m)])
    beta=alpha/r
    for i in range(m): beta[i,i]=Rational(1,r)
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
            <BLANKLINE>
             -15**(1/2)/10 + 1/2 |  -15**(1/2)/10 + 1/2  0                    0                    0
             -15**(1/2)/30 + 1/2 |  15**(1/2)/15         -15**(1/2)/10 + 1/2  0                    0
             15**(1/2)/30 + 1/2  |  15**(1/2)/15         15**(1/2)/15         -15**(1/2)/10 + 1/2  0
             15**(1/2)/10 + 1/2  |  15**(1/2)/15         15**(1/2)/15         15**(1/2)/15         -15**(1/2)/10 + 1/2
            _____________________|____________________________________________________________________________________
                                 |                  1/4                  1/4                  1/4                  1/4

            >>> ISSP43.absolute_monotonicity_radius()
            6.872983343691885

        **References**:
            #. [ketcheson2009]_
    """
    from sympy import sqrt, Rational
    r=m-1+sqrt(m**2-1)
    alpha=np.vstack([snp.zeros(m),snp.eye(m)])
    alpha[-1,-1]=((m+1)*r)/(m*(r+2))
    beta=alpha/r
    for i in range(m): beta[i,i]=Rational(1,2)*(1-sqrt(Rational(m-1,m+1)))
    name='SSPIRK'+str(m)+'3'
    return RungeKuttaMethod(alpha=alpha,beta=beta,name=name)


#============================================================
# Families of Runge-Kutta-Chebyshev methods
#============================================================
def RKC1(m,epsilon=0):
    """ Construct the m-stage, first order 
        explicit Runge-Kutta-Chebyshev methods of Verwer (m>=1).

        'epsilon' is a damping parameter used to avoid tangency of the
        stability region boundary to the negative real axis.

        **Input**: m -- number of stages
        **Output**: A ExplicitRungeKuttaMethod

        **Examples**::
            
            Load the 4-stage method:
            >>> RKC41=RKC1(4)
            >>> RKC41
            RKC41
            <BLANKLINE>
             0    |
             1/16 |  1/16
             1/4  |  1/8   1/8
             9/16 |  3/16  1/4   1/8
            ______|________________________
                  |   1/4   3/8   1/4   1/8

        **References**: 
            #. [verwer2004]_
    """

    import sympy
    one = sympy.Rational(1)

    x=sympy.Symbol('x')
    Tm=sympy.special.polynomials.chebyshevt_poly(m,x)

    w0=one+epsilon/m**2
    w1=sympy.Rational(Tm.subs(x,w0),Tm.diff().subs(x,w0))

    alpha=snp.zeros([m+1,m])
    beta=snp.zeros([m+1,m])

    b=snp.zeros(m+1)
    a=snp.zeros(m+1)
    mu=snp.zeros(m+1)
    nu=snp.zeros(m+1)
    mut=snp.zeros(m+1)
    gamt=snp.zeros(m+1)

    b[0]=one
    b[1]=one/w0
    mut[1] = b[1]*w1
    alpha[1,0]=one
    beta[1,0]=mut[1]

    for j in range(2,m+1):
        Tj=sympy.special.polynomials.chebyshevt_poly(j,x)
        b[j] = one/Tj.subs(x,w0)
        a[j] = one-b[j]*Tj.subs(x,w0)
        mu[j]= 2*b[j]*w0/b[j-1]
        nu[j]= -b[j]/b[j-2]
        mut[j] = mu[j]*w1/w0
        gamt[j] = -a[j-1]*mut[j]

        alpha[j,0]=one-mu[j]-nu[j]
        alpha[j,j-1]=mu[j]
        alpha[j,j-2]=nu[j]
        beta[j,j-1]=mut[j]
        beta[j,0]=gamt[j]

    name='RKC'+str(m)+'1'
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name)


def RKC2(m,epsilon=0):
    """ Construct the m-stage, second order 
        Explicit Runge-Kutta-Chebyshev methods of Verwer (m>=2).

        **Inputs**: 
                m -- number of stages
                epsilon -- damping factor

        **Output**: A ExplicitRungeKuttaMethod

        **Examples**::
            
            Load the 4-stage method:
            >>> RKC42=RKC2(4)
            >>> RKC42
            RKC42
            <BLANKLINE>
             0    |
             1/5  |  1/5
             1/5  |  1/10    1/10
             8/15 |  -8/45   32/135  64/135
            ______|________________________________
                  |  -51/64     3/8       1   27/64

        **References**: 
            #. [verwer2004]_
    """
    import sympy
    one = sympy.Rational(1)

    x=sympy.Symbol('x')
    Tm=sympy.special.polynomials.chebyshevt_poly(m,x)

    w0=one+epsilon/m**2
    w1=sympy.Rational(Tm.diff().subs(x,w0),Tm.diff(x,2).subs(x,w0))

    alpha=snp.zeros([m+1,m])
    beta=snp.zeros([m+1,m])

    b=snp.zeros(m+1)
    a=snp.zeros(m+1)
    mu=snp.zeros(m+1)
    nu=snp.zeros(m+1)
    mut=snp.zeros(m+1)
    gamt=snp.zeros(m+1)
    
    T2 = sympy.special.polynomials.chebyshevt_poly(2,x)
    b[0]=sympy.Rational(T2.diff(x,2).subs(x,w0),(T2.diff().subs(x,w0))**2)

    b[1]=one/w0
    mut[1] = b[1]*w1
    alpha[1,0]=one
    beta[1,0]=mut[1]

    for j in range(2,m+1):
        Tj=sympy.special.polynomials.chebyshevt_poly(j,x)
        b[j] = sympy.Rational(Tj.diff(x,2).subs(x,w0),(Tj.diff().subs(x,w0))**2)

        a[j] = one-b[j]*Tj.subs(x,w0)
        mu[j]= 2*b[j]*w0/b[j-1]
        nu[j]= -b[j]/b[j-2]
        mut[j] = mu[j]*w1/w0
        gamt[j] = -a[j-1]*mut[j]

        alpha[j,0]=one-mu[j]-nu[j]
        alpha[j,j-1]=mu[j]
        alpha[j,j-2]=nu[j]
        beta[j,j-1]=mut[j]
        beta[j,0]=gamt[j]

    name='RKC'+str(m)+'2'
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name)

#============================================================
# Spectral deferred correction methods
#============================================================
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
    n=snp.arange(len(x))+1
    for i in range(len(x)-1):
        a=x[i]; b=x[i+1]
        f=(b**n-a**n)/n
        F[:,i]=f
    w=snp.solve(A,F)

    return w

def DC_pair(s):

    if s<2:
        raise Exception('s must be equal to or greater than 2')
    dc = DC(s)
    name='Deferred Correction pair of order '+str(s+1)+'('+str(s)+')'
    return ExplicitRungeKuttaPair(A=dc.A,b=dc.b,bhat=dc.A[-1],name=name).dj_reduce()


def DC(s,theta=0.,grid='eq'):
    """ Spectral deferred correction methods.
        For now, based on explicit Euler and equispaced points.
        TODO: generalize base method and grid.

        **Input**: s -- number of grid points & number of correction iterations

        **Output**: A ExplicitRungeKuttaMethod

        Note that the number of stages is NOT equal to s.  The order
        is equal to s+1.

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
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name).dj_reduce()


#============================================================
# Extrapolation methods
#============================================================
def extrap(s,seq='harmonic'):
    """ Construct extrapolation methods.
        For now, based on explicit Euler, but allowing arbitrary sequences.

        **Input**: s -- number of grid points & number of correction iterations
                   seq -- extrapolation sequence

        **Output**: A ExplicitRungeKuttaMethod

        Note that the number of stages is NOT equal to s.  The order
        is equal to s.


        **Examples**::

            >>> from nodepy import rk
            >>> ex3 = rk.extrap(3)
            >>> ex3
            extrap3
            <BLANKLINE>
             0   |
             1/2 |  1/2
             1/3 |  1/3  0
             2/3 |  1/3  0    1/3
            _____|____________________
                 |    0   -2  3/2  3/2

            >>> ex3.num_seq_dep_stages()
            3
            >>> ex3.principal_error_norm()
            0.04606423319938055

        **References**: 

            #. [Hairer]_ chapter II.9

        **TODO**: 

            - generalize base method
    """

    if seq=='harmonic': 
        N=snp.arange(s)+1
    elif seq=='Romberg': 
        N=snp.arange(s)+1;  N=2**(N-1)

    J=np.cumsum(N)+1
    
    # Number of real stages:
    nrs = J[-1]

    # Shu-Osher arrays
    alpha=snp.zeros([nrs+s*(s-1)/2,nrs+s*(s-1)/2-1])
    beta=snp.zeros([nrs+s*(s-1)/2,nrs+s*(s-1)/2-1])

    # T_11
    alpha[1,0]=1
    beta[1,0]=1/N[0]

    for j in range(1,len(N)):
        #Form T_j1:
        alpha[J[j-1],0] = 1
        beta[J[j-1],0]=1/N[j]

        for i in range(1,N[j]):
            alpha[J[j-1]+i,J[j-1]+i-1]=1
            beta[J[j-1]+i,J[j-1]+i-1]=1/N[j]
    
    #Really there are no more "stages", and we could form T_ss directly.
    #but it is simpler to add auxiliary stages and then reduce.
    for j in range(1,s):
        #form T_{j+1,2}:
        alpha[nrs-1+j,J[j]-1]=1+1/(N[j]/N[j-1]-1)
        alpha[nrs-1+j,J[j-1]-1]=-1/(N[j]/N[j-1]-1)

    #Now form all the rest, up to T_ss:
    nsd = nrs-1+s # Number of stages done
    for k in range(2,s):
        for ind,j in enumerate(range(k,s)):
            #form T_{j+1,k+1}:
            alpha[nsd+ind,nsd-(s-k)+ind] = 1+1/(N[j]/N[j-k]-1)
            alpha[nsd+ind,nsd-(s-k)+ind-1] = -1/(N[j]/N[j-k]-1)
        nsd += s-k

    name='extrap'+str(s)
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name).dj_reduce()

def extrap_gbs(s,embedded=False, shuosher=False):
    """ Construct extrapolation methods based on GBS.

        **Input**: s -- number of extrapolation terms

        **Output**: A ExplicitRungeKuttaMethod

        Note that the number of stages is NOT equal to s.  The order
        is equal to 2s.

        **References**: 

            #. [Hairer]_ chapter II.9
    """

    # Only supports harmonic sequence right now
    N=2*snp.arange(s)+2
    J=np.cumsum(N)+1
    order_reducer=0
    if embedded:
        if s>1:
            order_reducer=1
        else:
            raise Exception('Embedded pair must have order > 0')
    nrs = J[-1]
    
    alpha=snp.zeros([nrs+s*(s-1)/2-order_reducer,nrs+s*(s-1)/2-1-order_reducer])
    beta=snp.zeros([nrs+s*(s-1)/2-order_reducer,nrs+s*(s-1)/2-1-order_reducer])
    
    # T_11
    alpha[1,0]=1
    alpha[2,0]=1
    beta[1,0]=1/N[0]
    beta[2,1]=2/N[0]

    for j in range(1,len(N)):
        #Form T_j1:
        alpha[J[j-1],0] = 1
        alpha[J[j-1]+1,0] = 1
        beta[J[j-1],0]=1/N[j]
        beta[J[j-1]+1,J[j-1]]=2/N[j]
        for i in range(1,int(N[j]/2)):
            alpha[J[j-1]+2+2*(i-1),J[j-1]+2*(i-1)  ]=1
            alpha[J[j-1]+3+2*(i-1),J[j-1]+2*(i-1)+1]=1
            beta[ J[j-1]+2+2*(i-1),J[j-1]+2*(i-1)+1]=2/N[j]
            beta[ J[j-1]+3+2*(i-1),J[j-1]+2*(i-1)+2]=2/N[j]
    
    #Really there are no more "stages", and we could form T_ss directly
    if (embedded and s>2) or not embedded:
        for j in range(1,s):
            #form T_{j+1,2}:
            alpha[nrs-1+j,J[j]-1]=1+1/((N[j]/N[j-1])**2-1)
            alpha[nrs-1+j,J[j-1]-1]=-1/((N[j]/N[j-1])**2-1)
    
    #Now form all the rest, up to T_ss

    nsd = nrs-1+s
    for k in range(2,s-order_reducer):
        for ind,j in enumerate(range(k,s)):
            #form T_{j+1,k+1}:
            alpha[nsd+ind,nsd-(s-k)+ind] = 1+1/((N[j]/N[j-k])**2-1)
            alpha[nsd+ind,nsd-(s-k)+ind-1] = -1/((N[j]/N[j-k])**2-1)
        nsd += s-k
    name='GBS extrapolation method of order '+str(2*s)
    if shuosher:
        return alpha, beta
    else:
        return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name).dj_reduce()
   

def extrap_gbs_pair(s, seq='harmonic'):
    """ 
        Returns an embedded RK pair.  The prinicpal method has
        order 2s and the embedded method has order 2(s-1).
        Both methods are constructed based on GBS extrapolation.
    """
    if s<2:
        raise Exception('Embedded pair must have order > 0')

    alpha1, beta1 = extrap_gbs(s, shuosher=True)
    alpha2, beta2 = extrap_gbs(s,embedded=True, shuosher=True)

    rk1 = ExplicitRungeKuttaMethod(alpha=alpha1,beta=beta1)
    rk2 = ExplicitRungeKuttaMethod(alpha=alpha2,beta=beta2)
    bhat = np.resize(rk2.b,rk1.b.shape)

    name='GBS extrapolation pair of order '+str(2*s)+'('+str(2*s-2)+')'
    return ExplicitRungeKuttaPair(A=rk1.A, b=rk1.b, bhat=bhat, name=name).dj_reduce()


#============================================================
# Miscellaneous functions
#============================================================
def rk_order_conditions_hardcoded(rkm,p,tol):
    """ 
        Returns a vector that is identically zero if the
        Runge-Kutta method satisfies the conditions of order p (only) 

        This function involves explicitly coded order conditions up to
        order six.  It is deprecated for now.
    """
    print 'rk_order_conditions_hardcoded: This function is deprecated!'
    A=rkm.A
    b=rkm.b
    c=rkm.c
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
        raise Exception('Order conditions not implemented for p>6')
    return z


def runge_kutta_order_conditions(p,ind='all'):
    """ 
        This is the current method of producing the code on-the-fly
        to test order conditions for RK methods.  May be deprecated
        soon.
    """
    import rooted_trees as rt
    strings=rt.recursiveVectors(p,ind)
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
    """ The method obtained by applying
    RK2, followed by RK1, each with half the timestep.

    **Output**::

        The method
             c_2 | A_2  0
           1+c_1 | b_2 A_1
           _____________
                 | b_2 b_1

        but with everything divided by two.
        The b_2 matrix block consists of m_1 (row) copies of b_2.

    **Examples**::

        What method is obtained by two successive FE steps?
        >>> from nodepy import rk
        >>> fe=rk.loadRKM('FE')
        >>> fe*fe
        Runge-Kutta Method
        <BLANKLINE>
         0     |
         0.500 |  0.500
        _______|______________
               |  0.500  0.500
                

    TODO: Generalize this for any number of inputs
    """
    f1=h1/(h1+h2)
    f2=h2/(h1+h2)
    A=np.vstack([
    np.hstack([RK2.A*f2,np.zeros([np.size(RK2.A,0),np.size(RK1.A,1)])]),
        np.hstack([np.tile(RK2.b*f2,(len(RK1),1)),RK1.A*f1])]).squeeze()
    b=np.hstack([RK2.b*f2,RK1.b*f1]).squeeze()
    if RK1.is_explicit() and RK2.is_explicit():
        return ExplicitRungeKuttaMethod(A,b)
    else:
        return RungeKuttaMethod(A,b)

def plot_rational_stability_region(p,q,N=200,bounds=[-10,1,-5,5],
                          color='r',filled=True,scaled=False):
    r""" 
        Plot the region of absolute stability
        of a rational function i.e. the set

        `\{ z \in C : |\phi (z)|\le 1 \}`

        where `\phi(z)=p(z)/q(z)` is the rational function.

        **Input**: 
            required
                - p       -- numerator (numpy.poly1d)
                - p       -- denominator (numpy.poly1d)
            
            optional
                - N       -- Number of gridpoints to use in each direction
                - bounds  -- limits of plotting region
                - color   -- color to use for this plot
                - filled  -- if true, stability region is filled in (solid); otherwise it is outlined
    """
    m=len(p)
    x=np.linspace(bounds[0],bounds[1],N)
    y=np.linspace(bounds[2],bounds[3],N)
    X=np.tile(x,(N,1))
    Y=np.tile(y[:,np.newaxis],(1,N))
    Z=X+Y*1j
    if not scaled: R=np.abs(p(Z)/q(Z))
    else: R=np.abs(p(Z*m)/q(Z*m))
    if filled:
        pl.contourf(X,Y,R,[0,1],colors=color)
    else:
        pl.contour(X,Y,R,[0,1],colors=color)
    pl.hold(True)
    pl.plot([0,0],[bounds[2],bounds[3]],'--k',linewidth=2)
    pl.plot([bounds[0],bounds[1]],[0,0],'--k',linewidth=2)
    pl.axis('Image')
    pl.hold(False)
    pl.show()

def python_to_matlab(code):
    r"""
        Convert python code string (order condition) to matlab code string
        Doesn't really work yet.  We need to do more parsing.
    """
    #print code
    outline=code
    outline=outline.replace("**",".^")
    outline=outline.replace("*",".*")
    outline=outline.replace("dot(b,","b'*(")
    outline=outline.replace("dot(bhat,","bhat'*(")
    outline=outline.replace("dot(Ahat,","Ahat*(")
    outline=outline.replace("dot(A,","(A*")
    outline=outline.replace("( c)","c")
    outline=outline.replace("-0","")
    #print outline
    #print '******************'
    return outline

def relative_accuracy_efficiency(rk1,rk2):
    r"""
    Compute the accuracy efficiency of method rk1 relative to that of rk2,
    for two methods with the same order of accuracy.

    The relative accuracy efficiency is

    `\eta = \frac{s_2}{s_1} \left(\frac{A_2}{A_1}\right)^{1/p+1}`

    where `s_1,s_2` are the number of stages of the two methods and
    `A_1,A_2` are their principal error norms.

    If the result is >1, method 1 is more efficient.

    **Examples**::

        Compare Fehlberg's method with Dormand-Prince
        >>> from nodepy import rk
        >>> dp5 = rk.loadRKM('DP5')
        >>> f45 = rk.loadRKM('Fehlberg45')
        >>> rk.relative_accuracy_efficiency(dp5,f45)
        1.2222911649987882
    """

    p=rk1.order()
    if rk2.order()!=p: raise Exception('Methods have different orders')

    A1=rk1.principal_error_norm()
    A2=rk2.principal_error_norm()

    return len(rk2)/len(rk1) * (A2/A1)**(1./(p+1))

def linearly_stable_step_size(rk, L, acc=1.e-7, plot=1):
    r"""
        Determine the maximum linearly stable step size for Runge-Kutta method
        rk applied to the IVP `u' = Lu`, by computing the eigenvalues of `L`
        and determining the values of the stability function of rk at the eigenvalues.

        Note that this analysis is not generally appropriate if L is non-normal.

        **Examples**::

            >>> from nodepy import rk, semidisc

            4th-order Runge-Kutta scheme:
            >>> rk44=rk.loadRKM('RK44')

            Centered differences on a grid with spacing 1/100:
            >>> L1=semidisc.centered_diffusion_matrix(100)
            >>> L2=semidisc.centered_advection_diffusion_matrix(1.,1./500,100)

            >>> rk.linearly_stable_step_size(rk44,L1,plot=0)
            6.9580078125000116e-05
            >>> rk.linearly_stable_step_size(rk44,L2,plot=0)
            0.024225094627893518
    """

    from utils import bisect
    import matplotlib.pyplot as plt

    tol=1.e-14
    p,q = rk.__num__().stability_function()
    lamda = np.linalg.eigvals(L)
    hmax = 2.5*len(rk)**2 / max(abs(lamda))
    h=bisect(0,hmax,acc,tol,_is_linearly_stable, params=(p,q,lamda))
    if plot:
        rk.plot_stability_region()
        plt.hold(True)
        plt.plot(np.real(h*lamda), np.imag(h*lamda),'o')
    return h


def _is_linearly_stable(h,tol,params):
    p=params[0]
    q=params[1]
    lamda=params[2]
    R = abs(p(h*lamda)/q(h*lamda))
    if max(R) > 1.+tol:
        return 0
    else:
        return 1

def _get_column_widths(coefflists):
    lenmax = []
    for coefflist in coefflists:
        lenmax.append(max([len(ai) for ai in coefflist]))
    colmax=max(lenmax)
    return lenmax, colmax
 

if __name__ == "__main__":
    import doctest
    doctest.testmod()
