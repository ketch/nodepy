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
    >>> sorted(RK.keys())
    ['BE', 'BS3', 'BS5', 'BuRK65', 'CK5', 'CMR6', 'DP5', 'FE', 'Fehlberg43', 'Fehlberg45', 'GL2', 'GL3', 'HH5', 'HH5S', 'Heun33', 'Lambert65', 'LobattoIIIA2', 'LobattoIIIA3', 'LobattoIIIC2', 'LobattoIIIC3', 'LobattoIIIC4', 'MTE22', 'Merson43', 'Mid22', 'NSSP32', 'NSSP33', 'PD8', 'RK44', 'RadauIIA2', 'RadauIIA3', 'SDIRK23', 'SDIRK34', 'SDIRK54', 'SSP104', 'SSP22', 'SSP22star', 'SSP33', 'SSP53', 'SSP54', 'SSP63', 'SSP75', 'SSP85', 'SSP95', 'TR-BDF2', 'Tsit5']

    >>> print(RK['Mid22'])
    Midpoint Runge-Kutta
    <BLANKLINE>
     0   |
     1/2 | 1/2
    _____|__________
         | 0    1

* Many methods are naturally implemented in some Shu-Osher form different
  from the Butcher form::

    >>> ssp42 = SSPRK2(4)
    >>> ssp42.print_shu_osher()
    SSPRK(4,2)
    <BLANKLINE>
         |                     |
     1/3 | 1                   | 1/3
     2/3 |      1              |      1/3
     1   |           1         |           1/3
    _____|_____________________|_____________________
         | 1/4            3/4  |                1/4


**References**:
    * :cite:`butcher2003`
    * :cite:`hairer1993`
"""
from __future__ import print_function
from __future__ import division

from __future__ import absolute_import
import numpy as np
import sympy

import nodepy.snp as snp
from nodepy.general_linear_method import GeneralLinearMethod
from six.moves import range

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
                 name='Runge-Kutta Method',shortname='RKM',
                 description='',mode='exact',order=None):
        r"""
            Initialize a Runge-Kutta method.  For explicit methods,
            the class ExplicitRungeKuttaMethod should be used instead.

            TODO: make A a property and update c when it is changed

            Now that we store (alpha,beta) as auxiliary data,
            maybe it's okay to specify both `(A,b)` and `(\alpha,\beta)`.
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

        if alpha is None and beta is None:
            s = A.shape[0]
            if A.dtype == object:
                alpha = snp.normalize(np.zeros((s+1,s),dtype=object))
                beta = snp.normalize(np.zeros((s+1,s),dtype=object))
            else:
                alpha = np.zeros((s+1,s))
                beta = np.zeros((s+1,s))
            beta[:-1,:] = A.copy()
            beta[-1,:] = b.copy()

        self.alpha=alpha
        self.beta=beta

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
                print("""Warning: this method appears to be explicit, but is
                       being initialized as a RungeKuttaMethod rather than
                       as an ExplicitRungeKuttaMethod.""")

        if order is not None:
            self._p = order
        else:
            self._p = None

    @property
    def p(self):
        r"""Order of the method.  This can be imposed and cached, which is advantageous
        to avoid issues with roundoff error and slow computation of the order conditions."""
        if self._p is None:
            self._p = self.order()
        return self._p
    @p.setter
    def p(self,p):
        self._p = p

    def __num__(self):
        """
        Returns a copy of the method but with floating-point coefficients.
        This is useful whenever we need to operate numerically without
        worrying about the representation of the method.
        """
        import copy
        numself = copy.deepcopy(self)
        if self.A.dtype == object or self.b.dtype == object:
            numself.A=np.array(self.A,dtype=np.float64)
            numself.b=np.array(self.b,dtype=np.float64)
            numself.c=np.array(self.c,dtype=np.float64)
            numself.alpha=np.array(self.alpha,dtype=np.float64)
            numself.beta=np.array(self.beta,dtype=np.float64)
        return numself

    def latex(self):
        r"""A laTeX representation of the Butcher arrays.

            **Example**::

                >>> from nodepy import rk
                >>> merson = rk.loadRKM('Merson43')
                >>> print(merson.latex())
                \begin{align}
                \begin{array}{c|ccccc}
                 &  &  &  &  & \\
                \frac{1}{3} & \frac{1}{3} &  &  &  & \\
                \frac{1}{3} & \frac{1}{6} & \frac{1}{6} &  &  & \\
                \frac{1}{2} & \frac{1}{8} &  & \frac{3}{8} &  & \\
                1 & \frac{1}{2} &  & - \frac{3}{2} & 2 & \\
                \hline
                 & \frac{1}{6} &  &  & \frac{2}{3} & \frac{1}{6}\\
                 & \frac{1}{10} &  & \frac{3}{10} & \frac{2}{5} & \frac{1}{5}
                \end{array}
                \end{align}
        """
        from nodepy.snp import printable
        sep = ' & '
        s= r'\begin{align}' + '\n'
        s+=r'\begin{array}{c|'
        s+='c'*len(self)
        s+='}\n'
        for i in range(len(self)):
            s+=printable(self.c[i]) + sep
            s += sep.join([printable(aij) for aij in self.A[i,:]])
            s+=r'\\' + '\n'
        s+=r'\hline' + '\n'
        s += sep
        s += sep.join([printable(bj) for bj in self.b])
        s+=r'\\' + '\n'
        if hasattr(self,'bhat'):
            s += sep
            s += sep.join([printable(bj) for bj in self.bhat])
            s += '\n'
        s += r'\end{array}' + '\n'
        s += r'\end{align}'
        s=s.replace('- -','')
        return s


    def print_shu_osher(self):
        r"""
        Pretty-prints the Shu-Osher arrays in the form::

              |        |
            c | \alpha | \beta
            ______________________
              | amp1   | bmp1

        where amp1, bmp1 represent the last rows of `\alpha,\beta`.
        """
        if (self.alpha is None or self.beta is None):
            raise Exception('Shu-Osher arrays not defined for this method.')

        from nodepy.utils import array2strings

        c = array2strings(self.c)
        alpha = array2strings(self.alpha)
        beta  = array2strings(self.beta)
        lenmax, colmax = _get_column_widths([alpha,beta,c])

        s=self.name+'\n'+self.info+'\n'
        for i in range(len(self)):
            s+=c[i].ljust(colmax+1)+'|'
            for j in range(len(self)):
                s+=alpha[i,j].ljust(colmax+1)
            s+=' |'
            for j in range(len(self)):
                s+=beta[i,j].ljust(colmax+1)
            s=s.rstrip()+'\n'
        s+='_'*(colmax+1)+('|_'+'_'*(colmax+1)*len(self))*2+'\n'
        s+= ' '*(colmax+1)+'|'
        for j in range(len(self)):
            s+=alpha[-1,j].ljust(colmax+1)
        s+=' |'
        for j in range(len(self)):
            s+=beta[-1,j].ljust(colmax+1)
        print(s.rstrip())


    def __str__(self):
        """
        Pretty-prints the Butcher array in the form

          |
        c | A
        ______
          | b
        """
        from nodepy.utils import array2strings

        c = array2strings(self.c,printzeros=True)
        A = array2strings(self.A)
        b = array2strings(self.b,printzeros=True)
        lenmax, colmax = _get_column_widths([A,b,c])

        s=self.name+'\n'+self.info+'\n'
        for i in range(len(self)):
            s+=c[i].ljust(colmax+1)+'|'
            for j in range(len(self)):
                s+=A[i,j].ljust(colmax+1)
            s=s.rstrip()+'\n'
        s+='_'*(colmax+1)+'|'+('_'*(colmax+1)*len(self))+'\n'
        s+= ' '*(colmax+1)+'|'
        for j in range(len(self)):
            s+=b[j].ljust(colmax+1)
        return s.rstrip()

    def __eq__(self,rkm):
        """
            Methods considered equal if their Butcher arrays are

            TODO: Instead check whether methods have the same elementary weights
                  up to some order.

            **Example**:

            Load a method and try do DJ-reduce it::

                >>> from nodepy import rk
                >>> merson = rk.loadRKM('Merson43')
                >>> reduced = merson.dj_reduce()

            Check that the two are actually equal::

                >>> print(reduced == merson)
                True
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

    def _check_consistency(self,tol=1.e-13):
        assert np.max(np.abs(self.A.sum(1)-self.c))<tol,'Abscissae are inconsistent with A.'
        if self.alpha is not None:
            A, b = shu_osher_to_butcher(self.alpha,self.beta)
            assert np.max(np.abs(self.A-A))<tol and np.max(np.abs(self.b-b))<tol, 'Shu-Osher coefficients are not consistent with Butcher coefficients'

    #============================================================
    # Reducibility
    #============================================================

    def _dj_reducible_stages(self,tol=1.e-13):
        """ Determine whether the method is DJ-reducible.

            A method is DJ-reducible if it contains any stage that
            does not influence the output.

            Returns a list of unnecessary stages.  If the method is
            DJ-irreducible, returns an empty list.

            This routine may not work correctly for RK pairs, as it
            doesn't check bhat.
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

                Reduce it:
                >>> print(rkm.dj_reduce())
                Runge-Kutta Method
                <BLANKLINE>
                 0 |
                ___|___
                   | 1
        """
        djs = self._dj_reducible_stages(tol=tol)
        if len(djs)>0:
            for stage in djs[::-1]:
                self._remove_stage(stage)
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

            **Examples**::

                Construct a reducible method:
                >>> from nodepy import rk
                >>> A=np.array([[1,0],[1,0]])
                >>> b=np.array([0.5,0.5])
                >>> rkm = rk.ExplicitRungeKuttaMethod(A,b)

                Check that it is reducible:
                >>> rkm._hs_reducible_stages()
                (True, [0, 1])
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
        """ Eliminate a stage of a DJ-reducible Runge-Kutta method.
            Typically used to reduce reducible methods.

            Note that stages in the NumPy arrays are indexed from zero,
            so to remove stage j use _remove_stage(j-1).
        """
        s = len(self)
        A=np.delete(np.delete(self.A,stage,1),stage,0)
        b=np.delete(self.b,stage)
        c=np.delete(self.c,stage)
        self.A=A
        self.b=b
        self.c=c
        if hasattr(self,'bhat'):
            bhat=np.delete(self.bhat,stage)
            self.bhat=bhat
        if self.alpha is not None:
            for i in range(s+1):
                if self.alpha[i,stage] != 0: # Doing this check speeds things up
                    self.alpha,self.beta = shu_osher_zero_alpha_ij(self.alpha,self.beta,i,stage)
            alpha=np.delete(np.delete(self.alpha,stage,1),stage,0)
            self.alpha = alpha
            beta=np.delete(np.delete(self.beta,stage,1),stage,0)
            self.beta = beta
        if hasattr(self,'alphahat'):
            for i in range(s+1):
                if self.alphahat[i,stage] != 0: # Doing this check speeds things up
                    self.alphahat,self.betahat = shu_osher_zero_alpha_ij(self.alphahat,self.betahat,i,stage)
            alphahat=np.delete(np.delete(self.alphahat,stage,1),stage,0)
            self.alphahat = alphahat
            betahat=np.delete(np.delete(self.betahat,stage,1),stage,0)
            self.betahat = betahat

    #============================================================
    # Accuracy
    #============================================================

    def error_coefficient(self,tree,mode='exact'):
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
        A,b,c = self.A,self.b,self.c

        if A.dtype == object:
            exec('coeff = simplify({} - Rational(1, {}))'.format(code, tree.density()))
        else:
            exec("coeff = ({} - 1.0 / {})".format(code, tree.density()))
        return locals()["coeff"] / tree.symmetry()

    def error_coeffs(self,p):
        r"""
        Returns the coefficients in the Runge-Kutta method's error expansion
        multiplying all elementary differentials of the given order.
        """
        import nodepy.rooted_trees as rt
        forest=rt.list_trees(p)
        err_coeffs=[]
        for tree in forest:
            err_coeffs.append(self.error_coefficient(tree))
        return snp.array(err_coeffs)

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
                (sqrt(1745)/2880, 1/120, sqrt(8531)/5760, 1/144, 1)

            Reference: :cite:`kennedy2000`
        """
        q=self.order()
        tau_1=self.error_coeffs(q+1)
        tau_2=self.error_coeffs(q+2)

        A_qp1 = snp.norm(tau_1)
        A_qp1_max=max([abs(tau) for tau in tau_1])
        A_qp2 = snp.norm(tau_2)
        A_qp2_max=max([abs(tau) for tau in tau_2])

        D=max(np.max(np.abs(self.A)),
                np.max(np.abs(self.b)),np.max(np.abs(self.c)))
        return A_qp1, A_qp1_max, A_qp2, A_qp2_max, D


    def principal_error_norm(self,tol=1.e-13,mode='float'):
        r"""The 2-norm of the vector of leading order error coefficients."""
        import nodepy.rooted_trees as rt
        forest=rt.list_trees(self.p+1)
        errs=[]

        if mode == 'float':
            method = self.__num__()
            from numpy import sqrt
        elif mode == 'exact':
            method = self
            from sympy import sqrt
        else:
            raise Exception('Unrecognized mode value')

        for tree in forest:
            errs.append(method.error_coefficient(tree))
        if mode == 'float':
            return sqrt(float(np.sum(np.array(errs)**2)))
        else:
            return sqrt(np.sum(np.array(errs)**2))

    def order(self,tol=1.e-14,mode='float',extremely_high_order=False):
        """ The order of a Runge-Kutta method.

            **Examples**::

                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> rk4.order()
                4
                >>> rk4.order(mode='exact')
                4

            mode == 'float': (default)
                Check that conditions hold approximately, to within tolerance `tol`.
                Appropriate when coefficients are floating-point, or for faster
                checking of high-order methods.

            mode == 'exact':
                Check that conditions hold exactly.  Appropriate when coefficients
                are specified as rational or algebraic numbers, but may be very
                slow for high order methods.
        """
        if mode=='float':
            if not extremely_high_order:
                import nodepy.oc_butcher as oc_butcher
                p = oc_butcher.order(self.__num__(),tol)
            else:
                import nodepy.oc_butcher_high_order as oc_butcher_high_order
                p = oc_butcher_high_order.order(self.__num__(),tol)
            if p==0:
                print('Apparent order is 0; this may be due to round-off.  Try order(mode="exact") or increase tol.')
        elif mode=='exact':
            from sympy import simplify
            p=0
            while True:
                z=self.order_condition_residuals(p+1)
                z = snp.array([simplify(zz) for zz in z])
                if np.any(abs(z)>tol): break
                p=p+1
        return p

    def order_condition_residuals(self,p):
        """
            Generates and evaluates code to test whether a method
            satisfies the order conditions of order p (only).

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

    def effective_order(self,tol=1.e-14):
        """
            Returns the effective order of a Runge-Kutta method.
            This may be higher than the classical order.

            **Example**:
            >>> from nodepy import rk
            >>> RK4 = rk.loadRKM('RK44')
            >>> RK4.effective_order()
            4

        """
        q=0
        while True:
            if q==4: return q
            z=self.effective_order_condition_residuals(q+1)
            if np.any(abs(z)>tol): return q
            q=q+1

    def effective_order_condition_residuals(self,q):
        """
            Generates and evaluates code to test whether a method
            satisfies the effective order q conditions (only).

            Similar to order_condition_residuals(self,p), but at the moment
            works only for q <= 4. (enough to find Explicit SSPRK)
        """
        from sympy import factorial,Rational
        A,b,c=self.A,self.b,self.c
        C=snp.diag(c)
        code=runge_kutta_order_conditions(q)
        tau=snp.zeros([q,len(self)])
        for j in range(1,q):
            tau[j,:]=(c**j/j-np.dot(A,c**(j-1)))/factorial(j-1)
        if q<=2:
            z=snp.zeros(len(code)+1)
            z[-1]=np.dot(b,c**(q-1))-Rational(1,q)
        if q==3:
            z=snp.zeros(len(code))
            exec('z[0]='+code[0]+'-'+'np.dot(b,c**2)/2.+1/6.')
        if q==4:
            code2=runge_kutta_order_conditions(q-1)
            z=snp.zeros(len(code)-1)
            exec('z[0]='+code[1]+'-'+'np.dot(b,np.dot(A,c**2))/2.+1/24.')
            exec('z[1]='+code2[0]+'-'+code[1]+'-'+code[2])
        if q>4:
            raise NotImplementedError
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
                * :cite:`dekker1984`
                * :cite:`butcher2003`
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
    def stability_function_unexpanded(self):
        r"""Compute the stability function expression but don't simplify it.
            This can be useful for performance reasons.

            **Example**::

                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> rk4.stability_function_unexpanded()
                z*(z/2 + 1)/3 + z*(z*(z/2 + 1)/2 + 1)/3 + z*(z*(z*(z/2 + 1)/2 + 1) + 1)/6 + z/6 + 1
                >>> rk4.stability_function_unexpanded().simplify()
                z**4/24 + z**3/6 + z**2/2 + z + 1
        """
        import sympy
        z = sympy.var('z')
        s = len(self)
        I = sympy.eye(s)

        v = 1 - self.alpha.sum(1)
        vstar = sympy.Matrix(v[:-1])
        v_mp1 = sympy.Rational(v[-1])
        alpha_star = sympy.Matrix(self.alpha[:-1,:])
        beta_star = sympy.Matrix(self.beta[:-1,:])
        alpha_mp1 = sympy.Matrix(self.alpha[-1,:])
        beta_mp1 = sympy.Matrix(self.beta[-1,:])
        p1 = (alpha_mp1 + z*beta_mp1).T*(I-alpha_star-z*beta_star).lower_triangular_solve(vstar)
        p1 = p1[0] + v_mp1
        return p1


    def stability_function(self,stage=None,mode='exact',formula='lts',use_butcher=False):
        r"""
            The stability function of a Runge-Kutta method is
            `\\phi(z)=p(z)/q(z)`, where

            $$p(z)=\\det(I - z A + z e b^T)$$

            $$q(z)=\\det(I - z A)$$

            The function can also be computed via the formula

            $$`\\phi(z) = 1 + b^T (I-zA)^{-1} e$$

            where `e` is a column vector with all entries equal to one.

            This function constructs the numerator and denominator of the
            stability function of a Runge-Kutta method.

            For methods with rational coefficients, mode='exact' computes
            the stability function using rational arithmetic.  Alternatively,
            you can set mode='float' to force computation using floating point,
            in case the exact computation is too slow.

            For explicit methods, the denominator is simply `1` and there
            are three options for computing the numerator (this is the
            'formula' option).  These only affect
            the speed, and only matter if the computation is symbolic.
            They are:

              - 'lts': SymPy's lower_triangular_solve
              - 'det': ratio of determinants
              - 'pow': power series

            For implicit methods, only the 'det' (determinant) formula
            is supported.  If mode='float' is selected, the formula
            automatically switches to 'det'.

            The user can also select whether to compute the function based
            on Butcher or Shu-Osher coefficients by setting `use_butcher`.

            **Output**:
                - p -- Numpy poly representing the numerator
                - q -- Numpy poly representing the denominator

            **Examples**::

                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> p,q = rk4.stability_function()
                >>> print(p)
                         4          3       2
                0.04167 x + 0.1667 x + 0.5 x + 1 x + 1

                >>> dc = rk.DC(3)
                >>> dc.stability_function(mode='exact')
                (poly1d([1/3888, 1/648, 1/24, 1/6, 1/2, 1, 1], dtype=object), poly1d([1], dtype=object))

                >>> dc.stability_function(mode='float')  # doctest: +NORMALIZE_WHITESPACE
                (poly1d([2.57201646e-04, 1.54320988e-03, 4.16666667e-02, 1.66666667e-01,
                       5.00000000e-01, 1.00000000e+00, 1.00000000e+00]), poly1d([1.]))
                >>> ssp3 = rk.SSPIRK3(4)
                >>> ssp3.stability_function()
                (poly1d([-67/300 + 13*sqrt(15)/225, 1/6 - sqrt(15)/25, 9/10 - sqrt(15)/5,
                       -1 + 2*sqrt(15)/5, 1], dtype=object), poly1d([31/100 - 2*sqrt(15)/25, -7/5 + 9*sqrt(15)/25, 12/5 - 3*sqrt(15)/5,
                       -2 + 2*sqrt(15)/5, 1], dtype=object))

                >>> ssp3.stability_function(mode='float')  # doctest: +NORMALIZE_WHITESPACE
                (poly1d([4.39037781e-04, 1.17473328e-02, 1.25403331e-01, 5.49193338e-01,
                       1.00000000e+00]), poly1d([ 1.61332303e-04, -5.72599537e-03, 7.62099923e-02, -4.50806662e-01,
                       1.00000000e+00]))
                >>> ssp2 = rk.SSPIRK2(1)
                >>> ssp2.stability_function()
                (poly1d([1/2, 1], dtype=object), poly1d([-1/2, 1], dtype=object))
        """
        if mode=='float': # Override performance options
            use_butcher = True
            formula = 'det'

        if use_butcher == False and self.alpha is None:
            raise Exception('No Shu-Osher coefficients provided.')

        if formula == 'pow' and use_butcher == False:
            m = len(self)
        elif self.is_explicit():
            m = self.num_seq_dep_stages()
        else:
            m = np.inf
            formula = 'det'
            use_butcher = True

        #if formula == 'det' and use_butcher == False:
        #    raise NotImplementedError("Ratio of determinants not yet implemented for Shu-Osher coefficients.")

        if stage is None:
            stage = len(self)+1

        if use_butcher==False:
            alpha = self.alpha[0:stage,0:stage-1]
            beta  = self.beta[0:stage,0:stage-1]
            v_mp1 = 1-alpha[-1,:].sum()
        else:
            beta = np.vstack((self.A,self.b))
            alpha = beta*0

        p,q = _stability_function(alpha,beta,self.is_explicit(),m,formula=formula,mode=mode)

        if self.is_explicit():  # Trim leading coefficients that ought to be zero
            d_true = self.num_seq_dep_stages()
            d_num  = len(p.coeffs)-1
            if d_num>d_true:
                p = np.poly1d(p.coeffs[(d_num-d_true):])

        return p,q

    def E_polynomial(self):
        r"""Return the E-polynomial of the method."""

        from nodepy import stability_function
        p, q = self.stability_function()
        return stability_function.E_polynomial(p, q)


    def plot_stability_function(self,bounds=[-20,1]):
        r"""Plot the value of the stability function along the negative real axis.

            **Example**::

                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> rk4.plot_stability_function()

        """
        import matplotlib.pyplot as plt
        p,q=self.stability_function()
        xx=np.arange(bounds[0], bounds[1], 0.01)
        yy=p(xx)/q(xx)
        fig, = plt.plot(xx,yy)
        plt.draw()



    def plot_stability_region(self,N=200,color='r',filled=True,bounds=None,
                              plotroots=False,alpha=1.,scalefac=1.,
                              to_file=False, longtitle=True,fignum=None):
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

            **Example**::
                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> rk4.plot_stability_region() #doctest: +ELLIPSIS
                <Figure size...
        """
        import nodepy.stability_function as stability_function
        import matplotlib.pyplot as plt

        p,q=self.__num__().stability_function(mode='float')

        fig = stability_function.plot_stability_region(p,q,N,color,filled,bounds,
                    plotroots,alpha,scalefac,fignum)

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

    def plot_order_star(self,N=200,bounds=[-5,5,-5,5],plotroots=False,
                        color=('w','b'),filled=True,fignum=None):
        r""" The order star of a Runge-Kutta method is the set

            $$ \\{ z \\in C : | \\phi(z)/\\exp(z) | \\le 1 \\} $$

            where `\phi(z)` is the stability function of the method.

            **Input**: (all optional)
                - N       -- Number of gridpoints to use in each direction
                - bounds  -- limits of plotting region
                - color   -- color to use for this plot
                - filled  -- if true, order star is filled in (solid); otherwise it is outlined

            **Example**::
                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> rk4.plot_order_star() # doctest: +ELLIPSIS
                <Figure size...
        """
        import nodepy.stability_function as stability_function
        import matplotlib.pyplot as plt

        p,q=self.__num__().stability_function(mode='float')

        fig = stability_function.plot_order_star(p,q,N,bounds,plotroots,color,fignum)
        plt.title('Order star for '+self.name)
        return fig

    #============================================================
    # Nonlinear Stability
    #============================================================
    def circle_contractivity_radius(self,acc=1.e-13,rmax=1000):
        r"""
            Returns the radius of circle contractivity
            of a Runge-Kutta method.

            **Example**::

                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> rk4.circle_contractivity_radius() #doctest: +ELLIPSIS
                1.000...

        """
        from nodepy.utils import bisect

        tol=1.e-14
        r=bisect(0,rmax,acc,tol,self.__num__()._is_circle_contractive)
        return r

    def absolute_monotonicity_radius(self,acc=1.e-10,rmax=200,
                    tol=3.e-16):
        r"""
            Returns the radius of absolute monotonicity
            (also referred to as the radius of contractivity or
            the strong stability preserving coefficient
            of a Runge-Kutta method.
        """
        from nodepy.utils import bisect

        r=bisect(0,rmax,acc,tol,self._is_absolutely_monotonic)
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
                if p(zeroes[i])/q(zeroes[i])<p(xmax)/q(xmax) and zeroes[i]>xmax:
                    xmax=zeroes[i]
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


    def _is_circle_contractive(self,r,tol):
        r""" Returns 1 if the Runge-Kutta method has radius of circle
            contractivity at least `r`.

            See :cite:`dekker1984`.

        """
        B=np.diag(self.b)
        M=np.dot(B,self.A)+np.dot(self.A.T,B)-np.outer(self.b,self.b)
        X=M+B/r
        v,d=np.linalg.eig(X)
        if v.min()>-tol:
            return 1
        else:
            return 0


    def _is_absolutely_monotonic(self,r,tol):
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

            See :cite:`kraaijevanger1991`.
        """
        s=len(self)
        K=np.vstack([self.A.astype(float),self.b.astype(float)])
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
    def optimal_shu_osher_form(self):
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

            **Example**::

                >>> from nodepy import rk
                >>> rk2 = rk.loadRKM('MTE22')
                >>> rk2.optimal_shu_osher_form()
                (array([[0, 0, 0],
                       [1.00000000000000, 0, 0],
                       [0.625000000060027, 0.374999999939973, 0]], dtype=object), array([[0, 0, 0],
                       [0.666666666666667, 0, 0],
                       [4.00177668780088e-11, 0.750000000000000, 0]], dtype=object))

            See :cite:`higueras2005`.

        """
        m=len(self)
        r = self.absolute_monotonicity_radius()
        v, alpha = self.canonical_shu_osher_form(r)
        beta = alpha / r
        if self.is_explicit():
            for i in range(1,len(self)+1):
                alpha[i,0]=1.-np.sum(alpha[i,1:])
        return alpha, beta

    def canonical_shu_osher_form(self,r):
        r""" Returns d,P where P is the matrix `P=r(I+rK)^{-1}K`
             and d is the vector `d=(I+rK)^{-1}e=(I-P)e`.

             Note that this can be computed for any value of `r`,
             including values for which `d, P` may have negative
             entries.
        """
        s=len(self)
        K=np.vstack([self.A,self.b])
        K=np.hstack([K,np.zeros([s+1,1])])
        I=snp.eye(s+1)
        P=r*snp.solve(I+r*K,K)
        d=(I-P).sum(1)
        return d,P

    #==========================================
    # Optimal (downwind) perturbations
    #==========================================
    def lp_perturb(self,r,tol=None):
        r"""Find a perturbation via linear programming.

        Use linear programming to determine if there exists
        a perturbation of this method with radius of absolute
        monotonicity at least `r`.

        The linear program to be solved is

        .. math::

            \begin{align}
                (I-2\alpha^{down}_r)\alpha_r + \alpha^{down}_r & = (\alpha^{up}_r ) \ge 0 \\
                (I-2\alpha^{down}_r)v_r & = \gamma_r \ge 0.
            \end{align}

        This function requires cvxpy.
        """
        import cvxpy as cvx

        if not self.is_explicit():
            # We could find explicit perturbations for implicit methods,
            # but is that useful?
            raise Exception("LP perturbation algorithm works only for explicit methods.")

        s = len(self)
        I = np.eye(s+1)

        v_r, alpha_r = self.canonical_shu_osher_form(r)

        alpha_down = cvx.Variable(s+1,s+1)
        objective = cvx.Minimize(sum(alpha_down))
        constraints = [(I-2*alpha_down)*alpha_r + alpha_down >= 0,
                       (I-2*alpha_down)*v_r >= 0,
                       alpha_down >= 0]

        if self.is_explicit():
            # Constrain perturbation to be explicit
            for i in range(alpha_down.shape.rows):
                for j in range(i,alpha_down.shape.cols):
                    constraints.append(alpha_down[i,j] == 0)

        problem = cvx.Problem(objective, constraints)
        status = problem.solve()
        return (status == 0)


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


    def resplit(self,r,tol=1.e-15,max_iter=5):
        s = len(self)
        I = np.eye(s+1)
        gamma, alpha_up = self.canonical_shu_osher_form(r)
        alpha_down = 0*alpha_up

        for i in range(max_iter):
            aup, aum = sign_split(alpha_up)
            adp, adm = sign_split(alpha_down)

            G = np.linalg.inv(I + 2*(aum.astype(float) + adm.astype(float)))
            alpha_up = np.dot(G,aup+adm)
            alpha_down = np.dot(G,aum+adp)
            gamma = np.dot(G,gamma)

            if all(self.A[0,:] == 0):
                gamma, alpha_up, alpha_down = redistribute_gamma(gamma, alpha_up, alpha_down)

            if alpha_up.min()>=-tol and gamma.min()>=-tol and alpha_down.min()>=-tol:
                break

        return gamma, alpha_up, alpha_down


    def is_splittable(self,r,tol=1.e-15):
        d,alpha,alphatilde=self.resplit(r,tol=tol)
        if alpha.min()>=-tol and d.min()>=-tol and alphatilde.min()>=-tol:
            return True
        else:
            return False

    def optimal_perturbed_splitting(self,acc=1.e-12,rmax=50.01,tol=1.e-13,algorithm='split'):
        r"""
            Return the optimal downwind splitting of the method
            along with the optimal downwind SSP coefficient.

            The default algorithm (split with iteration) is not
            provably correct.  The LP algorithm is.  See the paper
            (Higueras & Ketcheson) for more details.

            **Example**::

                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> r, d, alpha, alphatilde = rk4.optimal_perturbed_splitting(algorithm='split')
                >>> print(r) # doctest: +ELLIPSIS
                0.68501606...
        """
        from nodepy.utils import bisect
        try:
            import cvxpy as cvx
        except:
            algorithm = 'split'

        if algorithm == 'LP':
            r=bisect(0,rmax,acc,tol,self.lp_perturb)
        elif algorithm == 'split':
            r=bisect(0,rmax,acc,tol,self.is_splittable)

        d,alpha,alphatilde=self.resplit(r,tol=tol)
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
        G=I2 + np.dot(np.kron(self.b[:,np.newaxis],I2).T,
                      np.dot(Z,np.dot(Xinv,e)))

        return G, Xinv


    def is_explicit(self):
        return False

    def is_zero_stable(self):
        return True

    def is_FSAL(self):
        """True if method is "First Same As Last"."""
        if np.all(self.A[-1,:]==self.b): return True
        else: return False

def sign_split(M):
    """Given a matrix M, return two matrices.  The first contains the
       positive entries of M; the second contains the negative entries of M,
       multiplied by -1.
    """
    M_plus  =  M*(M>0).astype(int)
    M_minus = -M*(M<0).astype(int)
    return M_plus, M_minus

def redistribute_gamma(gamma, alpha_up, alpha_down):
    alpha_up[1:,0] += gamma[1:]/2.
    alpha_down[1:,0] += gamma[1:]/2.
    gamma[1:] = 0.

    return gamma, alpha_up, alpha_down


#=====================================================
class ExplicitRungeKuttaMethod(RungeKuttaMethod):
#=====================================================
    r"""
        Class for explicit Runge-Kutta methods.  Mostly identical
        to RungeKuttaMethod, but also includes time-stepping and
        a few other functions.
    """
    def __step__(self,f,t,u,dt,x=None,estimate_error=False,use_butcher=False):
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
        if self.alpha is None:
            use_butcher = True

        m=len(self)
        u_old = u       # Initial value
        y = [np.zeros_like(np.atleast_1d(u)) for i in range(m+1)]
        fy = [np.zeros_like(np.atleast_1d(u)) for i in range(m)]

        # First stage
        y[0][:]=u_old
        if x is not None: fy[0][:]=f(t,y[0],x)
        else: fy[0][:]=f(t,y[0])

        if use_butcher:                 # Use Butcher coefficients
            for i in range(1,m):        # Compute stage i
                y[i][:] = u_old
                for j in range(i):
                    y[i] += self.A[i,j]*dt*fy[j]
                    if x is not None: fy[i][:] = f(t+self.c[i]*dt,y[i],x)
                    else: fy[i][:] = f(t+self.c[i]*dt,y[i])
            u_new=u_old+dt*sum([self.b[j]*fy[j] for j in range(m)])

        else:             # Use Shu-Osher coefficients
            v = 1 - self.alpha.sum(1)
            for i in range(1,m+1):
                y[i] = v[i]*u_old
                for j in range(i):
                    y[i] += self.alpha[i,j]*y[j] + dt*self.beta[i,j]*fy[j]
                if i<m:
                    if x is not None: fy[i][:] = f(t+self.c[i]*dt,y[i],x)
                    else: fy[i][:] = f(t+self.c[i]*dt,y[i])
            u_new = y[m]

        return u_new



    def imaginary_stability_interval(self,mode='exact',eps=1.e-14):
        r"""
            Length of imaginary axis half-interval contained in the
            method's region of absolute stability.

            **Examples**::

                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> rk4.imaginary_stability_interval() # doctest: +ELLIPSIS
                2.8284271247461...
        """
        import nodepy.stability_function as stability_function
        p,q=self.stability_function(mode=mode)
        return stability_function.imaginary_stability_interval(p,q,eps=eps)

    def real_stability_interval(self,mode='exact',eps=1.e-14):
        r"""
            Length of negative real axis interval contained in the
            method's region of absolute stability.

            **Examples**::

                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> I = rk4.real_stability_interval()
                >>> print("{:.10f}".format(I))
                2.7852935634
        """
        import nodepy.stability_function as stability_function
        p,q=self.stability_function(mode=mode)
        return stability_function.real_stability_interval(p,q,eps=eps)


    def linear_absolute_monotonicity_radius(self,acc=1.e-10,rmax=50,
                                            tol=3.e-16):
        """
            Returns the radius of absolute monotonicity
            of the stability function of a Runge-Kutta method.

            TODO: implement this functionality for implicit methods.
        """
        from nodepy.utils import bisect
        p,q=self.stability_function()
        if q.order!=0 or q[0]!=1:
            raise NotImplementedError(
                    'Not yet implemented for rational functions')
        else:
            r=bisect(0,rmax,acc,tol,is_absolutely_monotonic_poly,p=p)
        return r

    def is_explicit(self):
        return True

    def work_per_step(self):
        "Number of function evaluations required for one step."
        if self.is_FSAL(): return len(self)-1
        else: return len(self)

    def num_seq_dep_stages(self):
        r"""Number of sequentially dependent stages.

        Number of sequential function evaluations that must be made.

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
                17
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

    def internal_stability_polynomials(self,stage=None,mode='exact',formula='lts',use_butcher=False):
        r"""
            The internal stability polynomials of a Runge-Kutta method
            depend on the implementation and must therefore be constructed
            base on the Shu-Osher form used for the implementation.
            By default the Shu-Osher coefficients are used.  The
            Butcher coefficients are used if use_butcher=True or
            if Shu-Osher coefficients are not defined.

            The formula for the polynomials is:
            Modified Shu-Osher form: `(alphastarmp1+z betastarmp1)(I-alphastar-z betastar)^{-1}`
            Butcher array: `z b^T(I-zA)^{-1}`

            Note that in the first stage no perturbation is introduced because
            for an explicit method the first stage is equal to the solution at
            the current time level. Therefore, the first internal polynomial is
            set to zero.

            For symbolic computation,
            this routine has been significantly modified for efficiency
            relative to particular classes of methods.  Two formulas are
            implemented, one based on SymPy's Matrix.lower_triangular_solve()
            and the other using a power series for the inverse.  Different
            choices of these two are more efficient for different classes of
            methods (this only matters for methods with very many stages).

            **Options**
                - use_butcher

            **Output**:
                - numpy array of internal stability polynomials

            **Examples**::

                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> theta = rk4.internal_stability_polynomials()
                >>> for p in theta:
                ...     print(p)
                         3          2
                0.08333 x + 0.1667 x + 0.3333 x
                        2
                0.1667 x + 0.3333 x
                <BLANKLINE>
                0.1667 x
        """
        if stage is None:
            stage = len(self)+1

        if formula == 'pow' and use_butcher == False:
            m = len(self)
        elif self.is_explicit():
            m = self.num_seq_dep_stages()
        else:
            m = len(self)
        if use_butcher==False:
            alpha = self.alpha[0:stage,0:stage-1]
            beta  = self.beta[0:stage,0:stage-1]
        else:
            beta = np.vstack((self.A,self.b))
            alpha = beta*0

        explicit = self.is_explicit()
        theta = _internal_stability_polynomials(alpha,beta,explicit,m,formula=formula,mode=mode)

        return theta


    def internal_stability_plot(self,bounds=None,N=200,use_butcher=False,formula='lts',levels=[1,100,500,1000,1500,10000]):
        r"""Plot internal stability regions.

            Plots the $\epsilon$-internal-stability region contours.

            By default the Shu-Osher coefficients are used.  The
            Butcher coefficients are used if use_butcher=True or
            if Shu-Osher coefficients are not defined.

            **Examples**::

                >>> from nodepy import rk
                >>> rk4 = rk.loadRKM('RK44')
                >>> rk4.internal_stability_plot()
        """
        import nodepy.stability_function as stability_function
        import matplotlib.pyplot as plt
        from nodepy.utils import find_plot_bounds
        from matplotlib.colors import LogNorm

        p,q = self.stability_function(use_butcher=use_butcher,formula=formula)
        # Convert coefficients to floats for speed
        if p.coeffs.dtype=='object':
            p = np.poly1d([float(c) for c in p.coeffs])
            q = np.poly1d([float(c) for c in q.coeffs])

        stable = lambda z : np.abs(p(z)/q(z))<=1.0
        bounds = find_plot_bounds(stable,guess=(-10,1,-5,5))

        theta = self.internal_stability_polynomials(use_butcher=use_butcher,formula=formula)

        x=np.linspace(bounds[0],bounds[1],N)
        y=np.linspace(bounds[2],bounds[3],N)
        X=np.tile(x,(N,1))
        Y=np.tile(y[:,np.newaxis],(1,N))
        Z=X + Y * 1j

        th_vals = np.zeros((len(theta), N, N), dtype=np.complex64)

        for j in range(len(theta)):
            thetaj = np.poly1d([float(c) for c in theta[j].coeffs])
            th_vals[j,...] = thetaj(Z)
        th_max = np.max(np.abs(th_vals),axis=0)

        fig = plt.figure()
        CS = plt.contour(X,Y,th_max,colors='k',levels=levels)
        plt.clabel(CS, fmt='%d', colors='k')#,manual=True)

        p,q=self.__num__().stability_function(mode='float')
        stability_function.plot_stability_region(p,q,N,color='k',filled=False,bounds=bounds,
                fignum=fig.number)


    def maximum_internal_amplification(self,N=200,use_butcher=False,formula='lts'):
        r"""The maximum amount by which any stage error is amplified,
            assuming the step size is taken so that the method is absolutely
            stable:

            `\max_{z \in S,j} |\theta_j(z)|`

            where `S = \{z \in C : |R(z)|\le 1.`

            Here `R(z)` is the stability function and `\theta_j(z)`
            are the internal stability functions.

            By default the Shu-Osher coefficients are used.  The
            Butcher coefficients are used if use_butcher=True or
            if Shu-Osher coefficients are not defined.

            **Examples**::

                >>> from nodepy import rk
                >>> ssp2 = rk.SSPRK2(6)
                >>> ssp2.maximum_internal_amplification() # doctest: +ELLIPSIS
                (1.097405..., 0.83333333...)
                >>> ssp2.maximum_internal_amplification(use_butcher=True) # doctest: +ELLIPSIS
                (2.037051..., 0.0)
        """
        from nodepy.utils import find_plot_bounds

        if (self.alpha is None or self.beta is None): use_butcher = True

        p,q = self.stability_function(use_butcher=use_butcher,formula=formula)
        # Convert coefficients to floats for speed
        if p.coeffs.dtype=='object':
            p = np.poly1d([float(c) for c in p.coeffs])
            q = np.poly1d([float(c) for c in q.coeffs])

        stable = lambda z : np.abs(p(z)/q(z))<=1.0
        bounds = find_plot_bounds(stable,guess=(-10,1,-5,5))

        # Evaluate the stability function over a grid
        x=np.linspace(bounds[0],bounds[1],N)
        y=np.linspace(bounds[2],bounds[3],N)
        X=np.tile(x,(N,1))
        Y=np.tile(y[:,np.newaxis],(1,N))
        Z=X+Y*1j
        R=np.abs(p(Z)/q(Z))

        # Select just the absolutely stable points
        ij_stable = np.where(R<=1.)
        Z_stable = Z[ij_stable]

        # Evaluate the internal stability polynomials over the stable region
        theta = self.internal_stability_polynomials(use_butcher=use_butcher,formula=formula)
        maxamp = 0.
        maxamp_origin = 0.
        for thetaj in theta:
            thetaj = np.poly1d([float(c) for c in thetaj.coeffs])
            maxamp = max(maxamp, np.max(np.abs(thetaj(Z_stable))))
            maxamp_origin = max(maxamp_origin, np.abs(thetaj(0.)))

        return maxamp, maxamp_origin

#=====================================================
#End of ExplicitRungeKuttaMethod class
#=====================================================


class ContinuousRungeKuttaMethod(RungeKuttaMethod):

    def __init__(self,A=None,b=None,alpha=None,beta=None,
                 b_dense=None,
                 name='Continuous Runge-Kutta Method',shortname='CRKM',
                 description='',mode='exact',order=None):

        super(ContinuousRungeKuttaMethod,self).__init__(A,b,alpha,beta,name,
                                                   shortname,description)
        self.b_dense = b_dense

class ContinuousExplicitRungeKuttaMethod(ContinuousRungeKuttaMethod,ExplicitRungeKuttaMethod):
    def __step__(self,f,t,u,dt,thetas,x=None,estimate_error=False,use_butcher=False):
        """
            Take a time step on the ODE u'=f(t,u), providing output at
            times requested.

            **Input**:
                - f  -- function being integrated
                - t  -- array of previous solution times
                - u  -- array of previous solution steps (u[i] is the solution at time t[i])
                - dt -- length of time step to take
                - thetas -- list of relative times at which to provide output

            **Output**:
                - unew -- approximate solution at time t[-1]+dt
                - u_theta -- list of dense output values

            The implementation here is wasteful in terms of storage.
        """
        if self.alpha is None:
            use_butcher = True

        m=len(self)
        u_old = u.copy()		# Initial value
        y = [np.zeros_like(np.atleast_1d(u)) for i in range(m+1)]
        fy = [np.zeros_like(np.atleast_1d(u)) for i in range(m)]

        # First stage
        y[0][:]=u_old
        if x is not None: fy[0][:]=f(t,y[0],x)
        else: fy[0][:]=f(t,y[0])

        if use_butcher:                 # Use Butcher coefficients
            for i in range(1,m):        # Compute stage i
                y[i][:] = u_old
                for j in range(i):
                    y[i] += self.A[i,j]*dt*fy[j]
                    if x is not None: fy[i][:] = f(t+self.c[i]*dt,y[i],x)
                    else: fy[i][:] = f(t+self.c[i]*dt,y[i])
            u_new=u_old+dt*sum([self.b[j]*fy[j] for j in range(m)])

        else:             # Use Shu-Osher coefficients
            v = 1 - self.alpha.sum(1)
            for i in range(1,m+1):
                y[i] = v[i]*u_old
                for j in range(i):
                    y[i] += self.alpha[i,j]*y[j] + dt*self.beta[i,j]*fy[j]
                if i<m:
                    if x is not None: fy[i][:] = f(t+self.c[i]*dt,y[i],x)
                    else: fy[i][:] = f(t+self.c[i]*dt,y[i])
            u_new = y[m]

        if thetas:
            deg = self.b_dense.shape[1] # b_j polynomial degree
            u_theta = [np.zeros_like(np.atleast_1d(u)) for i in range(len(thetas))]
            for i, theta in enumerate(thetas):
                u_theta[i]= u_old.copy()
                for j in range(m):
                    bj = 0.
                    for d in range(1,deg):
                        bj += self.b_dense[j,d] * theta**d
                    u_theta[i] += dt*bj*fy[j]
        else:
            u_theta = None

        return u_new, u_theta



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

        The class also admits Shu-Osher representations:

        \\begin{align*}
        y_i = & v_i u^{n} + \\sum_{j=1}^s \\alpha_{ij} y_j + \\Delta t \\sum_{j=1}^{s} + \\beta_{ij} f(y_j)) & (1\\le j \\le s+1) \\\\
        u^{n+1} = & y_{s+1}
        \\hat{u}^{n+1} = & \\hat{v}_{s+1} u^{n} + \\sum_{j=1}^s \\hat{\\alpha}_{s+1,j} + \\Delta t \\sum_{j=1}^{s} \\hat{\\beta}_{s+1,j} f(y_j).
        \\end{align*}

        In NodePy, if *rkp* is a Runge-Kutta pair, the principal (usually
        higher-order) method is the one used if accuracy or stability properties
        are queried.  Properties of the embedded (usually lower-order) method can
        be accessed via *rkp.embedded_method*.

        When solving an IVP with an embedded pair, one can specify a desired
        error tolerance.  The step size will be adjusted automatically
        to achieve approximately this tolerance.
    """
    def __init__(self,A=None,b=None,bhat=None,alpha=None,beta=None,alphahat=None,betahat=None,
            name='Runge-Kutta Pair',shortname='RKM',description='',order=(None,None)):
        r"""
            In addition to the ordinary Runge-Kutta initialization,
            here the embedded coefficients `\hat{b}_j` are set as well.
        """
        super(ExplicitRungeKuttaPair,self).__init__(
                        A,b,alpha,beta,name,shortname,description,order=order[0])
        if bhat is None:
            Ahat,bhat=shu_osher_to_butcher(alphahat,betahat)
        if bhat.shape != self.b.shape:
            raise Exception("Dimensions of embedded method don't agree with those of principal method")
        self.bhat     = bhat
        self.alphahat = alphahat
        self.betahat  = betahat
        self.mtype = 'Explicit embedded Runge-Kutta pair'
        self._p_hat = order[1]

    @property
    def main_method(self):
        """Return the main method of the pair (usually the higher-order one)."""
        if self.alpha is None:
            return ExplicitRungeKuttaMethod(self.A,self.b,order=self._p)
        else:
            return ExplicitRungeKuttaMethod(alpha=self.alpha,beta=self.beta,order=self._p)

    @property
    def embedded_method(self):
        """Always recompute the embedded method on the fly.  This may be inefficient."""
        if self.alphahat is None:
            return ExplicitRungeKuttaMethod(self.A,self.bhat,order=self._p_hat)
        else:
            return ExplicitRungeKuttaMethod(alpha=self.alphahat,beta=self.betahat,order=self._p_hat)

    def __num__(self):
        """
        Returns a copy of the method but with floating-point coefficients.
        This is useful whenever we need to operate numerically without
        worrying about the representation of the method.
        """
        numself = super(ExplicitRungeKuttaPair,self).__num__()
        if self.A.dtype==object:
            numself.bhat=np.array(self.bhat,dtype=np.float64)
        if self.alphahat is not None:
            numself.alphahat=np.array(self.alphahat,dtype=np.float64)
            numself.betahat=np.array(self.betahat,dtype=np.float64)
        return numself

    def __str__(self):
        """
        Pretty-prints the Butcher array in the form:
          |
        c | A
        ________
          | b
          | bhat
        """
        s = super(ExplicitRungeKuttaPair,self).__str__()
        from nodepy.utils import array2strings

        c    = array2strings(self.c)
        A    = array2strings(self.A)
        b    = array2strings(self.b)
        bhat = array2strings(self.bhat)
        lenmax, colmax = _get_column_widths([A,b,c])
        s+= '\n'+' '*(colmax+1)+'|'
        for j in range(len(self)):
            s+=bhat[j].ljust(colmax+1)
        return s.rstrip()


    def __step__(self,f,t,u,dt,x=None,estimate_error=False,use_butcher=False):
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
        if self.alphahat is None:
            use_butcher = True

        m=len(self)
        u_old = u       # Initial value
        y = [np.zeros_like(np.atleast_1d(u)) for i in range(m+1)]
        fy = [np.zeros_like(np.atleast_1d(u)) for i in range(m)]

        # First stage
        y[0][:]=u_old
        if x is not None: fy[0][:]=f(t,y[0],x)
        else: fy[0][:]=f(t,y[0])

        if use_butcher:                 # Use Butcher coefficients
            for i in range(1,m):        # Compute stage i
                y[i][:] = u_old
                for j in range(i):
                    y[i] += self.A[i,j]*dt*fy[j]
                    if x is not None: fy[i][:] = f(t[-1]+self.c[i]*dt,y[i],x)
                    else: fy[i][:] = f(t+self.c[i]*dt,y[i])
            u_new=u_old+dt*sum([self.b[j]*fy[j] for j in range(m)])
            if estimate_error:
                u_hat=u+dt*sum([self.bhat[j]*fy[j] for j in range(m)])

        else:             # Use Shu-Osher coefficients
            v = 1 - self.alpha.sum(1)
            for i in range(1,m+1):
                y[i] = v[i]*u_old
                for j in range(i):
                    y[i] += self.alpha[i,j]*y[j] + dt*self.beta[i,j]*fy[j]
                if i<m:
                    if x is not None: fy[i][:] = f(t+self.c[i]*dt,y[i],x)
                    else: fy[i][:] = f(t+self.c[i]*dt,y[i])
            u_new = y[m]

            if estimate_error:
                u_hat = np.zeros_like(np.atleast_1d(u))
                #if dt<1e-10:
                    #print("Warning: very small step size: {} {}".format(dt, t[-1]))
                u_hat = (1-np.sum(self.alphahat[-1,:]))*u_old
                for j in range(m):
                    u_hat += self.alphahat[-1,j]*y[j] + dt*self.betahat[-1,j]*fy[j]

        if estimate_error:
            return u_new, np.max(np.abs(u_new-u_hat))
        else:
            return u_new

    def error_metrics(self,q=None,p=None):
        r"""Return full set of error metrics for an embedded RK pair.
            See :cite:`kennedy2000` p. 181

            **Example**::

                >>> from nodepy import rk
                >>> bs5 = rk.loadRKM('BS5')
                >>> bs5.error_metrics()
                main method has order 5
                embedded method has order 4
                (43*sqrt(83011)/558835200, 43/3386880, sqrt(29695176594765489880490334265)/810521680634265600, 1451/15966720, sqrt(870269901055795)/277898765760, 10147/131580855, sqrt(51577359825120524319571156056057595)/219308015066060340, 26201089/40912704, sqrt(5250600078722255566247933273951710555)/2193080150660603400, 305343067/400035328, 482048/414219, 5987277*sqrt(72241974756542598745)/243675572295622600, 5987277/36366848)
        """
        if q is None:
            q=self.order()
            print('main method has order {}'.format(q))
        if p is None:
            p=self.embedded_method.order()
            print('embedded method has order {}'.format(p))

        tau_1 = self.error_coeffs(q+1)
        tau_2 = self.error_coeffs(q+2)

        A_qp1 = snp.norm(tau_1)
        A_qp2 = snp.norm(tau_2)
        A_qp1_max=max([abs(tau) for tau in tau_1])
        A_qp2_max=max([abs(tau) for tau in tau_2])

        D=max(np.max(np.abs(self.A)),
                np.max(np.abs(self.b)),np.max(np.abs(self.c)))
        tau_pp2=self.error_coeffs(p+2)

        tau_pp1_hat=self.embedded_method.error_coeffs(p+1)
        tau_pp2_hat=self.embedded_method.error_coeffs(p+2)

        A_pp1_hat = snp.norm(tau_pp1_hat)
        A_pp2_hat = snp.norm(tau_pp2_hat)
        A_pp2 = snp.norm(tau_pp2)

        A_pp1_hat_max=max([abs(tau) for tau in tau_pp1_hat])

        A_pp2_max=    max([abs(tau) for tau in tau_pp2])
        A_pp2_hat_max=max([abs(tau) for tau in tau_pp2_hat])

        B_pp2=    A_pp2_hat    /A_pp1_hat
        B_pp2_max=A_pp2_hat_max/A_pp1_hat_max

        tau2diff=np.array(tau_pp2_hat)-np.array(tau_pp2)
        C_pp2 = snp.norm(tau2diff) / A_pp1_hat
        C_pp2_max=max([abs(tau) for tau in tau2diff])/A_pp1_hat_max

        D=max(np.max(self.A),np.max(self.b),np.max(self.bhat),np.max(self.c))

        E_pp2=    A_pp2    /A_pp1_hat
        E_pp2_max=A_pp2_max/A_pp1_hat_max

        return A_qp1, A_qp1_max, A_qp2, A_qp2_max, A_pp1_hat, A_pp1_hat_max, B_pp2, B_pp2_max, C_pp2, C_pp2_max, D, E_pp2, E_pp2_max


    def is_FSAL(self):
        if np.all(self.A[-1,:]==self.b): return True
        elif np.all(self.A[-1,:]==self.bhat): return True
        else: return False

    def plot_stability_region(self,N=200,color='r',filled=True,bounds=None,
                              plotroots=False,alpha=1.,scalefac=1.,
                              to_file=False,longtitle=True,fignum=None):
        r"""Plot the absolute stability region of an RK pair. By default,
            the region of the main method is filled in red and the region of
            the embedded method is outlined in black.

            **Example**::

                >>> from nodepy import rk
                >>> bs5 = rk.loadRKM('BS5')
                >>> bs5.plot_stability_region() # doctest: +ELLIPSIS
                <Figure size...

        """
        import nodepy.stability_function as stability_function
        import matplotlib.pyplot as plt

        p,q=self.__num__().stability_function(mode='float')

        fig = stability_function.plot_stability_region(p,q,N,color,filled,
                        bounds,plotroots,alpha,scalefac,fignum)

        p,q = self.embedded_method.__num__().stability_function(mode='float')
        stability_function.plot_stability_region(p,q,N,color='k',filled=False,bounds=bounds,
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

    def _plot_controller_stability_common(self, beta1, beta2, beta3,
                                          N=200, color='r', filled=True, bounds=None,
                                          plotroots=False, alpha=1., scalefac=1., longtitle=True,
                                          to_file_region=False,  fignum_region=None, fignum_controller=None):
        """Common functionality for all `plot_XXX_controller_stability` methods.
        """
        import nodepy.stability_function as stability_function
        import matplotlib.pyplot as plt
        import numpy as np

        num, den = self.__num__().stability_function(mode='float')
        numhat, denhat = self.embedded_method.__num__().stability_function(mode='float')

        if (den.order > 0) or (denhat.order > 0):
            print('Stepsize controller stability plots are only implemented for explicit schemes.')
            return

        # Plot the stability function
        fig_region = self.plot_stability_region(N=N, color=color, filled=filled, bounds=bounds,
                                                plotroots=plotroots, alpha=alpha, scalefac=scalefac,
                                                to_file=to_file_region, longtitle=longtitle, fignum=fignum_region)

        # Get the boundary of the stability region
        ax_region = fig_region.get_axes()[0]
        c = ax_region.collections[0]
        v = np.vstack([p.vertices for p in c.get_paths()])

        xx = v[:,0]
        yy = v[:,1]
        zz = (xx + 1j * yy) * scalefac
        angle = np.arctan2(yy, xx)
        # Use only one quadrant in the left half of the complex plane
        filter_idx = np.argwhere(angle >= np.pi/2).flatten()
        zz = zz[filter_idx]
        angle = angle[filter_idx]

        R = num
        Rprime = R.deriv()
        E = numhat - num
        Eprime = E.deriv()
        zRprime_R = np.real(Rprime(zz) * zz / R(zz))
        zEprime_E = np.real(Eprime(zz) * zz / E(zz))

        # Plot PID controller stability function
        k = min(self.main_method.p, self.embedded_method.p) + 1.0

        C = np.zeros((np.size(angle), 6, 6))
        C[:,0,0] = 1
        C[:,0,1] = zRprime_R
        C[:,1,0] = -beta1/k
        C[:,1,1] = 1 - beta1/k * zEprime_E
        C[:,1,2] = -beta2/k
        C[:,1,3] = -beta2/k * zEprime_E
        C[:,1,4] = -beta3/k
        C[:,1,5] = -beta3/k * zEprime_E
        C[:,2,0] = 1
        C[:,3,1] = 1
        C[:,4,2] = 1
        C[:,5,3] = 1
        rho = np.abs(np.linalg.eigvals(C)).max(axis=1)

        fig_controller = plt.figure(fignum_controller)
        plt.plot(angle, rho)
        plt.plot(angle, np.ones(np.size(rho)), '--k', linewidth=2)
        ax_controller = fig_controller.get_axes()[0]
        ax_controller.set_xticks([np.pi/2, np.pi*5/8, np.pi*3/4, np.pi*7/8, np.pi])
        ax_controller.set_xticklabels(['$\pi/2$', '$5\pi/8$', '$3\pi/4$', '$7\pi/8$', '$\pi$'])

        asp = np.diff(ax_controller.get_xlim())[0] / np.diff(ax_controller.get_ylim())[0]
        ax_controller.set_aspect(asp)

        return fig_region, fig_controller

    def plot_I_controller_stability(self, beta1=1.,
                                    N=200, color='r', filled=True, bounds=None,
                                    plotroots=False, alpha=1., scalefac=1., longtitle=True,
                                    to_file_region=False, to_file_controller=False,
                                    fignum_region=None, fignum_controller=None):
        r"""Plot the absolute stability region and the function characterizing
            stepsize control stability for an I controller of an RK pair,
            cf. :cite:`hairerODEs2`. The I controller is of the form

            .. math::

                \begin{equation}
                    h_{n+1} = \left(\frac{\mathrm{TOL}}{\mathrm{err}_{n}})^{\beta_1/k} h_{n},
                \end{equation}

            where `h` is the stepsize, `TOL` the tolerance, and `err = O(h^k)`
            the error estimate.

            By default, the region of the main method is filled in red and the
            region of the embedded method is outlined in black.

            **Example**::

                >>> from nodepy import rk
                >>> bs5 = rk.loadRKM('BS5')
                >>> bs5.plot_I_controller_stability() # doctest: +ELLIPSIS
                (<Figure size...
        """
        import matplotlib.pyplot as plt

        fig_region, fig_controller = self._plot_controller_stability_common(
            beta1=beta1, beta2=0.0, beta3=0.0,
            N=N, color=color, filled=filled, bounds=bounds, plotroots=plotroots,
            alpha=alpha, scalefac=scalefac, longtitle=longtitle, to_file_region=to_file_region,
            fignum_region=fignum_region, fignum_controller=fignum_controller)

        ax_controller = fig_controller.get_axes()[0]
        if longtitle:
            ax_controller.set_title("I Controller Stability Function for %s\n($\\beta_1 = %.2f$)" % (self.name, beta1))
        else:
            ax_controller.set_title("I Controller for %s" % (self.shortname))

        # Save or draw the plot.
        fig_controller.tight_layout()
        if to_file_controller:
            fig_controller.savefig(to_file_controller, transparent=True, bbox_inches='tight', pad_inches=0.3)
        else:
            fig_controller.canvas.draw_idle()

        return fig_region, fig_controller


    def plot_PI_controller_stability(self, beta1=2./3., beta2=-1./3.,
                                     N=200, color='r', filled=True, bounds=None,
                                     plotroots=False, alpha=1., scalefac=1.,longtitle=True,
                                     to_file_region=False, to_file_controller=False,
                                     fignum_region=None, fignum_controller=None):
        r"""Plot the absolute stability region and the function characterizing
            stepsize control stability for a PI controller of an RK pair,
            cf. :cite:`hairerODEs2`. The PI controller is of the form

            .. math::

                \begin{equation}
                    h_{n+1} = \left(\frac{\mathrm{TOL}}{\mathrm{err}_{n}})^{\beta_1/k} \left(\frac{\mathrm{TOL}}{\mathrm{err}_{n-1}})^{\beta_2/k} h_{n},
                \end{equation}

            where `h` is the stepsize, `TOL` the tolerance, and `err = O(h^k)`
            the error estimate.

            By default, the region of the main method is filled in red and the
            region of the embedded method is outlined in black.

            **Example**::

                >>> from nodepy import rk
                >>> bs5 = rk.loadRKM('BS5')
                >>> bs5.plot_PI_controller_stability() # doctest: +ELLIPSIS
                (<Figure size...
        """
        import matplotlib.pyplot as plt

        fig_region, fig_controller = self._plot_controller_stability_common(
            beta1=beta1, beta2=beta2, beta3=0.0,
            N=N, color=color, filled=filled, bounds=bounds, plotroots=plotroots,
            alpha=alpha, scalefac=scalefac, longtitle=longtitle, to_file_region=to_file_region,
            fignum_region=fignum_region, fignum_controller=fignum_controller)

        ax_controller = fig_controller.get_axes()[0]
        if longtitle:
            ax_controller.set_title("PI Controller Stability Function for %s\n($\\beta_1 = %.2f, \\beta_2 = %.2f$)" % (self.name, beta1, beta2))
        else:
            ax_controller.set_title("PI Controller for %s" % (self.shortname))

        # Save or draw the plot.
        fig_controller.tight_layout()
        if to_file_controller:
            fig_controller.savefig(to_file_controller, transparent=True, bbox_inches='tight', pad_inches=0.3)
        else:
            fig_controller.canvas.draw_idle()

        return fig_region, fig_controller

    def plot_PID_controller_stability(self, beta1=0.49, beta2=-0.34, beta3=0.10,
                                      N=200, color='r', filled=True, bounds=None,
                                      plotroots=False, alpha=1., scalefac=1., longtitle=True,
                                      to_file_region=False, to_file_controller=False,
                                      fignum_region=None, fignum_controller=None):
        r"""Plot the absolute stability region and the function characterizing
            stepsize control stability for a PID controller of an RK pair.
            The PID controller is of the form

            .. math::

                \begin{equation}
                    h_{n+1} = \left(\frac{\mathrm{TOL}}{\mathrm{err}_{n}})^{\beta_1/k} \left(\frac{\mathrm{TOL}}{\mathrm{err}_{n-1}})^{\beta_2/k} \left(\frac{\mathrm{TOL}}{\mathrm{err}_{n-2}})^{\beta_3/k} h_{n},
                \end{equation}

            where `h` is the stepsize, `TOL` the tolerance, and `err = O(h^k)`
            the error estimate.

            By default, the region of the main method is filled in red and the
            region of the embedded method is outlined in black.

            **Example**::

                >>> from nodepy import rk
                >>> bs5 = rk.loadRKM('BS5')
                >>> bs5.plot_PID_controller_stability() # doctest: +ELLIPSIS
                (<Figure size...
        """
        import matplotlib.pyplot as plt

        fig_region, fig_controller = self._plot_controller_stability_common(
            beta1=beta1, beta2=beta2, beta3=beta3,
            N=N, color=color, filled=filled, bounds=bounds, plotroots=plotroots,
            alpha=alpha, scalefac=scalefac, longtitle=longtitle, to_file_region=to_file_region,
            fignum_region=fignum_region, fignum_controller=fignum_controller)

        ax_controller = fig_controller.get_axes()[0]
        if longtitle:
            ax_controller.set_title("PID Controller Stability Function for %s\n($\\beta_1 = %.2f, \\beta_2 = %.2f, \\beta_3 = %.2f$)" % (self.name, beta1, beta2, beta3))
        else:
            ax_controller.set_title("PID Controller for %s" % (self.shortname))

        # Save or draw the plot.
        fig_controller.tight_layout()
        if to_file_controller:
            fig_controller.savefig(to_file_controller, transparent=True, bbox_inches='tight', pad_inches=0.3)
        else:
            fig_controller.canvas.draw_idle()

        return fig_region, fig_controller


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

        The latter is now available in Sympy, and I've started a
        test implementation.  The main issue now is that things like

        AxA**2

        don't get parentheses when they really mean

        (AxA)**2.

        It's not really a bug since Ax(A**2) does show parentheses,
        but it will make it harder to parse into code.

        **Examples**:

            >>> from nodepy import rk, rt
            >>> tree = rt.list_trees(2)[0]
            >>> tree
            '{T}'
            >>> rk.elementary_weight(tree)
            b*c

        See :cite:`butcher2003`.
    """
    #raise Exception('This function does not work correctly; use the _str version')
    import nodepy.rooted_trees as rt
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
            >>> rk.elementary_weight_str(rt.RootedTree('{T^10}'))
            'dot(b,c**10)'
            >>> rk.elementary_weight_str(rt.RootedTree('{{T^11}T}'))
            'dot(b,dot(A,c**11))'
    """
    from nodepy.strmanip import mysimp
    from nodepy.rooted_trees import Dmap_str
    ewstr='dot(b,'+tree.Gprod_str(RKeta_str,Dmap_str)+')'
    ewstr=ewstr.replace('1*','')
    ewstr=mysimp(ewstr)
    if style=='matlab': ewstr=python_to_matlab(ewstr)
    if style=='fortran': ewstr=python_to_fortran(ewstr)
    return ewstr

def RKeta(tree):
    from sympy.physics.quantum import TensorProduct
    #raise Exception('This function does not work correctly; use the _str version')
    from nodepy.rooted_trees import Dprod
    from sympy import symbols
    if tree=='':  return symbols('e',commutative=False)
    if tree=='T': return symbols('c',commutative=False)
    return TensorProduct(symbols('A',commutative=False),Dprod(tree,RKeta))

def RKeta_str(tree):
    """
    Computes eta(t) for Runge-Kutta methods
    """
    from nodepy.rooted_trees import Dprod_str
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
    alpha[i,j] = alpha[i,j]+val
    alpha[i,:] -= val*alpha[j,:]
    beta[i,:]  -= val* beta[j,:]
    return alpha,beta

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

        See :cite:`gottlieb2009`.
    """
    if np.triu(alpha).any(): # Check that alpha is lower-triangular
        raise NotImplementedError('This routine is only written for explicit methods so far.')

    m=np.size(alpha,1)
    if not np.all([np.size(alpha,0),np.size(beta,0),
                    np.size(beta,1)]==[m+1,m+1,m]):
        raise Exception('Inconsistent dimensions of Shu-Osher arrays')

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

        Explicit methods:

          * 'FE':         Forward (explicit) Euler
          * 'RK44':       Classical 4-stage 4th-order
          * 'SSP22':      Trapezoidal rule 2nd-order :cite:`shu1988`
          * 'MTE22':      Minimal truncation error 2-stage 2nd-order
          * 'Mid22':      Explicit midpoint 2-stage 2nd-order
          * 'SSP33':      Optimal 3rd-order SSP method of Shu & Osher :cite:`shu1988`
          * 'Heun33':     Third-order method of Heun :cite:`heun1900`
          * 'SSP22star':  Optimal 2nd-order downwind SSP
          * 'NSSP32':     :cite:`wang2007`
          * 'NSSP33':     :cite:`wang2007`
          * 'SSP104':     Optimal 10-stage, 4th-order SSP method :cite:`ketcheson2008`
          * 'Merson43'    Merson 4(3) pair :cite:`hairer1993` pg. 167
          * 'DP5':        Dormand-Prince 5th-order :cite:`dormand1980`
          * 'PD8':        Prince-Dormand 8th-order and 7th-order pair :cite:`prince1981`
          * 'CMR6':       Calvo et al.'s 6(5) pair :cite:`calvo1990`
          * 'Fehlberg43': 4(3) pair of Fehlberg :cite:`fehlberg1969`
          * 'Fehlberg45': 5(4) pair of Fehlberg :cite:`fehlberg1969`
          * 'Lambert65':
          * 'Tsit5':      5(4) pair of Tsitouras :cite:`tsitouras2011`
          * 'HH5':        5(4) pair of Highham and Hall with lower error constant :cite:`higham1990`
          * 'HH5S':       5(4) pair of Highham and Hall with increased stepsize stability :cite:`higham1990`
          * 'BuRK65':     6-stage, 5th-order method of Butcher
          * 'CK5':        Cash-Karp 5(4)6 :cite:`cash1990`
          * 'BS3':        Bogacki-Shampine 3(2)4 pair :cite:`bogacki1989`
          * 'BS5':        Bogacki-Shampine 5(4)8 pair :cite:`bogacki1996`
          * 'SSP75':      Ruuth-Spiteri optimal downwind SSP :cite:`ruuth2004`
          * 'SSP85':      Ruuth-Spiteri optimal downwind SSP :cite:`ruuth2004`
          * 'SSP95':      Ruuth-Spiteri optimal downwind SSP :cite:`ruuth2004`
          * 'SSP54':      Ruuth-Spiteri optimal downwind SSP :cite:`ruuth2004`
          * 'SSP53':      Ruuth-Spiteri optimal downwind SSP :cite:`ruuth2004`
          * 'SSP63':      Ruuth-Spiteri optimal downwind SSP :cite:`ruuth2004`

        Diagonally Implicit methods:

          * 'BE':         Backward Euler
          * 'SDIRK23':    :cite:`norsett1974`
          * 'SDIRK34':    :cite:`norsett1974`
          * 'SDIRK54':    :cite:`hairerODEs2`
          * 'TR-BDF2':    :cite:`bank1985`

        Fully Implicit methods:

          * 'GL2':        2-stage Gauss-Legendre; see e.g. :cite:`butcher1964`
          * 'GL3':        3-stage Gauss-Legendre; see e.g. :cite:`butcher1964`
          * 'LobattoIIIA2':  :cite:`ehle1969`
          * 'LobattoIIIA3':  :cite:`ehle1969`
          * 'LobattoIIIC2':  :cite:`chipman1971`
          * 'LobattoIIIC3':  :cite:`chipman1971`
          * 'LobattoIIIC4':  :cite:`chipman1971`
          * 'RadauIIA2':  :cite:`ehle1969`
          * 'RadauIIA3':  :cite:`ehle1969`
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
    RK['LobattoIIIC4']=RungeKuttaMethod(A,b,name='Lobatto IIIC4',
                description="The LobattoIIIC method with 4 stages",shortname='LobattoIIIC4')

    #================================================
    A=np.array([[one/6,-one/3,one/6],[one/6,5*one/12,-one/12],[one/6,2*one/3,one/6]])
    b=np.array([one/6,2*one/3,one/6])
    RK['LobattoIIIC3']=RungeKuttaMethod(A,b,name='Lobatto IIIC3',
                description="The LobattoIIIC method with 3 stages",shortname='LobattoIIC3')

    #================================================
    A=np.array([[half,-half],[half,half]])
    b=np.array([half,half])
    RK['LobattoIIIC2']=RungeKuttaMethod(A,b,name='Lobatto IIIC2',
                description="The LobattoIIIC method with 2 stages",shortname='LobattoIIIC2')

    #================================================
    A=np.array([[0,0],[half,half]])
    b=np.array([half,half])
    RK['LobattoIIIA2']=RungeKuttaMethod(A,b,name='Lobatto IIIA2',
                description="The LobattoIIIA method with 2 stages",shortname='LobattoIIIA2')

    #================================================
    A=np.array([[0,0,0],[5*one/24,one/3,-one/24],[one/6, 2*one/3, one/6]])
    b=np.array([one/6, 2*one/3, one/6])
    RK['LobattoIIIA3']=RungeKuttaMethod(A,b,name='Lobatto IIIA3',
                description="The LobattoIIIA method with 3 stages",shortname='LobattoIIIA3')

    #================================================
    A=np.array([[5*one/12,-1*one/12],[3*one/4,1*one/4]])
    b=np.array([3*one/4,1*one/4])
    RK['RadauIIA2']=RungeKuttaMethod(A,b,name='Radau IIA2',
                description="The RadauIIA method with 2 stages",shortname='RadauIIA2')

    #================================================
    A=np.array([[(88-7*sqrt(6))/360,(296-169*sqrt(6))/1800,(-2+3*sqrt(6))/225],
                [(296+169*sqrt(6))/1800,(88+7*sqrt(6))/360,(-2-3*sqrt(6))/225],
                [(16-sqrt(6))/36,(16+sqrt(6))/36,one/9]])
    b=np.array([(16-sqrt(6))/36,(16+sqrt(6))/36,one/9])
    RK['RadauIIA3']=RungeKuttaMethod(A,b,name='Radau IIA3',
                description="The RadauIIA method with 3 stages",shortname='RadauIIA3')

    #================================================
    # Diagonally implicit methods
    #================================================
    # This method is from Hairer & Wanner vol. II p. 100
    A = np.array([[one/4, 0, 0, 0, 0],
                  [one/2, one/4, 0, 0, 0],
                  [17*one/50, -one/25, one/4, 0, 0],
                  [371*one/1360, -137*one/2720, 15*one/544, one/4, 0],
                  [25*one/24, -49*one/48, 125*one/16, -85*one/12, one/4]])
    b = np.squeeze(A[-1,:])
    bhat = np.array([59*one/48, -17*one/96, 225*one/32, -85*one/12, 0])
    RK['SDIRK54'] = RungeKuttaMethod(A,b,name='SDIRK 54',
                        description=r"L-Stable SDIRK method of Hairer & Wanner",
                        shortname = 'SDIRK54')

    #================================================
    # This method is from Norsett 1974
    gamma = one/2 + sqrt(3)/6
    A = np.array([[gamma, 0],[1-2*gamma, gamma]])
    b = np.array([half,half])
    RK['SDIRK23'] = RungeKuttaMethod(A, b, name='SDIRK23',
                        description=r"3rd-order SDIRK method of Norsett",
                        shortname = 'SDIRK23')

    #================================================
    # This method is from Norsett 1974
    from sympy import pi, cos
    gamma = one/2 + sqrt(3)/3 * cos(pi/18)
    A = np.array([[gamma, 0, 0],
                  [half-gamma, gamma, 0],
                  [2*gamma, one-4*gamma, gamma]])
    x = one/(24*(half-gamma)**2)
    b = np.array([x, 1-2*x, x])
    RK['SDIRK34'] = RungeKuttaMethod(A, b, name='SDIRK34',
                        description=r"4th-order SDIRK method of Norsett",
                        shortname = 'SDIRK34')

    #================================================
    # This method is from Bank et al. 1985
    A=snp.array([[0,0,0],[one/4,one/4,0],[one/3,one/3,one/3]])
    b=snp.array([one/3,one/3,one/3])
    description=r"2nd-order, L-stable DIRK method of Bank et al."
    RK['TR-BDF2'] = RungeKuttaMethod(A, b, name='TR-BDF2',
                        description=description, shortname='TR-BDF2')

    #================================================
    # SSP methods
    #================================================
    A=np.array([[0,0],[one,0]])
    b=np.array([half,half])
    RK['SSP22']=ExplicitRungeKuttaMethod(A,b,name='SSPRK 22',
                description=
                "The optimal 2-stage, 2nd order SSP Runge-Kutta method",shortname='SSPRK22')

    #================================================
    A=np.array([[0,0,0],[one,0,0],[one/4,one/4,0]])
    b=np.array([one/6,one/6,2*one/3])
    RK['SSP33']=ExplicitRungeKuttaMethod(A,b,name='SSPRK 33',
                description=
                "The optimal 3-stage, 3rd order SSP Runge-Kutta method",shortname='SSPRK33')

    #================================================
    A=np.array([[0,0,0],[one/3,0,0],[0,2*one/3,0]])
    b=np.array([one/4,0,3*one/4])
    RK['Heun33']=ExplicitRungeKuttaMethod(A,b,name='Heun RK 33',
                description= "Heun's 3-stage, 3rd order",shortname='Heun33')

    #================================================
    A=np.array([[0,0,0],[one/3,0,0],[0,one,0]])
    b=np.array([one/2,0,one/2])
    RK['NSSP32']=ExplicitRungeKuttaMethod(A,b,name='non-SSPRK 32',
                description= "Wang and Spiteri NSSP32",shortname='NSSPRK32')

    #================================================
    A=np.array([[0,0,0],[-4*one/9,0,0],[7*one/6,-one/2,0]])
    b=np.array([one/4,0,3*one/4])
    RK['NSSP33']=ExplicitRungeKuttaMethod(A,b,name='non-SSPRK 33',
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
                        name='Lambert',shortname='Lambert65')
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
    A=np.array([
        [0,0,0,0,0],
        [one/4,0,0,0,0],
        [4*one/81,32*one/81,0,0,0],
        [57*one/98,-432*one/343,1053*one/686,0,0],
        [one/6,0,27*one/52,49*one/156,0]])
    b=np.array([43*one/288,0,243*one/416,343*one/1872,one/12])
    bhat=np.array(A[-1,:])
    RK['Fehlberg43']=ExplicitRungeKuttaPair(A,b,bhat,name='Fehlberg RK4(3)5',shortname='Fehlberg43')
    #================================================
    A=np.array([[0,0,0,0,0],[one/3,0,0,0,0],[one/6,one/6,0,0,0],
        [one/8,0,3*one/8,0,0],
        [one/2,0,-3*one/2,2*one,0]])
    b=np.array([one/6,0*one,0*one,2*one/3,1*one/6])
    bhat=np.array([one/10,0*one,3*one/10,2*one/5,1*one/5])
    RK['Merson43']=ExplicitRungeKuttaPair(A,b,bhat,name='Merson RK4(3)',shortname='Merson43')
    #================================================
    A    = np.zeros((7,7))
    b    = np.zeros(7)
    bhat = np.zeros(7)
    A[1,0] = 0.161
    A[2,0] = -.8480655492356988544426874250230774675121177393430391537369234245294192976164141156943e-2
    A[2,1] = .3354806554923569885444268742502307746751211773934303915373692342452941929761641411569
    A[3,0] = 2.897153057105493432130432594192938764924887287701866490314866693455023795137503079289
    A[3,1] = -6.359448489975074843148159912383825625952700647415626703305928850207288721235210244366
    A[3,2] = 4.362295432869581411017727318190886861027813359713760212991062156752264926097707165077
    A[4,0] = 5.325864828439256604428877920840511317836476253097040101202360397727981648835607691791
    A[4,1] = -11.74888356406282787774717033978577296188744178259862899288666928009020615663593781589
    A[4,2] = 7.495539342889836208304604784564358155658679161518186721010132816213648793440552049753
    A[4,3] = -.9249506636175524925650207933207191611349983406029535244034750452930469056411389539635e-1
    A[5,0] = 5.861455442946420028659251486982647890394337666164814434818157239052507339770711679748
    A[5,1] = -12.92096931784710929170611868178335939541780751955743459166312250439928519268343184452
    A[5,2] = 8.159367898576158643180400794539253485181918321135053305748355423955009222648673734986
    A[5,3] = -.7158497328140099722453054252582973869127213147363544882721139659546372402303777878835e-1
    A[5,4] = -.2826905039406838290900305721271224146717633626879770007617876201276764571291579142206e-1
    A[6,0] = .9646076681806522951816731316512876333711995238157997181903319145764851595234062815396e-1
    A[6,1] = 0.01
    A[6,2] = .4798896504144995747752495322905965199130404621990332488332634944254542060153074523509
    A[6,3] = 1.379008574103741893192274821856872770756462643091360525934940067397245698027561293331
    A[6,4] = -3.290069515436080679901047585711363850115683290894936158531296799594813811049925401677
    A[6,5] = 2.324710524099773982415355918398765796109060233222962411944060046314465391054716027841
    b[:] = A[-1,:]
    bhat[0] = .9468075576583945807478876255758922856117527357724631226139574065785592789071067303271e-1
    bhat[1] = .9183565540343253096776363936645313759813746240984095238905939532922955247253608687270e-2
    bhat[2] = .4877705284247615707855642599631228241516691959761363774365216240304071651579571959813
    bhat[3] = 1.234297566930478985655109673884237654035539930748192848315425833500484878378061439761
    bhat[4] = -2.707712349983525454881109975059321670689605166938197378763992255714444407154902012702
    bhat[5] = 1.866628418170587035753719399566211498666255505244122593996591602841258328965767580089
    bhat[6] = one / 66
    RK['Tsit5']=ExplicitRungeKuttaPair(A.astype('float64'),b.astype('float64'),bhat.astype('float64'),name='Tsitouras RK5(4)7',shortname='Tsit5')
    #================================================
    A=np.array([
        [0,0,0,0,0,0],
        [one/5,0,0,0,0,0],
        [3*one/40,9*one/40,0,0,0,0],
        [3*one/10,-9*one/10,6*one/5,0,0,0],
        [-11*one/54,5*one/2,-70*one/27,35*one/27,0,0],
        [1631*one/55296,175*one/512,575*one/13824,44275*one/110592,253*one/4096,0]])
    b=np.array([37*one/378,0,250*one/621,125*one/594,0,512*one/1771])
    bhat=np.array([2825*one/27648,0,18575*one/48384,13525*one/55296,277*one/14336,1*one/4])
    RK['CK5']=ExplicitRungeKuttaPair(A,b,bhat,name='Cash-Karp RK5(4)6',shortname='CK5')
    #================================================
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [2*one/9, 0, 0, 0, 0, 0, 0],
        [one/12, one/4, 0, 0, 0, 0, 0],
        [one/8, 0, 3*one/8, 0, 0, 0, 0],
        [91*one/500, -27*one/100, 78*one/125, 8*one/125, 0, 0, 0],
        [-11*one/20, 27*one/20, 12*one/5, -36*one/5, 5, 0, 0],
        [one/12, 0, 27*one/32, -4*one/3, 125*one/96, 5*one/48, 0]
    ])
    b = np.array([one/12, 0, 27*one/32, -4*one/3, 125*one/96, 5*one/48, 0])
    bhat = np.array([2*one/15, 0, 27*one/80, -2*one/15, 25*one/48, one/24, one/10])
    RK['HH5']=ExplicitRungeKuttaPair(A,b,bhat,name='Higham-Hall RK5(4)7',shortname='HH5')
    #================================================
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [11*one/45, 0, 0, 0, 0, 0, 0],
        [11*one/120, 11*one/40, 0, 0, 0, 0, 0],
        [106865*one/87808, -408375*one/87808, 193875*one/43904, 0, 0, 0, 0],
        [79503*one/121000, -1053*one/440, 147753*one/56870, 27048*one/710875, 0, 0, 0],
        [89303*one/78045, -2025*one/473, 994650*one/244541, -2547216*one/28122215, 475*one/2967, 0, 0],
        [1247*one/10890, 0, 57375*one/108053, -1229312*one/1962015, 125*one/207, 43*one/114, 0]
    ])
    b = np.array([1247*one/10890, 0, 57375*one/108053, -1229312*one/1962015, 125*one/207, 43*one/114, 0])
    bhat = np.array([21487*one/185130, 0, 963225*one/1836901, -39864832*one/33354255, 2575*one/3519, 4472*one/4845, -one/10])
    RK['HH5S']=ExplicitRungeKuttaPair(A,b,bhat,name='Higham-Hall RK5(4)7S',shortname='HH5S')
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
    A=np.array([[0,0,0,0,0,0,0,0,0],[2*one/15,0,0,0,0,0,0,0,0],[1*one/20,3*one/20,0,0,0,0,0,0,0],
        [3*one/40,0,9*one/40,0,0,0,0,0,0],[86727015*one/196851553,-60129073*one/52624712,957436434*one/1378352377,83886832*one/147842441,0,0,0,0,0],[-86860849*one/45628967,111022885*one/25716487,108046682*one/101167669,-141756746*one/36005461,73139862*one/60170633,0,0,0,0],[77759591*one/16096467,-49252809*one/6452555,-381680111*one/51572984,879269579*one/66788831,-90453121*one/33722162,111179552*one/157155827,0,0,0],[237564263*one/39280295,-100523239*one/10677940,-265574846*one/27330247,317978411*one/18988713,-124494385*one/35453627,86822444*one/100138635,-12873523*one/724232625,0,0],[17572349*one/289262523,0*one,57513011*one/201864250,15587306*one/354501571,71783021*one/234982865,29672000*one/180480167,65567621*one/127060952,-79074570*one/210557597,0]])
    b=np.array([17572349*one/289262523, 0*one, 57513011*one/201864250, 15587306*one/354501571, 71783021*one/234982865, 29672000*one/180480167, 65567621*one/127060952, -79074570*one/210557597, 0])
    bhat=np.array([15231665*one/510830334, 0, 59452991*one/116050448, -28398517*one/122437738, 56673824*one/137010559, 68003849*one/426673583, 7097631*one/37564021, -71226429*one/583093742, 1*one/20])
    RK['CMR6']=ExplicitRungeKuttaPair(A.astype('float64'),b,bhat,name='Calvo 6(5)',
                                      shortname='Calvo RK6(5)')
    #================================================
    A=np.array([[0,0,0,0],
                [1*one/2,0,0,0],
                [0,3*one/4,0,0],
                [2*one/9,1*one/3,4*one/9,0]])
    b=A[-1,:]
    bhat=np.array([7*one/24,1*one/4,1*one/3,1*one/8])
    RK['BS3']=ExplicitRungeKuttaPair(A,b,bhat,name='Bogacki-Shampine RK3(2)4',shortname='BS3')
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
    A=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1*one/18,0,0,0,0,0,0,0,0,0,0,0,0],
                [1*one/48,1*one/16,0,0,0,0,0,0,0,0,0,0,0],
                [1*one/32,0,3*one/32,0,0,0,0,0,0,0,0,0,0],
                [5*one/16,0,-75*one/64,75*one/64,0,0,0,0,0,0,0,0,0],
                [3*one/80,0,0,3*one/16,3*one/20,0,0,0,0,0,0,0,0],
                [29443841*one/614563906,0,0,77736538*one/692538347,-28693883*one/1125000000,23124283*one/1800000000,0,0,0,0,0,0,0],
                [16016141*one/946692911,0,0,61564180*one/158732637,22789713*one/633445777,545815736*one/2771057229,-180193667*one/1043307555,0,0,0,0,0,0],
                [39632708*one/573591083,0,0,-433636366*one/683701615,-421739975*one/2616292301,100302831*one/723423059,790204164*one/839813087,800635310*one/3783071287,0,0,0,0,0],
                [246121993*one/1340847787,0,0,-37695042795*one/15268766246,-309121744*one/1061227803,-12992083*one/490766935,6005943493*one/2108947869,393006217*one/1396673457,123872331*one/1001029789,0,0,0,0],
                [-1028468189*one/846180014,0,0,8478235783*one/508512852,1311729495*one/1432422823,-10304129995*one/1701304382,-48777925059*one/3047939560,15336726248*one/1032824649,-45442868181*one/3398467696,3065993473*one/597172653,0,0,0],
                [185892177*one/718116043,0,0,-3185094517*one/667107341,-477755414*one/1098053517,-703635378*one/230739211,5731566787*one/1027545527,5232866602*one/850066563,-4093664535*one/808688257,3962137247*one/1805957418,65686358*one/487910083,0,0],
                [403863854*one/491063109,0,0,-5068492393*one/434740067,-411421997*one/543043805,652783627*one/914296604,11173962825*one/925320556,-13158990841*one/6184727034,3936647629*one/1978049680,-160528059*one/685178525,248638103*one/1413531060,0,0]])
    b=np.array([14005451*one/335480064,0,0,0,0,-59238493*one/1068277825,181606767*one/758867731,561292985*one/797845732,-1041891430*one/1371343529,760417239*one/1151165299,118820643*one/751138087,-528747749*one/2220607170,1*one/4])
    bhat=np.array([13451932*one/455176623,0,0,0,0,-808719846*one/976000145,1757004468*one/5645159321,656045339*one/265891186,-3867574721*one/1518517206,465885868*one/322736535,53011238*one/667516719,2*one/45,0])
    RK['PD8']=ExplicitRungeKuttaPair(A.astype('float64'),b.astype('float64'),
                                     bhat.astype('float64'),
                                     name='Prince-Dormand 8(7)',
                                     shortname='Prince-Dormand RK8(7)')
    #================================================
    A=np.array([[0,0,0,0,0,0,0], [0.392382208054010,0,0,0,0,0,0],
                [0.310348765296963 ,0.523846724909595 ,0,0,0,0,0],
                [0.114817342432177 ,0.248293597111781 ,0,0,0,0,0],
                [0.136041285050893 ,0.163250087363657 ,0,0.557898557725281 ,0,0,0],
                [0.135252145083336 ,0.207274083097540 ,-0.180995372278096 ,0.326486467604174 ,0.348595427190109 ,0,0],
                [0.082675687408986 ,0.146472328858960 ,-0.160507707995237 ,0.161924299217425 ,0.028864227879979 ,0.070259587451358 ,0]])
    b=np.array([0.110184169931401 ,0.122082833871843 ,-0.117309105328437 ,0.169714358772186, 0.143346980044187, 0.348926696469455, 0.223054066239366])
    RK['SSP75']=ExplicitRungeKuttaMethod(A,b,name='SSP 75',
                                         description='From Ruuth-Spiteri paper',
                                         shortname='SSPRK75')
    #================================================
    A=np.array([[0,0,0,0,0,0,0,0],[0.276409720937984 ,0,0,0,0,0,0,0],[0.149896412080489 ,0.289119929124728 ,0,0,0,0,0,0],
                [0.057048148321026 ,0.110034365535150 ,0.202903911101136 ,0,0,0,0,0],
                [0.169059298369086 ,0.326081269617717 ,0.450795162456598 ,0,0,0,0,0],
                [0.061792381825461 ,0.119185034557281 ,0.199236908877949 ,0.521072746262762 ,-0.001094028365068 ,0,0,0],
                [0.111048724765050 ,0.214190579933444 ,0.116299126401843 ,0.223170535417453 ,-0.037093067908355 ,0.228338214162494 ,0,0],
                [0.071096701602448 ,0.137131189752988 ,0.154859800527808 ,0.043090968302309 ,-0.163751550364691 ,0.044088771531945 ,0.102941265156393 ,0]])
    b=np.array([0.107263534301213 ,0.148908166410810 ,0.105268730914375 ,0.124847526215373 ,-0.068303238298102 ,0.127738462988848 ,0.298251879839231 ,0.156024937628252 ])
    RK['SSP85']=ExplicitRungeKuttaMethod(A,b,name='SSP 85',
                                         description='From Ruuth-Spiteri paper',
                                         shortname='SSPRK85')
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
    RK['SSP95']=ExplicitRungeKuttaMethod(A,b,name='SSP 95',
                                         description='From Ruuth-Spiteri paper',
                                         shortname='SSPRK95')
    #================================================
    A = np.array([[0,0,0,0,0],
                  [0.39175222700392,0,0,0,0],
                  [0.21766909633821, 0.36841059262959, 0,0,0],
                  [0.08269208670950, 0.13995850206999, 0.25189177424738, 0,0],
                  [0.06796628370320, 0.11503469844438, 0.20703489864929, 0.54497475021237, 0]])
    b = np.array([0.14681187618661, 0.24848290924556, 0.10425883036650, 0.27443890091960, 0.22600748319395])
    RK['SSP54'] = ExplicitRungeKuttaMethod(A,b,name='SSP 54',
                                           description='From Ruuth-Spiteri paper',
                                           shortname='SSPRK54')
    #================================================
    A = np.array([[0,0,0,0,0],
                  [0.37726891511710,0,0,0,0],
                  [0.37726891511710,0.37726891511710, 0,0,0],
                  [0.16352294089771,0.16352294089771,0.16352294089771, 0,0],
                  [0.14904059394856,0.14831273384724,0.14831273384724,0.34217696850008, 0]])
    b = np.array([0.19707596384481,0.11780316509765,0.11709725193772,0.27015874934251,0.29786487010104])
    RK['SSP53'] = ExplicitRungeKuttaMethod(A,b,name='SSP 53',
                                           description='From Ruuth-Spiteri paper',
                                           shortname='SSPRK53')
    #================================================
    alpha = np.array([[0,0,0,0,0,0],
                      [1,0,0,0,0,0],
                      [0,1,0,0,0,0],
                      [0,0,1,0,0,0],
                      [0.476769811285196,0.098511733286064,0,0.424718455428740,0,0],
                      [0,0,0,0,1,0],
                      [0,0,0.155221702560091,0,0,0.844778297439909]])
    x = 0.284220721334261
    beta = np.array([[0,0,0,0,0,0],
                     [x,0,0,0,0,0],
                     [0,x,0,0,0,0],
                     [0,0,x,0,0,0],
                     [0,0,0,0.120713785765930,0,0],
                     [0,0,0,0,x,0],
                     [0,0,0,0,0,0.240103497065900]])
    RK['SSP63'] = ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name='SSP 63',
                                           description='From Ruuth 2006 paper',
                                           shortname='SSPRK63')

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
            >>> print(rk.RK22_family(-1))
            Runge-Kutta Method
            <BLANKLINE>
             0    |
             -1/2 | -1/2
            ______|____________
                  | 2     -1
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
            >>> print(rk.RK44_family(1))
            Runge-Kutta Method
            <BLANKLINE>
             0    |
             1/2  | 1/2
             1/2  | 1/3   1/6
             1    |       -2    3
            ______|________________________
                  | 1/6   -1/3  1     1/6

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
            >>> print(SSP42)
            SSPRK(4,2)
            <BLANKLINE>
             0   |
             1/3 | 1/3
             2/3 | 1/3  1/3
             1   | 1/3  1/3  1/3
            _____|____________________
                 | 1/4  1/4  1/4  1/4

            >>> SSP42.absolute_monotonicity_radius()
            2.999999999974534

        See :cite:`ketcheson2008`.
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
    # Dense output coefficients
    b_dense = np.zeros( (m,3) )
    b_dense[0,:] = [0., 1., -(m-1.)/m]
    for i in range(1,m):
        b_dense[i,2] = 1./m
    return ContinuousExplicitRungeKuttaMethod(alpha=alpha,beta=beta,
                                    b_dense=b_dense,name=name,
                                    shortname=name)


def SSPRK3(m):
    """
        Construct the optimal m-stage third order SSP
        Runge-Kutta method (m=n**2, n>=2)

        **Input**: m -- number of stages
        **Output**: A RungeKuttaMethod

        **Examples**::

            Load the 4-stage method:
            >>> SSP43=SSPRK3(4)
            >>> print(SSP43)
            SSPRK43
            <BLANKLINE>
             0   |
             1/2 | 1/2
             1   | 1/2  1/2
             1/2 | 1/6  1/6  1/6
            _____|____________________
                 | 1/6  1/6  1/6  1/2

            >>> SSP43.absolute_monotonicity_radius()
            1.9999999999527063

        See :cite:`ketcheson2008`.
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
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name,shortname=name)



def SSPRKm(m):
    """ Construct the optimal m-stage, linearly mth order SSP
        Explicit Runge-Kutta method (m>=2).

        **Input**: m -- number of stages
        **Output**: A ExplicitRungeKuttaMethod

        **Examples**::

            Load the 4-stage method:
            >>> SSP44=SSPRKm(4)
            >>> print(SSP44)
            SSPRK44
            <BLANKLINE>
             0    |
             1    | 1
             2    | 1     1
             3    | 1     1     1
            ______|________________________
                  | 5/8   7/24  1/24  1/24


            >>> SSP44.absolute_monotonicity_radius()
            0.9999999999308784

        See :cite:`gottlieb2001`.
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
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name,shortname=name)

def SSPIRK1(m):
    """ Construct the m-stage, first order unconditionally SSP
        Implicit Runge-Kutta method with smallest
        coefficient of z^2 (in the stability polynomial)

        **Input**: m -- number of stages
        **Output**: A RungeKuttaMethod

        **Examples**::

            Load the 4-stage method:
            >>> ISSP41=SSPIRK1(4)
            >>> print(ISSP41)
            SSPIRK41
            <BLANKLINE>
             1/4 | 1/4
             1/2 | 1/4  1/4
             3/4 | 1/4  1/4  1/4
             1   | 1/4  1/4  1/4  1/4
            _____|____________________
                 | 1/4  1/4  1/4  1/4
    """
    A=snp.tri(m)/m
    b=snp.ones(m)/m
    name='SSPIRK'+str(m)+'1'
    return RungeKuttaMethod(A,b,name=name,shortname=name)


def SSPIRK2(m):
    """ Construct the optimal m-stage, second order SSP
        Implicit Runge-Kutta method (m>=2).

        **Input**: m -- number of stages
        **Output**: A RungeKuttaMethod

        **Examples**::

            Load the 4-stage method:
            >>> ISSP42=SSPIRK2(4)
            >>> print(ISSP42)
            SSPIRK42
            <BLANKLINE>
             1/8 | 1/8
             3/8 | 1/4  1/8
             5/8 | 1/4  1/4  1/8
             7/8 | 1/4  1/4  1/4  1/8
            _____|____________________
                 | 1/4  1/4  1/4  1/4

            >>> ISSP42.absolute_monotonicity_radius() # doctest: +ELLIPSIS
            7.99...

		See :cite:`ketcheson2009`.
    """
    from sympy import Rational
    r=2*m
    alpha=np.vstack([snp.zeros(m),snp.eye(m)])
    beta=alpha/r
    for i in range(m): beta[i,i]=Rational(1,r)
    name='SSPIRK'+str(m)+'2'
    return RungeKuttaMethod(alpha=alpha,beta=beta,name=name,shortname=name)


def SSPIRK3(m):
    """ Construct the optimal m-stage, third order SSP
        Implicit Runge-Kutta method (m>=2).

        **Input**: m -- number of stages
        **Output**: A RungeKuttaMethod

        **Examples**::

            Load the 4-stage method:
            >>> ISSP43=SSPIRK3(4)
            >>> print(ISSP43)  # doctest: +NORMALIZE_WHITESPACE
            SSPIRK43
            <BLANKLINE>
             1/2 - sqrt(15)/10 | 1/2 - sqrt(15)/10
             1/2 - sqrt(15)/30 | sqrt(15)/15        1/2 - sqrt(15)/10
             sqrt(15)/30 + 1/2 | sqrt(15)/15        sqrt(15)/15        1/2 - sqrt(15)/10
             sqrt(15)/10 + 1/2 | sqrt(15)/15        sqrt(15)/15        sqrt(15)/15        1/2 - sqrt(15)/10
            ___________________|____________________________________________________________________________
                               | 1/4                1/4                1/4                1/4

            >>> x=ISSP43.absolute_monotonicity_radius()
            >>> print("{:.5f}".format(x))
            6.87298

		See :cite:`ketcheson2009`.
    """
    from sympy import sqrt, Rational
    r=m-1+sqrt(m**2-1)
    alpha=np.vstack([snp.zeros(m),snp.eye(m)])
    alpha[-1,-1]=((m+1)*r)/(m*(r+2))
    beta=alpha/r
    for i in range(m): beta[i,i]=Rational(1,2)*(1-sqrt(Rational(m-1,m+1)))
    name='SSPIRK'+str(m)+'3'
    return RungeKuttaMethod(alpha=alpha,beta=beta,name=name,shortname=name)


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
            >>> print(RKC41)
            Runge-Kutta-Chebyshev (4,1)
            <BLANKLINE>
             0    |
             1/16 | 1/16
             1/4  | 1/8   1/8
             9/16 | 3/16  1/4   1/8
            ______|________________________
                  | 1/4   3/8   1/4   1/8

		See :cite:`verwer2004`.
    """

    import sympy
    one = sympy.Rational(1)

    x=sympy.Symbol('x')
    Tm=sympy.special.polynomials.chebyshevt_poly(m,x)

    w0=one+sympy.Rational(epsilon,m**2)
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

    shortname='RKC'+str(m)+'1'
    name = 'Runge-Kutta-Chebyshev ('+str(m)+',1)'
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name,shortname=shortname)


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
            >>> print(RKC42)
            Runge-Kutta-Chebyshev (4,2)
            <BLANKLINE>
             0      |
             1/5    | 1/5
             1/5    | 1/10    1/10
             8/15   | -8/45   32/135  64/135
            ________|________________________________
                    | -51/64  3/8     1       27/64

		See :cite:`verwer2004`.
    """
    import sympy
    one = sympy.Rational(1)

    x=sympy.Symbol('x')
    Tm=sympy.special.polynomials.chebyshevt_poly(m,x)

    w0=one+sympy.Rational(epsilon,m**2)
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

    shortname='RKC'+str(m)+'2'
    name = 'Runge-Kutta-Chebyshev ('+str(m)+',2)'
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name,shortname=shortname)

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

    return w[:,:-1]

def DC_pair(s,theta=0.,grid='eq'):
    r"""Spectral deferred correction embedded pairs.
        See also the help for DC().
        **Examples**::

            >>> from nodepy import rk
            >>> DC2 = rk.DC_pair(2)
            >>> print(DC2)
            Picard 3(2)
            <BLANKLINE>
             0     |
             1/2   | 1/2
             1     | 1/2    1/2
             1/2   | 5/24   1/3    -1/24
             1     | 1/6    2/3    1/6
            _______|___________________________________
                   | 1/6    0      0      2/3    1/6
                   | 1/6    2/3    1/6
    """

    if s<2:
        raise Exception('s must be equal to or greater than 2')
    dc = DC(s,theta=theta,grid=grid)
    if theta==0:
        bhat_ind = -1
    else:
        bhat_ind = -3
    if theta == 0:
        name='Picard '+str(s+1)+'('+str(s)+')'
        shortname = 'Picard'+str(s+1)+str(s)
    else:
        name='Deferred Correction '+str(s+1)+'('+str(s)+')'
    return ExplicitRungeKuttaPair(A=dc.A,b=dc.b,bhat=dc.A[bhat_ind],name=name,shortname=shortname).dj_reduce()


def DC(s,theta=0,grid='eq',num_corr=None):
    """ Spectral deferred correction methods.
        For now, based on explicit Euler and equispaced points.
        For theta=0, this is Picard iteration.

        **Input**: s -- number of grid points & number of correction iterations

        **Output**: A ExplicitRungeKuttaMethod

        Note that the number of stages is NOT equal to s.  The order
        is equal to s+1.

        **Examples**::

            >>> from nodepy import rk
            >>> dc3 = rk.DC(3)
            >>> dc3.order()
            4
            >>> dc3.principal_error_norm() #doctest: +ELLIPSIS
            0.0069444...
            >>> dc3_cheb = rk.DC(3,grid='cheb')
            >>> dc3_cheb.order()
            4
            >>> dc3_cheb.principal_error_norm() #doctest: +ELLIPSIS
            0.0066478...

        See :cite:`dutt2000,gottlieb2009`.
    """
    if num_corr is None:
        num_corr = s

    # Choose the grid:
    if grid=='eq':
        t=snp.arange(s+1)/s # Equispaced
    elif grid=='cheb':
        t=0.5*(np.cos(np.arange(0,s+1)*np.pi/s)+1.)  #Chebyshev
        t=t[::-1]

    dt=np.diff(t)

    alpha=snp.zeros([s*(num_corr+1)+1,s*(num_corr+1)])
    beta=snp.zeros([s*(num_corr+1)+1,s*(num_corr+1)])

    w=dcweights(t)       #Get the quadrature weights for our grid
                         #w[i,j] is the weight of node i for the integral
                         #over [x_j,x_j+1]

    #first iteration (k=1)
    for i in range(1,s+1):
        alpha[i,i-1] = 1
        beta[i ,i-1] = dt[i-1]

    #subsequent iterations:
    for k in range(1,num_corr+1):
        beta[s*k+1,0]=w[0,0]
        for i in range(1,s+1):
            alpha[s*k+1,0]=1
            beta[s*k+1,s*(k-1)+i]=w[i,0]

        for m in range(1,s):
            alpha[s*k+m+1,s*k+m] = 1
            beta[s*k+m+1,s*k+m] = theta*dt[m]
            beta[s*k+m+1,0]=w[0,m]
            for i in range(1,s+1):
                beta[s*k+m+1,s*(k-1)+i]=w[i,m]
                if i==m:
                    beta[s*k+m+1,s*(k-1)+i]-=theta*dt[m]

    name='Deferred correction method of order '+str(s+1)
    return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name,order=s+1).dj_reduce()


#============================================================
# Extrapolation methods
#============================================================
def extrap(p,base='euler',seq='harmonic',embedded=False, shuosher=False):
    """ Construct extrapolation methods.
        For now, based on explicit Euler, but allowing arbitrary sequences.

        **Input**: p -- number of grid points & number of extrapolation iterations
                   base -- the base method to be used ('euler' or 'midpoint')
                   seq -- extrapolation sequence

        **Output**: A ExplicitRungeKuttaMethod

        **Examples**::

            >>> from nodepy import rk
            >>> ex3 = rk.extrap(3)
            >>> print(ex3)
            Ex-Euler 3
            <BLANKLINE>
             0   |
             1/2 | 1/2
             1/3 | 1/3
             2/3 | 1/3       1/3
            _____|____________________
                 | 0    -2   3/2  3/2

            >>> ex3.num_seq_dep_stages()
            3
            >>> ex3.principal_error_norm()
            0.04606423319938055
            >>> ex3.principal_error_norm(mode='exact')
            sqrt(11)/72

            >>> ex4 = rk.extrap(2,'midpoint')
            >>> print(ex4)
            Ex-Midpoint 2
            <BLANKLINE>
             0    |
             1/2  | 1/2
             1/4  | 1/4
             1/2  |             1/2
             3/4  | 1/4               1/2
            ______|______________________________
                  | 0     -1/3  2/3   0     2/3

            >>> ex4.order()
            4

        See :cite:`hairer1993` chapter II.9.
    """
    base = base.lower()
    if not base in ['euler','midpoint']:
        raise Exception('Unrecognized base method '+base)

    if base == 'euler':
        name = 'Ex-Euler '+str(p)
        an_exp = 1
        if seq == 'harmonic':
            N = snp.arange(p)+1
        elif seq == 'Romberg':
            N = snp.arange(p)+1;  N = 2**(N-1)
    elif base == 'midpoint':
        name = 'Ex-Midpoint '+str(p)
        an_exp = 2
        N = 2*snp.arange(p)+2

    J = np.cumsum(N)+1
    order_reducer = 0
    if embedded:
        if p>1:
            order_reducer = 1
        else:
            raise Exception('Embedded pair must have order>0')
    # Number of real stages:
    nrs = J[-1]

    # Shu-Osher arrays
    alpha = snp.zeros([nrs+p*(p-1)/2-order_reducer,nrs+p*(p-1)/2-1-order_reducer])
    beta =  snp.zeros([nrs+p*(p-1)/2-order_reducer,nrs+p*(p-1)/2-1-order_reducer])

    # T_11
    alpha[1,0] = 1
    beta[1,0] = 1/N[0]
    if base == 'midpoint':
        alpha[2,0] = 1
        beta[2,1] = 2/N[0]

    for j in range(1,len(N)):
        #Form T_j1:
        alpha[J[j-1],0] = 1
        beta[ J[j-1],0] = 1/N[j]
        if base == 'midpoint':
            alpha[J[j-1]+1,0] = 1
            beta[ J[j-1]+1,J[j-1]] = 2/N[j]

        if base == 'euler':
            for i in range(1,N[j]):
                alpha[J[j-1]+i,J[j-1]+i-1] = 1
                beta[ J[j-1]+i,J[j-1]+i-1] = 1/N[j]
        elif base == 'midpoint':
            for i in range(1,int(N[j]/2)):
                alpha[J[j-1]+2+2*(i-1),J[j-1]+2*(i-1)  ] = 1
                alpha[J[j-1]+3+2*(i-1),J[j-1]+2*(i-1)+1] = 1
                beta[ J[j-1]+2+2*(i-1),J[j-1]+2*(i-1)+1] = 2/N[j]
                beta[ J[j-1]+3+2*(i-1),J[j-1]+2*(i-1)+2] = 2/N[j]


    #Really there are no more "stages", and we could form T_ss directly.
    #but it is simpler to add auxiliary stages and then reduce.
    if (embedded and p>2) or (not embedded):
        for j in range(1,p):
            #form T_{j+1,2}:
            alpha[nrs-1+j,J[j]-1] = 1 + 1/((N[j]/N[j-1])**an_exp - 1)
            alpha[nrs-1+j,J[j-1]-1] = - 1/((N[j]/N[j-1])**an_exp - 1)

    #Now form all the rest, up to T_ss:
    nsd = nrs-1+p # Number of stages done
    for k in range(2,p-order_reducer):
        for ind,j in enumerate(range(k,p)):
            #form T_{j+1,k+1}:
            alpha[nsd+ind,nsd-(p-k)+ind] = 1 + 1/((N[j]/N[j-k])**an_exp - 1)
            alpha[nsd+ind,nsd-(p-k)+ind-1] = - 1/((N[j]/N[j-k])**an_exp - 1)
        nsd += p-k

    if shuosher:
        return alpha, beta
    else:
        if base == 'midpoint':
            p = 2*p
        return ExplicitRungeKuttaMethod(alpha=alpha,beta=beta,name=name,order=p).dj_reduce()

def extrap_pair(p, base='euler'):
    """
        Returns an embedded RK pair.  If the base method is Euler, the prinicpal method has
        order p and the embedded method has order p-1.  If the base
        method is midpoint, the orders are $2p, 2(p-1)$.

        **Examples**::

            >>> from nodepy import rk
            >>> ex32 = rk.extrap_pair(3,base='Euler')
            >>> ex32.order()
            3
            >>> ex32.embedded_method.order()
            2
            >>> ex42 = rk.extrap_pair(2,base='midpoint')
            >>> ex42.order()
            4
            >>> ex42.embedded_method.order()
            2
    """
    if p<2:
        raise Exception('Embedded method must have order > 0')

    alpha1, beta1 = extrap(p, base, shuosher=True)
    alpha2, beta2 = extrap(p, base, embedded=True, shuosher=True)

    alphahat = alpha1.copy()
    alphahat[-1,:-1] = alpha2[-1,:]
    alphahat[-1,-1] = 0
    betahat = beta1.copy()
    betahat[-1,:-1] = beta2[-1,:]
    betahat[-1,-1] = 0

    if base.lower() == 'euler':
        name='Euler extrapolation '+str(p)+'('+str(p-1)+')'
        shortname='Euler_extrapolation_'+str(p)+str(p-1)
        order = (p,p-1)
    elif base.lower() == 'midpoint':
        name='Midpoint extrapolation '+str(2*p)+'('+str(2*(p-1))+')'
        shortname='Midpoint_extrapolation_'+str(2*p)+str(2*(p-1))
        order = (2*p,2*(p-1))
    return ExplicitRungeKuttaPair(alpha=alpha1, beta=beta1, alphahat=alphahat, betahat=betahat, name=name, shortname=shortname, order=order).dj_reduce()



#============================================================
# Miscellaneous functions
#============================================================
def runge_kutta_order_conditions(p,ind='all'):
    """
        This is the current method of producing the code on-the-fly
        to test order conditions for RK methods.  May be deprecated
        soon.
    """
    import nodepy.rooted_trees as rt
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
        >>> print(fe*fe)
        Runge-Kutta Method
        <BLANKLINE>
         0     |
         0.500 | 0.500
        _______|______________
               | 0.500  0.500


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


def python_to_fortran(code):
    code = code.replace("dot(b","dot_product(b")
    return code.replace("dot(","matmul(")

def python_to_matlab(code):
    r"""
        Convert python code string (order condition) to matlab code string
        Doesn't really work yet.  We need to do more parsing.
    """
    outline=code
    outline=outline.replace("**",".^")
    outline=outline.replace("*",".*")
    outline=outline.replace("dot(b,","b'*(")
    outline=outline.replace("dot(bhat,","bhat'*(")
    outline=outline.replace("dot(Ahat,","Ahat*(")
    outline=outline.replace("dot(A,","(A*")
    outline=outline.replace("( c)","c")
    outline=outline.replace("-0","")
    #print(outline)
    #print('******************')
    return outline

def relative_accuracy_efficiency(rk1,rk2,mode='float',tol=1.e-14):
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
        >>> rk.relative_accuracy_efficiency(dp5,f45) # doctest: +ELLIPSIS
        1.22229116499...
    """

    p=rk1.order(mode=mode,tol=tol)
    if rk2.order()!=p: raise Exception('Methods have different orders')

    A1=rk1.principal_error_norm(mode=mode,tol=tol)
    A2=rk2.principal_error_norm(mode=mode,tol=tol)

    return len(rk2)/len(rk1) * (A2/A1)**(1./(p+1))

def accuracy_efficiency(rk1,parallel=False,mode='float',tol=1.e-14,p=None):
    r"""
    Compute the accuracy efficiency of method rk1.

    The accuracy efficiency is

    `\eta = \frac{1}{s_1} \left(\frac{1}{A_1}\right)^{1/p+1}`

    where `s_1` are the number of stages of the the method and
    `A_1` is its principal error norms.

    **Examples**::

        Accuracy efficiency of Dormand-Prince
        >>> from nodepy import rk
        >>> dp5 = rk.loadRKM('DP5')
        >>> rk.accuracy_efficiency(dp5) # doctest: +ELLIPSIS
        0.5264921944121...
    """

    if p is None:
        p=rk1.order(mode=mode,tol=tol)
    A1=rk1.principal_error_norm(mode=mode,tol=tol)
    if parallel:
        # If we consider parallelization then we divide by number of parallel stages
        return 1.0/rk1.num_seq_dep_stages() * (1.0/A1)**(1./(p+1))
    else:
        # If we DO NOT consider parallelization then we divide by total number of stages
        return 1.0/len(rk1) * (1.0/A1)**(1./(p+1))

def linearly_stable_step_size(rk, L, acc=1.e-7, tol=1.e-13, plot=1):
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

            >>> print("{:.5f}".format(rk.linearly_stable_step_size(rk44,L1,plot=0)))
            0.00007
            >>> print("{:.5f}".format(rk.linearly_stable_step_size(rk44,L2,plot=0)))
            0.02423

            >>> sd = semidisc.load_semidisc('spectral difference advection',order=1)
            >>> print("{:.5f}".format(rk.linearly_stable_step_size(rk44,sd.L,plot=0)))
            0.01393
    """

    from nodepy.utils import bisect
    import matplotlib.pyplot as plt

    p,q = rk.__num__().stability_function(mode='float')
    if len(L.shape)==2:
        lamda = np.linalg.eigvals(L)
    else:
        lamda = L
    hmax = 2.5*len(rk)**2 / max(abs(lamda))
    h=bisect(0,hmax,acc,tol,_is_linearly_stable, params=(p,q,lamda))
    if plot:
        rk.plot_stability_region()
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

def _get_column_widths(coeffarrays):
    lenmax = []
    for coeffarray in coeffarrays:
        lenmax.append(max([len(ai) for ai in coeffarray.reshape(-1)]))
    colmax=max(lenmax)
    return lenmax, colmax


def _stability_function(alpha,beta,explicit,m,formula,mode='exact'):
    r"""
        Compute stability function from the Shu-Osher representation.
    """
    s = alpha.shape[1]

    if mode=='float':
        # Floating point calculation using numpy's
        # characteristic polynomial function
        # This is always fast, so no need for alternative
        # formulas
        p1 = np.poly(beta[:-1,:].astype(float)-np.tile(beta[-1,:].astype(float),(s,1)))
        q1 = np.poly(beta[:-1,:].astype(float))
        p = np.poly1d(p1[::-1])    # Numerator
        q = np.poly1d(q1[::-1])    # Denominator

    else: # Compute symbolically
        import sympy
        z = sympy.var('z')

        if explicit:
            v = 1 - alpha[:,1:].sum(1)
            alpha[:,0]=0.
            q1 = [sympy.Rational(1)]
        else:
            v = 1 - alpha.sum(1)

        alpha_star=sympy.Matrix(alpha[:-1,:])
        beta_star=sympy.Matrix(beta[:-1,:])
        I = sympy.eye(s)

        v_mp1 = v[-1]
        vstar = sympy.Matrix(v[:-1])
        alpha_mp1 = sympy.Matrix(alpha[-1,:]).T
        beta_mp1 = sympy.Matrix(beta[-1,:]).T

        if formula == 'det':
            xsym = I - alpha_star - z*beta_star + vstar/v_mp1 * (alpha_mp1+z*beta_mp1)
            p1 = sympy.simplify(xsym.det(method='berkowitz')*v_mp1)
            p1 = p1.as_poly(z).all_coeffs()

            denomsym = I - alpha_star - z*beta_star
            q1 = sympy.simplify(denomsym.det(method='berkowitz'))
            q1 = q1.as_poly(z).all_coeffs()

        elif formula == 'lts': # lower_triangular_solve
            p1 = (alpha_mp1 + z*beta_mp1)*((I-alpha_star-z*beta_star).lower_triangular_solve(vstar))
            p1 = sympy.poly(p1[0])+v_mp1
            p1 = p1.all_coeffs()

        elif formula == 'pow': # Power series
            apbz_star = alpha_star + beta_star*z
            apbz = sympy.Matrix(alpha_mp1+z*beta_mp1)

            # Compute (I-(alpha + z beta)^(-1) = I + (alpha + z beta) + (alpha + z beta)^2 + ... + (alpha + z beta)^(s-1)
            # This is coded for Shu-Osher coefficients
            # For them, we need to take m=s
            # For Butcher coefficients, perhaps we could get away with m=num_seq_dep_stages (???)
            apbz_power = I
            Imapbz_inv = I
            for i in range(1,m):
                apbz_power = apbz_star*apbz_power
                Imapbz_inv = Imapbz_inv + apbz_power
            p1 = apbz*Imapbz_inv

            p1 = p1*vstar
            p1 = sympy.poly(p1[0])+v_mp1
            p1 = p1.all_coeffs()


        else:
            raise Exception("Unknown value of 'formula'")

        p = np.poly1d(p1)    # Numerator
        q = np.poly1d(q1)    # Denominator

    if m < p.order:
        c = p.coeffs[-(m+1):]
        p = np.poly1d(c)

    return p,q



def _internal_stability_polynomials(alpha,beta,explicit,m,formula,mode='exact'):
    r"""
        Compute internal stability polynomials from a Shu-Osher representation.
    """
    s = alpha.shape[1]

    if mode=='float':
        # Floating-point calculation
        raise NotImplementedError
    else:
        # Symbolic calculation
        import sympy
        z = sympy.var('z')
        I = sympy.eye(s)

        if explicit:
            v = 1 - alpha[:,1:].sum(1)
            alpha[:,0]=0.
            q1 = [sympy.Rational(1)]
        else:
            v = 1 - alpha.sum(1)

        alpha_star = sympy.Matrix(alpha[:-1,:])
        beta_star  = sympy.Matrix(beta[:-1,:])

        apbz_star = alpha_star + beta_star*z
        apbz = sympy.Matrix(alpha[-1,:]+z*beta[-1,:])

        if formula == 'pow':
            apbz_power = I
            Imapbz_inv = I

            for i in range(m):
                apbz_power = apbz_star*apbz_power
                Imapbz_inv = Imapbz_inv + apbz_power
            thet = (apbz*Imapbz_inv).applyfunc(sympy.expand)

        elif formula == 'lts':
            thet = (I-apbz_star).T.upper_triangular_solve(apbz)
            thet = thet.applyfunc(sympy.expand_mul)

    # Don't consider perturbations to first stage:
    theta = [np.poly1d(theta_j.as_poly(z).all_coeffs()) for theta_j in thet[1:]]
    return theta


if __name__ == "__main__":
    import doctest
    doctest.testmod()
