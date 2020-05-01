"""
**Examples**::

    >>> import nodepy.linear_multistep_method as lm
    >>> ab3=lm.Adams_Bashforth(3)
    >>> ab3.order()
    3
    >>> bdf2=lm.backward_difference_formula(2)
    >>> bdf2.order()
    2
    >>> bdf2.is_zero_stable()
    True
    >>> bdf7=lm.backward_difference_formula(7)
    >>> bdf7.is_zero_stable()
    False
    >>> bdf3=lm.backward_difference_formula(3)
    >>> bdf3.A_alpha_stability()
    86
    >>> ssp32=lm.elm_ssp2(3)
    >>> ssp32.order()
    2
    >>> ssp32.ssp_coefficient()
    1/2
    >>> ssp32.plot_stability_region() #doctest: +ELLIPSIS
    <Figure...

"""
from __future__ import print_function

from __future__ import absolute_import
import numpy as np
import copy
import sympy
import nodepy.snp as snp
import matplotlib.pyplot as plt
from sympy import symbols, latex
from nodepy.snp import printable
from nodepy.general_linear_method import GeneralLinearMethod
from six.moves import map
from six.moves import range
from sympy import Rational
try:
    import sympy.combinatorial as combinatorial
except ImportError:
    import sympy.functions.combinatorial as combinatorial


class LinearMultistepMethod(GeneralLinearMethod):
    r"""
        Implementation of linear multistep methods in the form:

        `\alpha_k y_{n+k} + \alpha_{k-1} y_{n+k-1} + ... + \alpha_0 y_n
        = h ( \beta_k f_{n+k} + ... + \beta_0 f_n )`

        Methods are automatically normalized so that \alpha_k=1.

        Notes: Representation follows Hairer & Wanner p. 368, NOT Butcher.

        **References**:
            * :cite:`hairer1993` Chapter III
            * :cite:`butcher2003`
    """
    def __init__(self,alpha,beta,name='Linear multistep method',shortname='LMM',
                 description=''):
        self.beta  = beta /alpha[-1]
        self.alpha = alpha/alpha[-1]
        self.name = name
        self.shortname = shortname
        if description is not '':
            self.info = description
        else:
            if self.is_explicit():
                exp_str = "Explicit"
            else:
                exp_str = "Implicit"

            self.info = "%s %d-step method of order %d" % (exp_str, len(self), self.order())

    def __num__(self):
        """
        Returns a copy of the method but with floating-point coefficients.
        This is useful whenever we need to operate numerically without
        worrying about the representation of the method.
        """
        numself = copy.deepcopy(self)
        if self.alpha.dtype==object:
            numself.alpha=np.array(self.alpha,dtype=np.float64)
            numself.beta=np.array(self.beta,dtype=np.float64)
        return numself

    def characteristic_polynomials(self):
        r"""
        Returns the characteristic polynomials (also known as generating
        polynomials) of a linear multistep method.  They are:

        `\rho(z) = \sum_{j=0}^k \alpha_k z^k`

        `\sigma(z) = \sum_{j=0}^k \beta_k z^k`

        **Examples**::

            >>> from nodepy import lm
            >>> ab5 = lm.Adams_Bashforth(5)
            >>> rho,sigma = ab5.characteristic_polynomials()
            >>> print(rho)
               5     4
            1 x - 1 x

            >>> print(sigma)
                  4         3         2
            2.64 x - 3.853 x + 3.633 x - 1.769 x + 0.3486

        **Reference**: :cite:`hairer1993` p. 370, eq. 2.4

        """
        rho=np.poly1d(self.alpha[::-1])
        sigma=np.poly1d(self.beta[::-1])
        return rho, sigma

    def order(self,tol=1.e-10):
        r""" Return the order of the local truncation error of a linear multistep method.

        **Examples**::

            >>> from nodepy import lm
            >>> am3=lm.Adams_Moulton(3)
            >>> am3.order()
            4
        """
        p = 0
        while True:
            if self._satisfies_order_conditions(p+1,tol):
                p = p + 1
            else:
                return p

    def _satisfies_order_conditions(self,p,tol):
        """ Return True if the linear multistep method satisfies
            the conditions of order p (only) """
        ii=snp.arange(len(self.alpha))
        return abs(sum(ii**p*self.alpha-p*self.beta*ii**(p-1)))<tol

    def absolute_monotonicity_radius(self):
        return self.ssp_coefficient()

    @property
    def p(self):
        return self.order()

    def latex(self):
        r"""
        Print a LaTeX representation of a linear multistep formula.

        **Example**::

            >>> from nodepy import lm
            >>> print(lm.Adams_Bashforth(2).latex())
            \begin{align} y_{n + 2} - y_{n + 1} = \frac{3}{2}h f(y_{n + 1}) - \frac{1}{2}h f(y_{n})\end{align}

        """
        n = symbols('n')
        k = len(self)
        alpha_terms = []
        beta_terms = []
        for i in range(k+1):
            subscript = latex(n+k-i)
            if self.alpha[k-i] != 0:
                alpha_terms.append(printable(self.alpha[k-i],return_one=False) +
                                   ' y_{'+subscript+'}')
            if self.beta[k-i] != 0:
                beta_terms.append(printable(self.beta[k-i],return_one=False) +
                                  'h f(y_{'+subscript+'})')
        lhs = ' + '.join(alpha_terms)
        rhs = ' + '.join(beta_terms)
        s = r'\begin{align}'+ ' = '.join([lhs,rhs]) + r'\end{align}'
        s = s.replace('+ -','-')
        return s

    def ssp_coefficient(self):
        r""" Return the SSP coefficient of the method.

         The SSP coefficient is given by

        `\min_{0 \le j < k} -\alpha_k/beta_k`

        if `\alpha_j<0` and `\beta_j>0` for all `j`, and is equal to
        zero otherwise.


        **Examples**::

            >>> from nodepy import lm
            >>> ssp32=lm.elm_ssp2(3)
            >>> ssp32.ssp_coefficient()
            1/2

            >>> bdf2=lm.backward_difference_formula(2)
            >>> bdf2.ssp_coefficient()
            0
        """
        if np.any(self.alpha[:-1]>0) or np.any(self.beta<0):
            return 0

        return min([-self.alpha[j]/self.beta[j]
                    for j in range(len(self.alpha)-1) if self.beta[j]!=0])

    def plot_stability_region(self,N=100,bounds=None,color='r',filled=True, alpha=1.,
                              to_file=False,longtitle=False):
        r"""
            The region of absolute stability of a linear multistep method is
            the set

            `\{ z \in C : \rho(\zeta) - z \sigma(zeta) \text{ satisfies the root condition} \}`

            where `\rho(zeta)` and `\sigma(zeta)` are the characteristic
            functions of the method.

            Also plots the boundary locus, which is
            given by the set of points z:

            `\{z | z=\rho(\exp(i\theta))/\sigma(\exp(i\theta)), 0\le \theta \le 2*\pi \}`

            Here `\rho` and `\sigma` are the characteristic polynomials
            of the method.

            Reference: :cite:`leveque2007` section 7.6.1


            **Input**: (all optional)
                - N       -- Number of gridpoints to use in each direction
                - bounds  -- limits of plotting region
                - color   -- color to use for this plot
                - filled  -- if true, stability region is filled in (solid); otherwise it is outlined
        """
        rho, sigma = self.__num__().characteristic_polynomials()
        mag = lambda z: _root_condition(rho-z*sigma)
        vmag = np.vectorize(mag)

        z = self._boundary_locus()
        if bounds is None:
            # Use boundary locus to decide plot region
            realmax, realmin = np.max(np.real(z)), np.min(np.real(z))
            imagmax, imagmin = np.max(np.imag(z)), np.min(np.imag(z))
            deltar, deltai = realmax-realmin, imagmax-imagmin
            bounds = (realmin-deltar/5.,realmax+deltar/5.,
                      imagmin-deltai/5.,imagmax+deltai/5.)

        y=np.linspace(bounds[2],bounds[3],N)
        Y=np.tile(y[:,np.newaxis],(1,N))
        x=np.linspace(bounds[0],bounds[1],N)
        X=np.tile(x,(N,1))
        Z=X+Y*1j

        R=1.5-vmag(Z)

        if filled:
            plt.contourf(X,Y,R,[0,1],colors=color,alpha=alpha)
        else:
            plt.contour(X,Y,R,[0,1],colors=color,alpha=alpha)

        fig = plt.gcf()
        ax = fig.get_axes()
        if longtitle:
            plt.setp(ax,title='Absolute Stability Region for '+self.name)
        else:
            plt.setp(ax,title='Stability region')

        plt.plot([0,0],[bounds[2],bounds[3]],'--k',linewidth=2)
        plt.plot([bounds[0],bounds[1]],[0,0],'--k',linewidth=2)
        plt.plot(np.real(z),np.imag(z),color='k',linewidth=3)
        plt.axis(bounds)
        plt.axis('image')

        if to_file:
            plt.savefig(to_file, transparent=True, bbox_inches='tight', pad_inches=0.3)

        plt.draw()
        return fig

    def plot_boundary_locus(self,N=1000,figsize=None):
        r"""Plot the boundary locus, which is
            given by the set of points

            `\{z | z=\rho(\exp(i\theta))/\sigma(\exp(i\theta)), 0\le \theta \le 2*\pi \}`

            where `\rho` and `\sigma` are the characteristic polynomials
            of the method.

            Reference: :cite:`leveque2007` section 7.6.1
        """
        z = self._boundary_locus(N)

        if figsize is None:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
        plt.plot(np.real(z),np.imag(z),color='k',linewidth=3)
        plt.axis('image')
        bounds = plt.axis()
        plt.plot([0,0],[bounds[2],bounds[3]],'--k',linewidth=2)
        plt.plot([bounds[0],bounds[1]],[0,0],'--k',linewidth=2)
        plt.title('Boundary locus for '+self.name)
        plt.draw()

    def _boundary_locus(self, N=1000):
        r"""Compute the boundary locus, which is
            given by the set of points

            `\{z | z=\rho(\exp(i\theta))/\sigma(\exp(i\theta)), 0\le \theta \le 2*\pi \}`

            where `\rho` and `\sigma` are the characteristic polynomials
            of the method.

            Reference: :cite:`leveque2007` section 7.6.1
        """
        theta=np.linspace(0.,2*np.pi,N)
        zeta = np.exp(theta*1j)
        rho,sigma=self.__num__().characteristic_polynomials()
        z = rho(zeta)/sigma(zeta)

        return z

    def A_alpha_stability(self, N=1000, tol=1.e-14):
        r"""Angle of `A(\alpha)`-stability.

        The result is given in degrees.  The result is only accurate to
        about 1 degree, so we round down.

        **Examples**:

            >>> from nodepy import lm
            >>> bdf5 = lm.backward_difference_formula(5)
            >>> bdf5.A_alpha_stability()
            51
        """
        from math import atan2, floor

        z = self._boundary_locus(N)
        rad = list(map(atan2,np.imag(z),np.real(z)))
        rad = np.mod(np.array(rad),2*np.pi)

        return min(int(floor(np.min(np.abs(np.where(np.real(z)<-tol,rad,1.e99)-np.pi))/np.pi*180)),90)

    def is_explicit(self):
        return self.beta[-1]==0

    def is_zero_stable(self,tol=1.e-13):
        r""" True if the method is zero-stable.

        **Examples**::

            >>> from nodepy import lm
            >>> bdf5=lm.backward_difference_formula(5)
            >>> bdf5.is_zero_stable()
            True
        """
        rho, sigma = self.characteristic_polynomials()
        return _root_condition(rho,tol)

    def __len__(self):
        r"""Returns the number of steps used."""
        return len(self.alpha)-1


class AdditiveLinearMultistepMethod(GeneralLinearMethod):
    r"""
        Method for solving equations of the form

        `y'(t) = f(y) + g(y)`

        The method takes the form

        `\alpha_k y_{n+k} + \alpha_{k-1} y_{n+k-1} + ... + \alpha_0 y_n
        = h ( \beta_k f_{n+k} + ... + \beta_0 f_n
        + \gamma_k f_{n+k} + ... + \gamma_0 f_n )`

        Methods are automatically normalized so that \alpha_k=1.

        The usual reference for these is Ascher, Ruuth, and Whetton.
        But we follow a different notation (as just described).
    """
    def __init__(self, alpha, beta, gamma, name='Additive linear multistep method'):
        self.beta  = beta /alpha[-1]
        self.gamma = gamma/alpha[-1]
        self.alpha = alpha/alpha[-1]
        self.name = name
        self.method1 = LinearMultistepMethod(alpha, beta)
        self.method2 = LinearMultistepMethod(alpha, gamma)

    def __num__(self):
        """
        Returns a copy of the method but with floating-point coefficients.
        This is useful whenever we need to operate numerically without
        worrying about the representation of the method.
        """
        numself = copy.deepcopy(self)
        if self.alpha.dtype == object:
            numself.alpha = np.array(self.alpha, dtype=np.float64)
            numself.beta  = np.array(self.beta, dtype=np.float64)
            numself.gamma = np.array(self.gamma, dtype=np.float64)
        return numself

    def order(self,tol=1.e-10):
        r""" Return the order of the local truncation error of an additive
             linear multistep method.  The output is the minimum of the
             order of the component methods.
        """
        orders = []
        for method in (self.method1,self.method2):
            p = 0
            while True:
                if method._satisfies_order_conditions(p+1,tol):
                    p = p + 1
                else:
                    orders.append(p)
                    break

        return min(orders)

    def plot_imex_stability_region(self,both_real=False,N=100,color='r',filled=True,
                                   alpha=1.,fignum=None,bounds=[-10, 1, -5, 5]):
        r"""
            **Input**: (all optional)
                - N       -- Number of gridpoints to use in each direction
                - bounds  -- limits of plotting region
                - color   -- color to use for this plot
                - filled  -- if true, stability region is filled in (solid); otherwise it is outlined
        """
        rho, sigma1 = self.method1.__num__().characteristic_polynomials()
        rho, sigma2 = self.method2.__num__().characteristic_polynomials()
        if both_real:
            mag = lambda a, b: _max_root(rho - a*sigma1 - b*sigma2)
        else:
            mag = lambda a, b: _max_root(rho - a*sigma1 - 1j*b*sigma2)
        vmag = np.vectorize(mag)

        y = np.linspace(bounds[2],bounds[3],N)
        Y = np.tile(y[:,np.newaxis],(1,N))
        x = np.linspace(bounds[0],bounds[1],N)
        X = np.tile(x,(N,1))

        R = vmag(X,Y)

        h = plt.figure(fignum)
        if filled:
            plt.contourf(X,Y,R,[0,1],colors=color,alpha=alpha)
        else:
            plt.contour(X,Y,R,[0,1],colors=color,alpha=alpha)
        plt.contour(X,Y,R,np.linspace(0,1,10),colors='k')
        plt.title('IMEX Stability Region for '+self.name)
        plt.plot([0,0],[bounds[2],bounds[3]],'--k',linewidth=2)
        plt.plot([bounds[0],bounds[1]],[0,0],'--k',linewidth=2)
        plt.axis(bounds)
        return h

    def stiff_damping_factor(self,epsilon=1.e-7):
        r"""
        Return the magnitude of the largest root at z=-inf.
        This routine just computes a numerical approximation
        to the true value (with absolute accuracy epsilon).
        """
        rho, sigma1 = self.method1.__num__().characteristic_polynomials()
        rho, sigma2 = self.method2.__num__().characteristic_polynomials()
        mag = lambda a, b: _max_root(rho - a*sigma1 - 1j*b*sigma2)

        f=[]
        z=-1.
        f.append(mag(z,0))
        while True:
            z = z*10.
            f.append(mag(z,0))
            if np.abs(f[-1]-f[-2]) < epsilon:
                return f[-1]
            if len(f)>100:
                print(f)
                raise Exception('Unable to compute stiff damping factor: slow convergence')


# ======================================================
def _max_root(p):
    return max(np.abs(p.r))

def _root_condition(p,tol=1.e-13):
    r""" True if the polynomial `p` has all roots inside
    the unit circle and roots on the boundary of the unit circle
    are simple.

    **Examples**::

        >>> from nodepy import lm
        >>> p = np.poly1d((1,0.4,2,0.5))
        >>> lm._root_condition(p)
        False
    """
    if max(np.abs(p.r))>(1+tol):
        return False

    mod_one_roots = [r for r in p.r if abs(abs(r)-1)<tol]
    for i,r1 in enumerate(mod_one_roots):
        for r2 in mod_one_roots[i+1:]:
            if abs(r1-r2)<tol:
                return False
    return True


# ======================================================
# Families of multistep methods
# ======================================================

def Adams_Bashforth(k):
    r"""
    Construct the k-step, Adams-Bashforth method.
    The methods are explicit and have order k.
    They have the form:

    `y_{n+1} = y_n + h \sum_{j=0}^{k-1} \beta_j f(y_n-k+j+1)`

    They are generated using equations (1.5) and (1.7) from
    :cite:`hairer1993` III.1, along with the binomial expansion.

    **Examples**::

        >>> import nodepy.linear_multistep_method as lm
        >>> ab3=lm.Adams_Bashforth(3)
        >>> ab3.order()
        3

        Reference: :cite:`hairer1993`
    """
    one = Rational(1,1)

    alpha=snp.zeros(k+1)
    beta=snp.zeros(k+1)
    alpha[k]=one
    alpha[k-1]=-one
    gamma=snp.zeros(k)
    gamma[0]=one
    beta[k-1]=one
    betaj=snp.zeros(k+1)
    for j in range(1,k):
        gamma[j]=one-sum(gamma[:j]/snp.arange(j+1,1,-1))
        for i in range(0,j+1):
            betaj[k-i-1]=(-one)**i*combinatorial.factorials.binomial(j,i)*gamma[j]
        beta=beta+betaj
    name=str(k)+'-step Adams-Bashforth'
    return LinearMultistepMethod(alpha,beta,name=name,shortname='AB'+str(k))

def Nystrom(k):
    r"""
    Construct the k-step explicit Nystrom linear multistep method.
    The methods are explicit and have order k.
    They have the form:

    `y_{n+1} = y_{n-1} + h \sum_{j=0}^{k-1} \beta_j f(y_n-k+j+1)`

    They are generated using equations (1.13) and (1.7) from
    :cite:`hairer1993` III.1, along with the binomial expansion
    and the relation in exercise 4 on p. 367.

    Note that the term "Nystrom method" is also commonly used to refer
    to a class of methods for second-order ODEs; those are NOT
    the methods generated by this function.

    **Examples**::

        >>> import nodepy.linear_multistep_method as lm
        >>> nys3=lm.Nystrom(6)
        >>> nys3.order()
        6

        Reference: :cite:`hairer1993`
    """
    one = Rational(1,1)

    alpha = snp.zeros(k+1)
    alpha[k] = one
    alpha[k-2] = -one

    beta  = snp.zeros(k+1)
    kappa = snp.zeros(k)
    gamma = snp.zeros(k)
    gamma[0]  =   one
    kappa[0]  = 2*one
    beta[k-1] = 2*one
    betaj = snp.zeros(k+1)
    for j in range(1,k):
        gamma[j] = one-sum(gamma[:j]/snp.arange(j+1,1,-1))
        kappa[j] = 2 * gamma[j] - gamma[j-1]
        for i in range(0,j+1):
            betaj[k-i-1] = (-one)**i*combinatorial.factorials.binomial(j,i)*kappa[j]
        beta = beta+betaj
    name = str(k)+'-step Nystrom'
    return LinearMultistepMethod(alpha,beta,name=name,shortname='Nys'+str(k))

def Adams_Moulton(k):
    r"""
        Construct the k-step, Adams-Moulton method.
        The methods are implicit and have order k+1.
        They have the form:

        `y_{n+1} = y_n + h \sum_{j=0}^{k} \beta_j f(y_n-k+j+1)`

        They are generated using equation (1.9) and the equation in
        Exercise 3 from Hairer & Wanner III.1, along with the binomial
        expansion.

        **Examples**::

            >>> import nodepy.linear_multistep_method as lm
            >>> am3=lm.Adams_Moulton(3)
            >>> am3.order()
            4

        Reference: :cite:`hairer1993`
    """

    alpha=snp.zeros(k+1)
    beta=snp.zeros(k+1)
    alpha[k]=1
    alpha[k-1]=-1
    gamma=snp.zeros(k+1)
    gamma[0]=1
    beta[k]=1
    betaj=snp.zeros(k+1)
    for j in range(1,k+1):
        gamma[j]= -sum(gamma[:j]/snp.arange(j+1,1,-1))
        for i in range(0,j+1):
            betaj[k-i]=(-1)**i*combinatorial.factorials.binomial(j,i)*gamma[j]
        beta=beta+betaj
    name=str(k)+'-step Adams-Moulton'
    return LinearMultistepMethod(alpha,beta,name=name,shortname='AM'+str(k))

def Milne_Simpson(k):
    r"""
        Construct the k-step, Milne-Simpson method.
        The methods are implicit and (for k>=3) have order k+1.
        They have the form:

        `y_{n+1} = y_{n-1} + h \sum_{j=0}^{k} \beta_j f(y_n-k+j+1)`

        They are generated using equation (1.15), the equation in
        Exercise 3, and the relation in exercise 4, all from Hairer & Wanner
        III.1, along with the binomial expansion.

        **Examples**::

            >>> import nodepy.linear_multistep_method as lm
            >>> ms3=lm.Milne_Simpson(3)
            >>> ms3.order()
            4

        Reference: :cite:`hairer1993`
    """

    alpha = snp.zeros(k+1)
    beta  = snp.zeros(k+1)
    alpha[k] = 1
    alpha[k-2] = -1
    gamma = snp.zeros(k+1)
    kappa = snp.zeros(k+1)
    gamma[0] = 1
    kappa[0] = 2
    beta[k]  = 2
    betaj = snp.zeros(k+1)
    for j in range(1,k+1):
        gamma[j] = -sum(gamma[:j]/snp.arange(j+1,1,-1))
        kappa[j] = 2 * gamma[j] - gamma[j-1]
        for i in range(0,j+1):
            betaj[k-i] = (-1)**i*combinatorial.factorials.binomial(j,i)*kappa[j]
        beta = beta+betaj
    name = str(k)+'-step Milne-Simpson'
    return LinearMultistepMethod(alpha,beta,name=name,shortname='MS'+str(k))

def backward_difference_formula(k):
    r"""
        Construct the k-step backward differentiation method.
        The methods are implicit and have order k.
        They have the form:

        `\sum_{j=0}^{k} \alpha_j y_{n+k-j+1} = h \beta_j f(y_{n+1})`

        They are generated using equation (1.22') from Hairer & Wanner III.1,
            along with the binomial expansion.

        **Examples**::

            >>> import nodepy.linear_multistep_method as lm
            >>> bdf4=lm.backward_difference_formula(4)
            >>> bdf4.A_alpha_stability()
            73

        **Reference**: :cite:`hairer1993` pp. 364-365
    """

    alpha=snp.zeros(k+1)
    beta=snp.zeros(k+1)
    beta[k]=1
    gamma=snp.zeros(k+1)
    gamma[0]=1
    alphaj=snp.zeros(k+1)
    for j in range(1,k+1):
        gamma[j]= sympy.Rational(1,j)
        for i in range(0,j+1):
            alphaj[k-i]=(-1)**i*combinatorial.factorials.binomial(j,i)*gamma[j]
        alpha=alpha+alphaj
    name=str(k)+'-step BDF'
    return LinearMultistepMethod(alpha,beta,name=name,shortname='BDF'+str(k))

def elm_ssp2(k):
    r"""
    Returns the optimal SSP k-step linear multistep method of order 2.

    **Examples**::

        >>> import nodepy.linear_multistep_method as lm
        >>> lm10=lm.elm_ssp2(10)
        >>> lm10.ssp_coefficient()
        8/9
    """
    alpha=snp.zeros(k+1)
    beta=snp.zeros(k+1)
    alpha[-1]=sympy.Rational(1,1)
    alpha[0]=sympy.Rational(-1,(k-1)**2)
    alpha[k-1]=sympy.Rational(-(k-1)**2+1,(k-1)**2)
    beta[k-1]=sympy.Rational(k,k-1)
    name='Optimal '+str(k)+'-step, 2nd order SSP method.'
    return LinearMultistepMethod(alpha,beta,name=name)

def sand_cc(s):
    r""" Construct Sand's circle-contractive method of order `p=2(s+1)`
         that uses `2^s + 1` steps.

    **Examples**::

        >>> import nodepy.linear_multistep_method as lm
        >>> cc4 = lm.sand_cc(4)
        >>> cc4.order()
        10
        >>> cc4.ssp_coefficient()
        1/8

    **Reference**: :cite:`sand1986`
    """
    one  = sympy.Rational(1)
    zero = sympy.Rational(0)

    k = 2**s + 1

    Jn = [k,k-1]
    for i in range(1,s+1):
        Jn.append(k-1-2**i)

    alpha = snp.zeros(k+1)
    beta  = snp.zeros(k+1)

    # This is inefficient
    for j in Jn:
        tau_product = one
        tau_sum = zero
        tau = [one/(j-i) for i in Jn if i!=j]
        tau_product = np.prod(tau)
        tau_sum = np.sum(tau)
        beta[j] = tau_product**2
        alpha[j] = 2*beta[j]*tau_sum
    return LinearMultistepMethod(alpha,beta,'Sand circle-contractive')

def arw2(gam,c):
    r"""Returns the second order IMEX additive multistep method based on the
    parametrization in Section 3.2 of Ascher, Ruuth, & Whetton.  The parameters
    are gam and c.  Known methods are obtained with the following values:

        (1/2,0):   CNAB
        (1/2,1/8): MCNAB
        (0,1):     CNLF
        (1,0):     SBDF

    **Examples**::

        >>> from nodepy import lm
        >>> import sympy
        >>> CNLF = lm.arw2(0,sympy.Rational(1))
        >>> CNLF.order()
        2
        >>> CNLF.method1.ssp_coefficient()
        1
        >>> CNLF.method2.ssp_coefficient()
        0
        >>> print(CNLF.stiff_damping_factor()) #doctest: +ELLIPSIS
        0.999...
    """
    half = sympy.Rational(1,2)
    alpha = snp.array([gam-half,-2*gam,gam+half])
    beta  = snp.array([c/2,1-gam-c,gam+c/2])  # implicit part
    gamma = snp.array([-gam,gam+1,0])  # explicit part
    return AdditiveLinearMultistepMethod(alpha,beta,gamma,'ARW2('+str(gam)+','+str(c)+')')

def arw3(gam,theta,c):
    r"""Returns the third order IMEX additive multistep method based on the
    parametrization in Section 3.3 of Ascher, Ruuth, & Whetton.  The parameters
    are gamma, theta, and c.  Known methods are obtained with the following values:

        (1,0,0):     SBDF3

    Note that there is one sign error in the ARW paper; it is corrected here.
    """
    half = sympy.Rational(1,2)
    third = sympy.Rational(1,3)
    alpha = snp.array([-half*gam**2+third/2, 3*half*gam**2+gam-1, -3*half*gam**2-2*gam+half-theta,
                       half*gam**2+gam+third+theta])
    beta  = snp.array([5*half/6*theta-c, (gam**2-gam)*half+3*c-4*theta*third, 1-gam**2-3*c+23*theta*third/4,
                       (gam**2+gam)*half+c])
    gamma = snp.array([(gam**2+gam)*half+5*theta*third/4, -gam**2-2*gam-4*theta*third,
                       (gam**2+3*gam)*half+1+23*theta*third/4,0])
    return AdditiveLinearMultistepMethod(alpha,beta,gamma,'ARW3('+str(gam)+','+str(theta)+','+str(c)+')')


def loadLMM(which='All'):
    """
    Load a set of standard linear multistep methods for testing.

    **Examples**::

        >>> from nodepy import lm
        >>> ebdf5 = lm.loadLMM('eBDF5')
        >>> ebdf5.is_zero_stable()
        True
    """
    LM={}
    # ================================================
    alpha = snp.array([-12,75,-200,300,-300,137])/sympy.Rational(137,1)
    beta = snp.array([60,-300,600,-600,300,0])/sympy.Rational(137,1)
    LM['eBDF5'] = LinearMultistepMethod(alpha,beta,'eBDF 5')
    # ================================================
    theta = sympy.Rational(1,2)
    alpha = snp.array([-1,1])
    beta = snp.array([1-theta,theta])
    gamma = snp.array([1,0])
    LM['ET112'] = AdditiveLinearMultistepMethod(alpha,beta,gamma,'Euler-Theta')
    # ================================================
    if which=='All':
        return LM
    else:
        return LM[which]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
