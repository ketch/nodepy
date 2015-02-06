"""
**Examples**::

    >>> import linear_multistep_method as lm
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
    172
    >>> ssp32=lm.elm_ssp2(3)
    >>> ssp32.order()
    2
    >>> ssp32.ssp_coefficient()
    1/2

"""
from general_linear_method import GeneralLinearMethod
import numpy as np
import sympy
import snp



#=====================================================
class LinearMultistepMethod(GeneralLinearMethod):
#=====================================================
    r"""
        Implementation of linear multistep methods in the form:

        `\alpha_k y_{n+k} + \alpha_{k-1} y_{n+k-1} + ... + \alpha_0 y_n
        = h ( \beta_k f_{n+k} + ... + \beta_0 f_n )`

        Methods are automatically normalized so that \alpha_k=1.

        Notes: Representation follows Hairer & Wanner p. 368, NOT Butcher.

        **References**:  
            #. [hairer1993]_ Chapter III
            #. [butcher2003]_
    """
    def __init__(self,alpha,beta,name='Linear multistep method'):
        self.beta  = beta /alpha[-1]
        self.alpha = alpha/alpha[-1]
        self.name = name

    def __num__(self):
        """
        Returns a copy of the method but with floating-point coefficients.
        This is useful whenever we need to operate numerically without
        worrying about the representation of the method.
        """
        import copy
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
            >>> print rho
               5     4
            1 x - 1 x

            >>> print sigma
                  4         3         2
            2.64 x - 3.853 x + 3.633 x - 1.769 x + 0.3486

        **References**:
            #. [hairer1993]_ p. 370, eq. 2.4

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


    def plot_stability_region(self,N=100,N2=1000,color='r',filled=True, alpha=1.):
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

            References:
                [leveque2007]_ section 7.6.1


            **Input**: (all optional)
                - N       -- Number of gridpoints to use in each direction
                - bounds  -- limits of plotting region
                - color   -- color to use for this plot
                - filled  -- if true, stability region is filled in (solid); otherwise it is outlined
        """
        import matplotlib.pyplot as plt
        from utils import find_plot_bounds
        rho, sigma = self.__num__().characteristic_polynomials()
        mag = lambda z : _root_condition(rho-z*sigma)
        vmag = np.vectorize(mag)
        bounds = find_plot_bounds(vmag,guess=(-10,1,-5,5),N=101)

        y=np.linspace(bounds[2],bounds[3],N)
        Y=np.tile(y[:,np.newaxis],(1,N))
        x=np.linspace(bounds[0],bounds[1],N)
        X=np.tile(x,(N,1))
        Z=X+Y*1j

        R=1.5-vmag(Z)

        z = self._boundary_locus()

        if filled:
            plt.contourf(X,Y,R,[0,1],colors=color,alpha=alpha)
        else:
            plt.contour(X,Y,R,[0,1],colors=color,alpha=alpha)
        plt.title('Absolute Stability Region for '+self.name)
        plt.hold(True)
        plt.plot([0,0],[bounds[2],bounds[3]],'--k',linewidth=2)
        plt.plot([bounds[0],bounds[1]],[0,0],'--k',linewidth=2)
        plt.plot(np.real(z),np.imag(z),color='k',linewidth=3)
        plt.axis(bounds)
        plt.hold(False)
        plt.draw()

    def plot_boundary_locus(self,N=1000):
        r"""Plot the boundary locus, which is
            given by the set of points

            `\{z | z=\rho(\exp(i\theta))/\sigma(\exp(i\theta)), 0\le \theta \le 2*\pi \}`

            where `\rho` and `\sigma` are the characteristic polynomials 
            of the method.

            References:
                [leveque2007]_ section 7.6.1
        """
        import matplotlib.pyplot as plt

        z = self._boundary_locus()

        plt.figure()
        plt.plot(np.real(z),np.imag(z),color='k',linewidth=3)
        plt.axis('image')
        plt.hold(True)
        bounds = plt.axis()
        plt.plot([0,0],[bounds[2],bounds[3]],'--k',linewidth=2)
        plt.plot([bounds[0],bounds[1]],[0,0],'--k',linewidth=2)
        plt.title('Boundary locus for '+self.name)
        plt.hold(False)
        plt.draw()


    def _boundary_locus(self, N=1000):
        r"""Compute the boundary locus, which is
            given by the set of points

            `\{z | z=\rho(\exp(i\theta))/\sigma(\exp(i\theta)), 0\le \theta \le 2*\pi \}`

            where `\rho` and `\sigma` are the characteristic polynomials 
            of the method.

            References:
                [leveque2007]_ section 7.6.1
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
            103
        """
        from math import atan2, floor

        z = self._boundary_locus(N)
        rad = map(atan2,np.imag(z),np.real(z))
        rad = np.mod(np.array(rad),2*np.pi)

        return int(floor(np.min(np.abs(np.where(np.real(z)<-tol,rad,1.e99)-np.pi))/np.pi*360))

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
        return len(self.alpha)


#=====================================================
class AdditiveLinearMultistepMethod(GeneralLinearMethod):
#=====================================================
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
        import copy
        numself = copy.deepcopy(self)
        if self.alpha.dtype == object:
            numself.alpha = np.array(self.alpha, dtype=np.float64)
            numself.beta  = np.array(self.beta,  dtype=np.float64)
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


    def plot_imex_stability_region(self,N=100,N2=1000,color='r',filled=True, alpha=1.,fignum=None):
        r""" 
            **Input**: (all optional)
                - N       -- Number of gridpoints to use in each direction
                - bounds  -- limits of plotting region
                - color   -- color to use for this plot
                - filled  -- if true, stability region is filled in (solid); otherwise it is outlined
        """
        import matplotlib.pyplot as plt
        rho, sigma1 = self.method1.__num__().characteristic_polynomials()
        rho, sigma2 = self.method2.__num__().characteristic_polynomials()
        mag = lambda a, b : _max_root(rho - a*sigma1 - 1j*b*sigma2)
        vmag = np.vectorize(mag)
        bounds = [-10, 1, -5, 5]

        y = np.linspace(bounds[2],bounds[3],N)
        Y = np.tile(y[:,np.newaxis],(1,N))
        x = np.linspace(bounds[0],bounds[1],N)
        X = np.tile(x,(N,1))
        Z = X+Y*1j

        R = vmag(X,Y)

        h = plt.figure(fignum)
        plt.hold(True)
        if filled:
            plt.contourf(X,Y,R,[0,1],colors=color,alpha=alpha)
        else:
            plt.contour(X,Y,R,[0,1],colors=color,alpha=alpha)
        plt.contour(X,Y,R,np.linspace(0,1,10),colors='k')
        plt.title('IMEX Stability Region for '+self.name)
        plt.plot([0,0],[bounds[2],bounds[3]],'--k',linewidth=2)
        plt.plot([bounds[0],bounds[1]],[0,0],'--k',linewidth=2)
        plt.axis(bounds)
        plt.hold(False)
        return h

    def stiff_damping_factor(self,epsilon=1.e-7):
        r"""
        Return the magnitude of the largest root at z=-inf.
        This routine just computes a numerical approximation
        to the true value (with absolute accuracy epsilon).
        """
        rho, sigma1 = self.method1.__num__().characteristic_polynomials()
        rho, sigma2 = self.method2.__num__().characteristic_polynomials()
        mag = lambda a, b : _max_root(rho - a*sigma1 - 1j*b*sigma2)

        f=[]
        z=-1.
        f.append(mag(z,0))
        while True:
            z = z*10.
            f.append(mag(z,0))
            if np.abs(f[-1]-f[-2]) < epsilon:
                return f[-1]
            if len(f)>100:
                print f
                raise Exception('Unable to compute stiff damping factor: slow convergence')


#======================================================
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


#======================================================
# Families of multistep methods
#======================================================

def Adams_Bashforth(k):
    r""" 
    Construct the k-step, Adams-Bashforth method.
    The methods are explicit and have order k.
    They have the form:

    `y_{n+1} = y_n + h \sum_{j=0}^{k-1} \beta_j f(y_n-k+j+1)`

    They are generated using equations (1.5) and (1.7) from 
    [hairer1993]_ III.1, along with the binomial expansion.

    **Examples**::

        >>> import linear_multistep_method as lm
        >>> ab3=lm.Adams_Bashforth(3)
        >>> ab3.order()
        3

        References:
            #. [hairer1993]_
    """
    import sympy
    from sympy import Rational

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
            betaj[k-i-1]=(-one)**i*sympy.combinatorial.factorials.binomial(j,i)*gamma[j]
        beta=beta+betaj
    name=str(k)+'-step Adams-Bashforth method'
    return LinearMultistepMethod(alpha,beta,name=name)

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

            >>> import linear_multistep_method as lm
            >>> am3=lm.Adams_Moulton(3)
            >>> am3.order()
            4

        References:
            [hairer1993]_
    """
    import sympy

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
            betaj[k-i]=(-1)**i*sympy.combinatorial.factorials.binomial(j,i)*gamma[j]
            #betaj[k-i]=(-1.)**i*comb(j,i)*gamma[j]
        beta=beta+betaj
    name=str(k)+'-step Adams-Moulton method'
    return LinearMultistepMethod(alpha,beta,name=name)

def backward_difference_formula(k):
    r""" 
        Construct the k-step backward differentiation method.
        The methods are implicit and have order k.
        They have the form:

        `\sum_{j=0}^{k} \alpha_j y_{n+k-j+1} = h \beta_j f(y_{n+1})`

        They are generated using equation (1.22') from Hairer & Wanner III.1,
            along with the binomial expansion.

        **Examples**::

            >>> import linear_multistep_method as lm
            >>> bdf4=lm.backward_difference_formula(4)
            >>> bdf4.A_alpha_stability()
            146

        **References**:
            #.[hairer1993]_ pp. 364-365
    """
    import sympy

    alpha=snp.zeros(k+1)
    beta=snp.zeros(k+1)
    beta[k]=1
    gamma=snp.zeros(k+1)
    gamma[0]=1
    alphaj=snp.zeros(k+1)
    for j in range(1,k+1):
        gamma[j]= sympy.Rational(1,j)
        for i in range(0,j+1):
            alphaj[k-i]=(-1)**i*sympy.combinatorial.factorials.binomial(j,i)*gamma[j]
        alpha=alpha+alphaj
    name=str(k)+'-step BDF method'
    return LinearMultistepMethod(alpha,beta,name=name)

def elm_ssp2(k):
    r"""
    Returns the optimal SSP k-step linear multistep method of order 2.

    **Examples**::

        >>> import linear_multistep_method as lm
        >>> lm10=lm.elm_ssp2(10)
        >>> lm10.ssp_coefficient()
        8/9
    """
    import sympy

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

        >>> import linear_multistep_method as lm
        >>> cc4 = lm.sand_cc(4)
        >>> cc4.order()
        10
        >>> cc4.ssp_coefficient()
        1/8

    **References**:
        #. [sand1986]_
    """
    import sympy

    one  = sympy.Rational(1)
    zero = sympy.Rational(0)

    k = 2**s + 1
    p = 2*(s+1)

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
    r"""Returns the second order IMEX multistep method based on the
    parametrization in Section 3.2 of Ascher, Ruuth, & Spiteri.  The parameters
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
    """
    half = sympy.Rational(1,2)
    alpha = snp.array([gam-half,-2*gam,gam+half])
    beta  = snp.array([c/2,1-gam-c,gam+c/2]) #implicit part
    gamma = snp.array([-gam,gam+1,0]) #explicit part
    return AdditiveLinearMultistepMethod(alpha,beta,gamma,'ARW2('+str(gam)+','+str(c)+')')


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
    #================================================
    alpha = snp.array([-12,75,-200,300,-300,137])/sympy.Rational(137,1)
    beta = snp.array([60,-300,600,-600,300,0])/sympy.Rational(137,1)
    LM['eBDF5'] = LinearMultistepMethod(alpha,beta,'eBDF 5')
    #================================================
    theta = sympy.Rational(1,2)
    alpha = snp.array([-1,1])
    beta = snp.array([1-theta,theta])
    gamma = snp.array([1,0])
    LM['ET112'] = AdditiveLinearMultistepMethod(alpha,beta,gamma,'Euler-Theta')
    #================================================
    if which=='All':
        return LM
    else:
        return LM[which]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
