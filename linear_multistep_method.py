"""
**Examples**::

    >>> import linear_multistep_method as lmm
    >>> ab3=lmm.Adams_Bashforth(3)
    >>> ab3.order()
    3
    >>> am3=lmm.Adams_Moulton(3)
    >>> am3.order()
    4
    >>> bdf2=lmm.backwards_difference(2)
    >>> bdf2.order()
    2
    >>> ssp32=lmm.lmm_ssp2(3)
    >>> ssp32.order()
    2
    >>> ssp32.ssp_coefficient()
    0.5

"""
from general_linear_method import GeneralLinearMethod
import numpy as np

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
        self.beta=beta/float(alpha[-1])
        self.alpha=alpha/float(alpha[-1])
        self.name=name

    def characteristic_polynomials(self):
        r"""
        Returns the characteristic polynomials (also known as generating
        polynomials) of a linear multistep method.  They are:

        `\rho(z) = \sum_{j=0}^k \alpha_k z^k`

        `\sigma(z) = \sum_{j=0}^k \beta_k z^k`

        **References**:
            #. [hairer1993]_ p. 370, eq. 2.4
        """
        rho=np.poly1d(self.alpha[::-1])
        sigma=np.poly1d(self.beta[::-1])
        return rho, sigma

    def order(self,tol=1.e-10):
        """ Return the order of a linear multistep method """
        p=0
        if abs(sum(self.alpha))>tol: return 0
        ocHolds=True
        while ocHolds:
            p=p+1
            ocHolds=self.satisfies_order_conditions(p,tol)
        return p-1

    def satisfies_order_conditions(self,p,tol):
        """ Return True if the linear multistep method satisfies 
            the conditions of order p (only) """
        #Roundoff errors seem to be amplified here...
        ii=np.arange(len(self.alpha))
        return abs(sum(ii**p*self.alpha-p*self.beta*ii**(p-1)))<tol

    def ssp_coefficient(self):
        r""" Return the SSP coefficient of the method.
             The SSP coefficient is given by
            
            `\min_{0 \le j < k} -\alpha_k/beta_k`

            if `\alpha_j<0` and `\beta_j>0` for all `j`, and is equal to
            zero otherwise
        """
        if np.any(self.alpha[:-1]>0) or np.any(self.beta<0): return 0
        return min([-self.alpha[j]/self.beta[j] 
                    for j in range(len(self.alpha)-1) if self.beta[j]!=0])


    def plot_stability_region(self,N=1000,color='r'):
        r""" 
            Plot the region of absolute stability of a linear multistep method.

            Uses the boundary locus method.  The stability boundary is
            given by the set of points z:

            `\{z | z=\rho(\exp(i\theta))/\sigma(\exp(i\theta)), 0\le \theta \le 2*\pi \}`

            Here `\rho` and `\sigma` are the characteristic polynomials 
            of the method.

            References:
                [leveque2007]_ section 7.6.1

            TODO: Implement something that works when the stability
                    region boundary crosses itself.
        """
        import pylab as pl
        theta=np.linspace(0.,2*np.pi,N)
        z=np.exp(theta*1j)
        rho,sigma=self.characteristic_polynomials()
        val=rho(z)/sigma(z)
        #clf()
        pl.plot(np.real(val),np.imag(val),color=color,linewidth=2)
        pl.title('Absolute Stability Region for '+self.name)
        pl.axis('Image')
        pl.hold(True)
        pl.plot([0,0],[-10,10],'--k',linewidth=2)
        pl.plot([-10,2],[0,0],'--k',linewidth=2)
        pl.hold(False)
        pl.show()

    def is_explicit(self):
        return self.beta[-1]==0

def Adams_Bashforth(k):
    r""" 
        Construct the k-step, Adams-Bashforth method.
        The methods are explicit and have order k.
        They have the form:

        `y_{n+1} = y_n + h \sum_{j=0}^{k-1} \beta_j f(y_n-k+j+1)`

        They are generated using equations (1.5) and (1.7) from 
        [hairer1993]_ III.1, along with the binomial expansion.

        .. note::
            Accuracy is lost when evaluating the order conditions
            for methods with many steps.  This could be avoided by using 
            SAGE rationals instead of NumPy doubles for the coefficient
            representation.

        References:
            #. [hairer1993]_
    """
    from scipy import comb
    alpha=np.zeros(k+1)
    beta=np.zeros(k+1)
    alpha[k]=1.
    alpha[k-1]=-1.
    gamma=np.zeros(k)
    gamma[0]=1.
    beta[k-1]=1.
    betaj=np.zeros(k+1)
    for j in range(1,k):
        gamma[j]=1-sum(gamma[:j]/np.arange(j+1,1,-1))
        for i in range(0,j+1):
            betaj[k-i-1]=(-1.)**i*comb(j,i)*gamma[j]
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

        .. note::
            Accuracy is lost when evaluating the order conditions
            for methods with many steps.  This could be avoided by using 
            SAGE rationals instead of NumPy doubles for the coefficient
            representation.

        References:
            [hairer1993]_
    """
    from scipy import comb
    alpha=np.zeros(k+1)
    beta=np.zeros(k+1)
    alpha[k]=1.
    alpha[k-1]=-1.
    gamma=np.zeros(k+1)
    gamma[0]=1.
    beta[k]=1.
    betaj=np.zeros(k+1)
    for j in range(1,k+1):
        gamma[j]= -sum(gamma[:j]/np.arange(j+1,1,-1))
        for i in range(0,j+1):
            betaj[k-i]=(-1.)**i*comb(j,i)*gamma[j]
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

        .. note::
            Accuracy is lost when evaluating the order conditions
            for methods with many steps.  This could be avoided by using 
            SAGE rationals instead of NumPy doubles for the coefficient
            representation.

        **References**:
            #.[hairer1993]_ pp. 364-365
    """
    from scipy import comb
    alpha=np.zeros(k+1)
    beta=np.zeros(k+1)
    beta[k]=1.
    gamma=np.zeros(k+1)
    gamma[0]=1.
    alphaj=np.zeros(k+1)
    for j in range(1,k+1):
        gamma[j]= 1./j
        for i in range(0,j+1):
            alphaj[k-i]=(-1.)**i*comb(j,i)*gamma[j]
        alpha=alpha+alphaj
    name=str(k)+'-step BDF method'
    return LinearMultistepMethod(alpha,beta,name=name)

def elmm_ssp2(k):
    r"""
        Returns the optimal SSP k-step linear multistep method of order 2.
    """
    alpha=np.zeros(k+1)
    beta=np.zeros(k+1)
    alpha[-1]=1.
    alpha[0]=-1./(k-1.)**2
    alpha[k-1]=-((k-1.)**2-1.)/(k-1.)**2
    beta[k-1]=k/(k-1.)
    name='Optimal '+str(k)+'-step, 2nd order SSP method.'
    return LinearMultistepMethod(alpha,beta,name=name)


def loadLMM(which='All'):
    """ 
        Load a set of standard Runge-Kutta methods for testing.

        TODO: 
            - Others?
    """
    LM={}
    #================================================
    alpha=np.array([-12.,75.,-200.,300.,-300.,137.])/137.
    beta=np.array([60.,-300.,600.,-600.,300.,0.])/137
    LM['eBDF5']=LinearMultistepMethod(alpha,beta,'eBDF 5')
    if which=='All':
        return LM
    else:
        return LM[which]


