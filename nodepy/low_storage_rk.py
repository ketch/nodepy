r"""
Typically, implementation of a Runge-Kutta method requires `s \times N`
memory locations, where `s` is the number of stages of the method and
`N` is the number of unknowns.  Certain classes of Runge-Kutta methods
can be implemented using substantially less memory, by taking advantage
of special relations among the coefficients.  Three main classes have
been developed in the literature:

    * 2N (Williamson) methods
    * 2R (van der Houwen/Wray) methods
    * 2S methods

Each of these classes requires only `2\times N` memory locations.
Additional methods have been developed that use more than two
memory locations per unknown but still provide a substantial savings
over traditional methods.  These are referred to as, e.g., 3S, 3R, 4R,
and so forth.
For a review of low-storage methods, see :cite:`ketcheson2010` .

In NodePy, low-storage methods are a subclass of explicit Runge-Kutta
methods (and/or explicit Runge-Kutta pairs).  In addition to the usual
properties, they possess arrays of low-storage coefficients.  They
override the generic RK implementation of time-stepping and use
special memory-efficient implementations instead.  It should be noted
that, although these low-storage algorithms are implemented, due to
Python language restrictions an extra intermediate copy of the solution
array will be created.  Thus the implementation in NodePy is not really
minimum-storage.

At the moment, the following classes are implemented:

    * 2S : Methods using two registers (under Ketcheson's assumption)
    * 2S* : Methods using two registers, one of which retains the previous
            step solution
    * 3S* : Methods using three registers, one of which retains the previous
            step solution
    * 2S embedded pairs
    * 3S* embedded pairs
    * 2N methods pairs
    * 2R methods and embedded pairs
    * 3R methods and embedded pairs

**Examples**::

    >>> from nodepy import lsrk, ivp
    >>> myrk = lsrk.load_low_storage('DDAS47')
    >>> print(myrk)
    DDAS4()7[2R]
    2R Method of Tselios \& Simos (2007)
     0.000 |
     0.336 | 0.336
     0.286 | 0.094  0.192
     0.745 | 0.094  0.150  0.501
     0.639 | 0.094  0.150  0.285  0.110
     0.724 | 0.094  0.150  0.285 -0.122  0.317
     0.911 | 0.094  0.150  0.285 -0.122  0.061  0.444
    _______|_________________________________________________
           | 0.094  0.150  0.285 -0.122  0.061  0.346  0.187

    >>> rk58 = lsrk.load_low_storage('RK58[3R]C').__num__()
    >>> rk58.name
    'RK5(4)8[3R+]C'
    >>> rk58.order()
    5
    >>> problem = ivp.load_ivp('vdp')
    >>> t,u = rk58(problem)
    >>> u[-1]
    array([-1.40278844,  1.23080499])
    >>> import nodepy
    >>> rk2S = lsrk.load_LSRK("{}/method_coefficients/58-2S_acc.txt".format(nodepy.__path__[0]),has_emb=True)
    >>> rk2S.order()
    5
    >>> rk2S.embedded_method.order()
    4
    >>> rk3S = lsrk.load_LSRK(nodepy.__path__[0]+'/method_coefficients/58-3Sstar_acc.txt',lstype='3S*')
    >>> rk3S.principal_error_norm() # doctest: +ELLIPSIS
    0.00035742076...
"""
from __future__ import absolute_import
from nodepy.runge_kutta_method import *
from six.moves import range

#=====================================================
class TwoNRungeKuttaMethod(ExplicitRungeKuttaMethod):
#=====================================================
    """ Class for 2N low-storage Runge-Kutta methods.

        These were developed by Williamson, and Carpenter & Kennedy.

        References:
            * :cite:`ketcheson2010`

        Examples::

            >>> from nodepy import lsrk
            >>> erk = lsrk.load_low_storage("RK45[2N]")
            >>> print(erk)
            RK45[2N]
            2N Method of Carpenter \& Kennedy (1994)
             0     |
             0.150 | 0.150
             0.370 |-0.009  0.379
             0.622 | 0.401 -0.602  0.823
             0.958 |-0.190  0.814 -0.365  0.699
            _______|___________________________________
                   | 0.006  0.345  0.029  0.468  0.153
    """
    def __init__(self, coef_a, coef_b,
            name='2N Runge-Kutta Method',description='',shortname='LSRK2N',order=None):
        r"""
            Initializes the 2N method by storing the
            low-storage coefficients and computing the Butcher
            array.

            The coefficients should be specified as follows:

                * The low-storage coefficients are `coef_a` and `coef_b`.
                * The Butcher and Shu-Osher coefficients are computed from the low-storage coefficients.
        """
        # compute alpha, beta
        m = len(coef_a)
        alpha = np.zeros([m+1, m])
        beta  = np.zeros([m+1, m])
        for i in range(2, m+1):
            alpha[i, i-2] = - coef_b[i-1] * coef_a[i-1] / coef_b[i-2]
            alpha[i, i-1] = 1 - alpha[i, i-2]
        for i in range(1, m+1):
            beta[i, i-1] = coef_b[i-1]
        super(TwoNRungeKuttaMethod,self).__init__(
            alpha=alpha, beta=beta, name=name, shortname=shortname, description=description, order=order)

#=====================================================
# End of class TwoNRungeKuttaMethod
#=====================================================

def twoR2butcher(a, b, regs):
    m = len(b)
    A = np.zeros([m,m], dtype=np.promote_types(a.dtype, b.dtype))
    for i in range(1,m):
        if regs == 2:
            A[i,i-1] = a[i-1]
            for j in np.arange(i-1):
                A[i,j] = b[j]
        elif regs == 3:
            A[i  ,i-1] = a[0,i-1]
            for j in np.arange(i-2):
                A[i,j] = b[j]
            if i < m-1:
                A[i+1,i-1] = a[1,i-1]
        else:
            #NEED TO FILL IN
            raise ValueError("Schemes with %d registers are not implemented yet."%regs)
    c = np.sum(A, 1)
    return A, b, c

#=====================================================
class TwoRRungeKuttaMethod(ExplicitRungeKuttaMethod):
#=====================================================
    """ Class for 2R/3R/4R low-storage Runge-Kutta pairs.

        These were developed by van der Houwen, Wray, and Kennedy et al.
        Only 2R and 3R methods have been implemented so far.

        References:
            * :cite:`kennedy2000`
            * :cite:`ketcheson2010`
    """
    def __init__(self,a,b,bhat=None,regs=2,
            name='2R Runge-Kutta Method',description='',shortname='LSRK2R',order=None):
        r"""
            Initializes the 2R method by storing the
            low-storage coefficients and computing the Butcher
            array.

            The coefficients should be specified as follows:

                * For all methods, the weights `b` are used to
                  fill in the appropriate entries in `A`.
                * For 2R methods, *a* is a vector of length `s-1`
                  containing the first subdiagonal of `A`
                * For 3R methods, *a* is a `2\times s-1` array
                  whose first row contains the first subdiagonal of `A`
                  and whose second row contains the second subdiagonal
                  of `A`.
        """
        self.b=b
        self.a=a
        A, _, c = twoR2butcher(a, b, regs)
        self.A = A; self.c = c
        if bhat is not None:
            self.bhat=bhat
            self.embedded_method=ExplicitRungeKuttaMethod(self.A,self.bhat)
        self.name=name
        self.shortname=shortname
        self.info=description
        self.lstype=str(regs)+'R+_pair'

        if order is not None:
            self._p = order
        else:
            self._p = None

        s = self.A.shape[0]
        if self.A.dtype == object:
            alpha = snp.normalize(np.zeros((s+1,s),dtype=object))
            beta = snp.normalize(np.zeros((s+1,s),dtype=object))
        else:
            alpha = np.zeros((s+1,s))
            beta = np.zeros((s+1,s))
        beta[:-1,:] = self.A.copy()
        beta[-1,:] = self.b.copy()

        self.alpha=alpha
        self.beta=beta

    def __num__(self):
        """
        Returns a copy of the method but with floating-point coefficients.
        This is useful whenever we need to operate numerically without
        worrying about the representation of the method.
        """
        numself = super(TwoRRungeKuttaPair, self).__num__()
        if self.a.dtype == object:
            numself.a = np.array(self.a, dtype=np.float64)
        else:
            numself.a = self.a.copy()
        return numself

#=====================================================
# End of class TwoRRungeKuttaMethod
#=====================================================


#=====================================================
class TwoRRungeKuttaPair(ExplicitRungeKuttaPair):
#=====================================================
    """ Class for 2R/3R/4R low-storage Runge-Kutta pairs.

        These were developed by van der Houwen, Wray, and Kennedy et al.
        Only 2R and 3R methods have been implemented so far.

        References:
            * :cite:`kennedy2000`
            * :cite:`ketcheson2010`
    """
    def __init__(self,a,b,bhat=None,regs=2,
            name='2R Runge-Kutta Method',description='',shortname='LSRK2R',order=(None,None)):
        r"""
            Initializes the 2R method by storing the
            low-storage coefficients and computing the Butcher
            array.

            The coefficients should be specified as follows:

                * For all methods, the weights `b` are used to
                  fill in the appropriate entries in `A`.
                * For 2R methods, *a* is a vector of length `s-1`
                  containing the first subdiagonal of `A`
                * For 3R methods, *a* is a `2\times s-1` array
                  whose first row contains the first subdiagonal of `A`
                  and whose second row contains the second subdiagonal
                  of `A`.
        """
        A, _, _ = twoR2butcher(a, b, regs)
        super(TwoRRungeKuttaPair,self).__init__(
            A=A, b=b, bhat=bhat, name=name, shortname=shortname, description=description, order=order)

        self.a = a
        self.lstype = str(regs)+'R+_pair'

    def __num__(self):
        """
        Returns a copy of the method but with floating-point coefficients.
        This is useful whenever we need to operate numerically without
        worrying about the representation of the method.
        """
        numself = super(TwoRRungeKuttaPair, self).__num__()
        if self.a.dtype == object:
            numself.a = np.array(self.a, dtype=np.float64)
        else:
            numself.a = self.a.copy()
        return numself

    def __step__(self,f,t,u,dt,errest=False,x=None,**kwargs):
        """
            Take a time step on the ODE u'=f(t,u).
            The implementation here is special for 2R/3R low-storage methods
            But it's not really ultra-low-storage yet.

            INPUT:
                f  -- function being integrated
                t  -- array of previous solution times
                u  -- array of previous solution steps
                        (u[i,:] is the solution at time t[i])
                dt -- length of time step to take

            OUTPUT:
                unew -- approximate solution at time t[-1]+dt
        """
        m=len(self); b=self.b; a=self.a
        S2=u[:]
        S1=u[:]
        S1=dt*f(t,S1)
        uhat = u[:]
        if self.lstype.startswith('2'):
            S2=S2+self.b[0]*S1
            uhat = uhat + self.bhat[0]*S1
            for i in range(1,m):
                S1 = S2 + (self.a[i-1]-self.b[i-1])*S1
                S1=dt*f(t+self.c[i]*dt,S1)
                S2=S2+self.b[i]*S1
                uhat = uhat + self.bhat[i]*S1
            if errest: return S2, np.max(np.abs(S2-uhat))
            else: return S2
        elif self.lstype.startswith('3'):
            S3=S2+self.b[0]*S1
            uhat = uhat + self.bhat[0]*S1
            S1=S3+(self.a[0,0]-self.b[0])*S1
            S2=(S1-S3)/(self.a[0,0]-self.b[0])
            for i in range(1,m-1):
                S1=dt*f(t+self.c[i]*dt,S1)
                S3=S3+self.b[i]*S1
                uhat = uhat + self.bhat[i]*S1
                S1=S3 + (self.a[0,i]-b[i])*S1 + (self.a[1,i-1]-b[i-1])*S2
                S2=(S1-S3+(self.b[i-1]-self.a[1,i-1])*S2)/(self.a[0,i]-self.b[i])
            S1=dt*f(t+self.c[m-1]*dt,S1)
            S3=S3+self.b[m-1]*S1
            uhat=uhat+self.bhat[m-1]*S1
            if errest: return S3, np.max(np.abs(S3-uhat))
            else: return S3
        else:
            raise Exception('Error: only 2R and 3R methods implemented so far!')

#=====================================================
# End of class TwoRRungeKuttaPair
#=====================================================


#=====================================================
class TwoSRungeKuttaMethod(ExplicitRungeKuttaMethod):
#=====================================================
    """
        Class for low-storage Runge-Kutta methods
        that use Ketcheson's assumption (2S, 2S*, and 3S* methods).

        This class cannot be used for embedded pairs.  Use
        the class TwoSRungeKuttaPair instead.

        The low-storage coefficient arrays `\beta,\gamma,\delta`
        follow the notation of :cite:`ketcheson2010`.

        The argument *lstype* must be one of the following values:

            * 2S
            * 2S*
            * 3S*
    """
    def __init__(self,betavec,gamma,delta,lstype,
            name='Low-storage Runge-Kutta Method',description='',shortname='LSRK2S',order=None):
        r"""
            Initializes the low-storage method by storing the
            low-storage coefficients and computing the Butcher
            coefficients.
       """
        self.betavec=betavec
        self.gamma=gamma
        self.delta=delta
        self.lstype=lstype
        self.shortname=shortname

        # Two-register methods
        if lstype=='2S' or lstype=='2S*':
            m=len(betavec)-1
            alpha=np.zeros([m+1,m])
            beta =np.zeros([m+1,m])
            for i in range(0,m): beta[i+1,i] = betavec[i+1]
            for i in range(1,m):
                beta[ i+1,i-1] = -beta[i,i-1]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i-1] = -gamma[0][i]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i  ] = 1. - alpha[i+1,i-1]

        # Three-register methods
        elif lstype.startswith('3S*'):
            m=len(betavec)-1
            alpha=np.zeros([m+1,m])
            beta =np.vstack([np.zeros(m),np.diag(betavec[1:])])
            for i in range(1,m):
                beta[ i+1,i-1] = -beta[i,i-1]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,  0] =  gamma[2][i+1]-gamma[2][i]* \
                                    gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i-1] = -gamma[0][i]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i  ] = 1. - alpha[i+1,i-1]-alpha[i+1,0]

        self.alpha, self.beta = alpha, beta
        self.A,self.b=shu_osher_to_butcher(alpha,beta)
        # Change type of A to float64
        # This can be a problem if A is symbolic
        self.A=np.tril(self.A.astype(np.float64),-1)
        self.c=np.sum(self.A,1)
        self.name=name
        self.info=description

        if order is not None:
            self._p = order
        else:
            self._p = None

    def __step__(self,f,t,u,dt):
        """
            Take a time step on the ODE u'=f(t,u).
            The implementation here is special for 2S/2S*/3S* low-storage methods,
            but it's not really ultra-low-storage yet.

            INPUT:
                f  -- function being integrated
                t  -- array of previous solution times
                u  -- array of previous solution steps
                        (u[i,:] is the solution at time t[i])
                dt -- length of time step to take

            OUTPUT:
                unew -- approximate solution at time t[-1]+dt
        """
        m=len(self)
        S1=u[-1]+0. # by adding zero we get a copy; is there a better way?
        S2=np.zeros(np.size(S1))
        if self.lstype.startswith('3S*'): S3=S1+0.
        for i in range(1,m+1):
            S2 = S2 + self.delta[i-1]*S1
            if self.lstype=='2S' or self.lstype=='2S*':
                S1 = self.gamma[0][i]*S1 + self.gamma[1][i]*S2 \
                     + self.betavec[i]*dt*f(t[-1]+self.c[i-1]*dt,S1)
            elif self.lstype.startswith('3S*'):
                S1 = self.gamma[0][i]*S1 + self.gamma[1][i]*S2 \
                     + self.gamma[2][i]*S3 \
                     + self.betavec[i]*dt*f(t[-1]+self.c[i-1]*dt,S1)
        return S1

#=====================================================
# End of class TwoSRungeKuttaMethod
#=====================================================

#=====================================================
class TwoSRungeKuttaPair(ExplicitRungeKuttaPair):
#=====================================================
    """
        Class for low-storage embedded Runge-Kutta pairs
        that use Ketcheson's assumption (2S, 2S*, and 3S* methods).

        This class is only for embedded pairs.  Use
        the class TwoSRungeKuttaMethod for single 2S/3S methods.

        The low-storage coefficient arrays `\beta,\gamma,\delta`
        follow the notation of :cite:`ketcheson2010` .

        The argument *lstype* must be one of the following values:

            * 2S
            * 2S*
            * 2S_pair
            * 3S*
            * 3S*_pair

        The 2S/2S*/3S* classes do not need an extra register for the
        error estimate, while the 2S_pair/3S_pair methods do.
    """
    def __init__(self,betavec,gamma,delta,lstype,bhat=None,
            name='Low-storage Runge-Kutta Pair',description='',shortname='LSRK2S',order=None):
        """
            Initializes the low-storage pair by storing the
            low-storage coefficients and computing the Butcher
            coefficients.
        """
        self.name=name
        self.info=description
        self.betavec=betavec
        self.gamma=gamma
        self.delta=delta
        self.lstype=lstype
        self.shortname=shortname
        self.alphahat = None
        self._p_hat = None

        if lstype=='2S_pair':
            m=len(betavec)-1
            alpha=np.zeros([m+1,m])
            beta =np.zeros([m+1,m])
            for i in range(0,m): beta[i+1,i] = betavec[i+1]
            for i in range(1,m):
                beta[ i+1,i-1] = -beta[i,i-1]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i-1] = -gamma[0][i]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i  ] = 1. - alpha[i+1,i-1]
            self.A,self.b=shu_osher_to_butcher(alpha,beta)
            # Change type of A to float64
            # This can be a problem if A is symbolic
            self.A=np.tril(self.A.astype(np.float64),-1)
            self.c=np.sum(self.A,1)
            self.bhat=np.dot(delta,np.vstack([self.A,self.b]))/sum(delta)

        elif lstype=='2S' or lstype=='2S*':
            m=len(betavec)-1
            alpha=np.zeros([m+1,m])
            beta =np.zeros([m+1,m])
            for i in range(0,m): beta[i+1,i] = betavec[i+1]
            for i in range(1,m):
                beta[ i+1,i-1] = -beta[i,i-1]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i-1] = -gamma[0][i]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i  ] = 1. - alpha[i+1,i-1]
            self.A,self.b=shu_osher_to_butcher(alpha,beta)
            # Change type of A to float64
            # This can be a problem if A is symbolic
            self.A=np.tril(self.A.astype(np.float64),-1)
            self.c=np.sum(self.A,1)
            self.bhat=bhat

        elif lstype=='3S*_pair':
            m=len(betavec)-1
            alpha=np.zeros([m+1,m])
            beta =np.zeros([m+1,m])
            for i in range(0,m): beta[i+1,i] = betavec[i+1]
            alpha[1,0]=gamma[0][1]+gamma[1][1]+gamma[2][1]
            for i in range(1,m):
                beta[ i+1,i-1]=-beta[i,i-1]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,  0]= gamma[2][i+1]-gamma[2][i]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i-1]=-gamma[0][i]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i  ]=1.-alpha[i+1,i-1]-alpha[i+1,0]
            self.A,self.b=shu_osher_to_butcher(alpha,beta)
            # Change type of A to float64
            # This can be a problem if A is symbolic
            self.A=np.tril(self.A.astype(np.float64),-1)
            self.c=np.sum(self.A,1)
            self.bhat=np.dot(delta[:m+1],np.vstack([self.A,self.b]))/sum(delta)

        elif lstype=='3S*':
            m=len(betavec)-1
            alpha=np.zeros([m+1,m])
            beta =np.vstack([np.zeros(m),np.diag(betavec[1:])])
            for i in range(1,m):
                beta[ i+1,i-1] = -beta[i,i-1]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,  0] =  gamma[2][i+1]-gamma[2][i]* \
                                    gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i-1] = -gamma[0][i]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i  ] = 1. - alpha[i+1,i-1]-alpha[i+1,0]
            self.A,self.b=shu_osher_to_butcher(alpha,beta)
            # Change type of A to float64
            # This can be a problem if A is symbolic
            self.A=np.tril(self.A.astype(np.float64),-1)
            self.c=np.sum(self.A,1)
            self.bhat=bhat

        self.alpha, self.beta = alpha, beta

        if order is not None:
            self._p = order
        else:
            self._p = None
        self.alpha = None


    def __step__(self,f,t,u,dt,errest=False,x=None):
        """
            Take a time step on the ODE u'=f(t,u).

            INPUT:
                f  -- function being integrated
                t  -- array of previous solution times
                u  -- array of previous solution steps
                        (u[i,:] is the solution at time t[i])
                dt -- length of time step to take

            OUTPUT:
                S1 -- approximate solution at time t[-1]+dt
                S2 -- error estimate at time t[-1]+dt

            The implementation here is special for 2S low-storage
            embedded pairs.
            But it's not really fully low-storage yet.
        """
        m=len(self)
        S1=u[-1]+0. # by adding zero we get a copy; is there a better way?
        S2=np.zeros(np.size(S1))
        if self.lstype.startswith('3S*'): S3=S1+0.; S4=u[-1]+0.
        elif self.lstype=='2S': S3=u[-1]+0.

        for i in range(1,m+1):
            S2 = S2 + self.delta[i-1]*S1
            if self.lstype=='2S_pair':
                S1 = self.gamma[0][i]*S1 + self.gamma[1][i]*S2 \
                     + self.betavec[i]*dt*f(t[-1]+self.c[i-1]*dt,S1)
            elif self.lstype=='2S':
                #Horribly inefficient hack:
                S3 = S3+self.bhat[i-1]*dt*f(t[-1]+self.c[i-1]*dt,S1)
                #End hack
                S1 = self.gamma[0][i]*S1 + self.gamma[1][i]*S2 \
                     + self.betavec[i]*dt*f(t[-1]+self.c[i-1]*dt,S1)
            elif self.lstype=='3S*':
                #Horribly inefficient hack:
                S4 = S4+self.bhat[i-1]*dt*f(t[-1]+self.c[i-1]*dt,S1)
                #End hack
                S1 = self.gamma[0][i]*S1 + self.gamma[1][i]*S2 \
                     + self.gamma[2][i]*S3 \
                     + self.betavec[i]*dt*f(t[-1]+self.c[i-1]*dt,S1)

        #Now put the embedded solution in S2
        if self.lstype=='2S_pair':
            S2=1./sum(self.delta[1:m+1])*(S2+self.delta[m+1]*S1)
        elif self.lstype=='2S': S2=S3
        elif self.lstype=='3S*': S2=S4

        if errest: return S1, np.max(np.abs(S1-S2))
        else: return S1



#=====================================================
#End of TwoSRungeKuttaPair class
#=====================================================

def load_LSRK(file,lstype='2S',has_emb=False):
    """
        Load low storage methods of the types 2S/2S*/3S*/2S_pair/3S_pair
        from a file containing the low-storage coefficient arrays.
        If has_emb=True, the method has an embedded method that
        requires an extra register.

        The use of both _pair and has_emb seems redundant; this should
        be corrected in the future.
    """
    from numpy import sum
    #Read in coefficients
    f=open(file,'r')
    coeff=[]
    for line in f:
        coeff.append(float(line))
    f.close()
    if has_emb:
        f=open(file+'.bhat','r')
        bhat=[]
        for line in f:
            bhat.append(float(line))
        bhat = np.array(bhat)

    # Determine number of stages
    if lstype=='2S' or lstype=='2S*': m=int(len(coeff)/3+1)
    elif lstype=='2S_pair': m=int((len(coeff)+1)/3)
    elif lstype=='3S*': m=int((len(coeff)+6.)/4.)
    elif lstype=='3S*_pair': m=int((len(coeff)+3.)/4.)

    # Fill in low-storage coefficient arrays
    beta=[0.]
    for i in range(m): beta.append(coeff[2*m-3+i])
    gamma=[[0.],[0.,1.]+coeff[0:m-1]]
    if lstype.startswith('3S*'): gamma.append([0,0,0,0]+coeff[3*m-3:4*m-6])
    if lstype=='2S' or lstype=='3S*':  delta=[1.]+coeff[m-1:2*m-3]+[0.]
    elif lstype=='2S*': delta=[1.]+[0.]*len(list(range(m-1,2*m-3)))+[0.]
    elif lstype=='2S_pair': delta=[1.]+coeff[m-1:2*m-3] +[coeff[-2],coeff[-1]]
    elif lstype=='3S*_pair': delta=[1.]+coeff[m-1:2*m-3] +[coeff[-3],coeff[-2],coeff[-1]]
    if lstype=='2S' or lstype=='2S*':
        for i in range(1,m+1): gamma[0].append(1.-gamma[1][i]*sum(delta[0:i]))
        if has_emb:
            meth = TwoSRungeKuttaPair(beta,gamma,delta,lstype,bhat=bhat)
        else:
            meth = TwoSRungeKuttaMethod(beta,gamma,delta,lstype)
    elif lstype=='2S_pair':
        for i in range(1,m+1): gamma[0].append(1.-gamma[1][i]*sum(delta[0:i]))
        meth = TwoSRungeKuttaPair(beta,gamma,delta,lstype)
    elif lstype.startswith('3S*'):
        for i in range(1,m+1): gamma[0].append(1.-gamma[2][i]
                                        -gamma[1][i]*sum(delta[0:i]))
        if lstype=='3S*':
            if has_emb:
                meth = TwoSRungeKuttaPair(beta,gamma,delta,lstype,bhat=bhat)
            else:
                meth = TwoSRungeKuttaMethod(beta,gamma,delta,lstype)

        elif lstype=='3S*_pair':
            meth = TwoSRungeKuttaPair(beta,gamma,delta,lstype)
    ord=meth.order()
    if lstype=='2S_pair':
        emb_ord=meth.embedded_order()
        eostr=str(emb_ord)
    else: eostr=''
    m=len(meth)
    lstypename=lstype
    if lstypename=='2S_pair': lstypename='2S'
    meth.name='RK'+str(ord)+'('+eostr+')'+str(m)+'['+lstypename+']'
    return meth


def load_LSRK_RKOPT(file,lstype='2S',has_emb=False):
    """
       Load low storage methods of the types 2S/2S*/3S*/2S_pair/3S_pair
       from a file containing the low-storage coefficient arrays. Such a file
       is usually written by RK-opt (see https://github.com/ketch/RK-opt).
    """
    import re

    if has_emb:
        f=open(file+'.bhat','r')
        bhat=[]
        for line in f:
            bhat.append(float(line))

    data = open(file).read()
    regex = re.compile(r"#stage.*order.*\n(?P<nb_stages>[0-9]+?)\s+(?P<order>[0-9]+?)\s+A",re.DOTALL)
    expr_match = re.match(regex, data).groupdict()

    nb_stages = int(expr_match['nb_stages'])

    beta = re.compile(r".*\nbeta\n(.*?)\n\n",re.DOTALL).match(data).group(1)
    beta = beta.split("\n")
    for k in range(nb_stages+1):
        beta[k] = [float(i) for i in beta[k].split()]

    beta_sub=[0.]
    for i in range(1,nb_stages+1):
        beta_sub.append(beta[i][i-1])

    delta = re.compile(r".*\ndelta\n(.*?)\n\n",re.DOTALL).match(data).group(1)
    delta = [float(i) for i in delta.split()]

    gamma = []
    if lstype.startswith('2S'):
        gamma1 = re.compile(r".*\ngamma1\n(.*?)\n\n",re.DOTALL).match(data).group(1)
        gamma.append([float(i) for i in gamma1.split()])

        gamma2 = re.compile(r".*\ngamma2\n(.*?)\n\n",re.DOTALL).match(data).group(1)
        gamma.append([float(i) for i in gamma2.split()])

    elif lstype.startswith('3S*'):
        gamma1 = re.compile(r".*\ngamma1\n(.*?)\n\n",re.DOTALL).match(data).group(1)
        gamma.append([float(i) for i in gamma1.split()])

        gamma2 = re.compile(r".*\ngamma2\n(.*?)\n\n",re.DOTALL).match(data).group(1)
        gamma.append([float(i) for i in gamma2.split()])

        gamma3 = re.compile(r".*\ngamma3\n(.*?)\n\n",re.DOTALL).match(data).group(1)
        gamma.append([float(i) for i in gamma3.split()])

    if lstype=='2S' or lstype=='2S*' or lstype=='3S*':
        if has_emb:
            meth = TwoSRungeKuttaPair(beta_sub,gamma,delta,lstype,bhat=bhat)
        else:
            meth = TwoSRungeKuttaMethod(beta_sub,gamma,delta,lstype)
    elif lstype=='2S_pair' or lstype=='3S*_pair':
        meth = TwoSRungeKuttaPair(beta_sub,gamma,delta,lstype)

    ord = meth.order()
    if lstype=='2S_pair':
        emb_ord=meth.embedded_order()
        eostr = str(emb_ord)
    else: eostr=''

    m = len(meth)
    lstypename = lstype

    if lstypename=='2S_pair': lstypename='2S'
    meth.name = 'RK'+str(ord)+'('+eostr+')'+str(m)+'['+lstypename+']'

    return meth


def load_low_storage(which='All'):
    """
        Loads low-storage methods from the literature.
    """
    from sympy import Rational

    RK = {}
    one = Rational(1, 1)

    #================================================
    fullname  = 'RK45[2N]'
    shortname = 'RK45[2N]'
    description = '2N Method of Carpenter \& Kennedy (1994)'
    coef_a = np.array([Rational(              0 ,              1 ),
                       Rational(  -567301805773 ,  1357537059087 ),
                       Rational( -2404267990393 ,  2016746695238 ),
                       Rational( -3550918686646 ,  2091501179385 ),
                       Rational( -1275806237668 ,   842570457699 )])
    coef_b = np.array([Rational(  1432997174477 ,  9575080441755 ),
                       Rational(  5161836677717 , 13612068292357 ),
                       Rational(  1720146321549 ,  2090206949498 ),
                       Rational(  3134564353537 ,  4481467310338 ),
                       Rational(  2277821191437 , 14882151754819 )])
    RK[shortname] = TwoNRungeKuttaMethod(coef_a, coef_b, fullname, description=description, shortname=shortname)
    #================================================
    fullname  = 'DDAS4()7[2R]'
    shortname = 'DDAS47'
    description = '2R Method of Tselios \& Simos (2007)'
    regs = 2
    b = np.array([0.0941840925477795334,
                  0.149683694803496998,
                  0.285204742060440058,
                  -0.122201846148053668,
                  0.0605151571191401122,
                  0.345986987898399296,
                  0.186627171718797670])
    g = np.array([0.241566650129646868,
                  0.0423866513027719953,
                  0.215602732678803776,
                  0.232328007537583987,
                  0.256223412574146438,
                  0.0978694102142697230])
    bhat = None
    a = b[:-1] + g
    RK[shortname] = TwoRRungeKuttaMethod(a, b, bhat, regs, fullname, description=description, shortname=shortname)
    #================================================
    fullname  = 'LDDC4()6[2R]'
    shortname = 'LDDC46'
    description = '2R Method of Calvo'
    regs = 2
    b = np.array([0.10893125722541,
                  0.13201701492152,
                  0.38911623225517,
                  -0.59203884581148,
                  0.47385028714844,
                  0.48812405426094])
    g = np.array([0.17985400977138,
                  0.14081893152111,
                  0.08255631629428,
                  0.65804425034331,
                  0.31862993413251])
    bhat = None
    a = b[:-1] + g
    RK[shortname] = TwoRRungeKuttaMethod(a, b, bhat, regs, fullname, description=description, shortname=shortname)
    #================================================
    fullname  = 'RK3(2)4[2R+]C'
    shortname = 'RK34[2R]C'
    description = 'A 2R Method of Kennedy, Carpenter, Lewis (2000)'
    regs = 2
    a = np.array([11847461282814 * one / 36547543011857,
                  3943225443063 * one /  7078155732230,
                  -346793006927 * one /  4029903576067])
    b = np.array([1017324711453 * one /  9774461848756,
                  8237718856693 * one / 13685301971492,
                  57731312506979 * one / 19404895981398,
                  -101169746363290 * one / 37734290219643])
    bhat = np.array([15763415370699 * one / 46270243929542,
                     514528521746 * one /  5659431552419,
                     27030193851939 * one /  9429696342944,
                     -69544964788955 * one / 30262026368149])
    RK[shortname] = TwoRRungeKuttaPair(a, b, bhat, regs, fullname, description=description, shortname=shortname)
    #================================================
    fullname  = 'RK4(3)5[2R+]C'
    shortname = 'RK45[2R]C'
    description = 'A 2R Method of Kennedy, Carpenter, Lewis (2000)'
    regs = 2
    a = np.array([970286171893 * one / 4311952581923,
                 6584761158862 * one / 12103376702013,
                 2251764453980 * one / 15575788980749,
                 26877169314380 * one / 34165994151039])
    b = np.array([1153189308089 * one / 22510343858157,
                 1772645290293 * one / 4653164025191,
                 -1672844663538 * one / 4480602732383,
                 2114624349019 * one / 3568978502595,
                 5198255086312 * one / 14908931495163])
    bhat = np.array([1016888040809 * one / 7410784769900,
                     11231460423587 * one / 58533540763752,
                     -1563879915014 * one / 6823010717585,
                     606302364029 * one / 971179775848,
                     1097981568119 * one / 3980877426909])
    RK[shortname] = TwoRRungeKuttaPair(a, b, bhat, regs, fullname, description=description, shortname=shortname)
    #================================================
    fullname  = 'RK4(3)5[3R+]C'
    shortname = 'RK45[3R]C'
    description = 'A 3R Method of Kennedy, Carpenter, Lewis (2000)'
    regs = 3
    a = np.array([[Rational(  2365592473904 ,  8146167614645 ),
                   Rational(  4278267785271 ,  6823155464066 ),
                   Rational(  2789585899612 ,  8986505720531 ),
                   Rational( 15310836689591 , 24358012670437 )],
                  [Rational(  -722262345248 , 10870640012513 ),
                   Rational(  1365858020701 ,  8494387045469 ),
                   Rational(     3819021186 ,  2763618202291 ),
                   Rational(              0 ,              1 )]])
    b = np.array([Rational(   846876320697 ,  6523801458457 ),
                  Rational(  3032295699695 , 12397907741132 ),
                  Rational(   612618101729 ,  6534652265123 ),
                  Rational(  1155491934595 ,  2954287928812 ),
                  Rational(   707644755468 ,  5028292464395 )])
    bhat = np.array([Rational(  1296459667021 ,  9516889378644 ),
                     Rational(  2599004989233 , 11990680747819 ),
                     Rational(  1882083615375 ,  8481715831096 ),
                     Rational(  1577862909606 ,  5567358792761 ),
                     Rational(   328334985361 ,  2316973589007 )])
    RK[shortname] = TwoRRungeKuttaPair(a, b, bhat, regs, fullname, description=description, shortname=shortname)
    #================================================
    fullname  = 'RK5(4)8[3R+]C'
    shortname = 'RK58[3R]C'
    description = 'A 3R Method of Kennedy, Carpenter, Lewis (2000)'
    regs = 3
    a = np.array([[141236061735 * one / 3636543850841,
                  7367658691349 * one / 25881828075080,
                  6185269491390 * one / 13597512850793,
                  2669739616339 * one / 18583622645114,
                  42158992267337 * one / 9664249073111,
                  970532350048 * one / 4459675494195,
                  1415616989537 * one / 7108576874996], #1st subdiagonal
                [ -343061178215 * one / 2523150225462,
                  -4057757969325 * one / 18246604264081,
                  1415180642415 * one / 13311741862438,
                  -93461894168145 * one / 25333855312294,
                  7285104933991 * one / 14106269434317,
                  -4825949463597 * one / 16828400578907,0.]]) #2nd subdiagonal
    b = np.array([514862045033 * one / 4637360145389,
                  0.,
                  0.,
                  0.,
                  2561084526938 * one / 7959061818733,
                  4857652849 * one / 7350455163355,
                  1059943012790 * one / 2822036905401,
                  2987336121747 * one / 15645656703944])
    bhat = np.array([1269299456316 * one / 16631323494719,
                     0.,
                     2153976949307 * one / 22364028786708,
                     2303038467735 * one / 18680122447354,
                     7354111305649 * one / 15643939971922,
                     768474111281 * one / 10081205039574,
                     3439095334143 * one / 10786306938509,
                     -3808726110015 * one / 23644487528593])
    RK[shortname] = TwoRRungeKuttaPair(a, b, bhat, regs, fullname, description=description, shortname=shortname)
    #================================================
    fullname  = 'RK5(4)9[2R+]S'
    shortname = 'RK59[2R]S'
    description = 'A 2R Method of Kennedy, Carpenter, Lewis (2000)'
    regs = 2
    a = np.array([1107026461565 * one / 5417078080134,
                  38141181049399 * one / 41724347789894,
                  493273079041 * one / 11940823631197,
                  1851571280403 * one / 6147804934346,
                  11782306865191 * one / 62590030070788,
                  9452544825720 * one / 13648368537481,
                  4435885630781 * one / 26285702406235,
                  2357909744247 * one / 11371140753790])
    b = np.array([2274579626619 * one / 23610510767302,
                  693987741272 * one / 12394497460941,
                  -347131529483 * one / 15096185902911,
                  1144057200723 * one / 32081666971178,
                  1562491064753 * one / 11797114684756,
                  13113619727965 * one / 44346030145118,
                  393957816125 * one / 7825732611452,
                  720647959663 * one / 6565743875477,
                  3559252274877 * one / 14424734981077])
    bhat = np.array([266888888871 * one / 3040372307578,
                     34125631160 * one / 2973680843661,
                     -653811289250 * one / 9267220972999,
                     323544662297 * one / 2461529853637,
                     1105885670474 * one / 4964345317203,
                     1408484642121 * one / 8758221613943,
                     1454774750537 * one / 11112645198328,
                     772137014323 * one / 4386814405182,
                     277420604269 * one / 1857595682219])
    RK[shortname] = TwoRRungeKuttaPair(a, b, bhat, regs, fullname, description=description, shortname=shortname)
    #================================================
    fullname  = 'RK35[3S*]'
    shortname = 'RK35[3S*]'
    description = 'A 3S* Method of Parsani, Ketcheson, Deconinck (2013)'
    lstype = "3S*"
    gamma1 = np.array([0.000000000000000000000000000000000000e+00,
                       0.000000000000000000000000000000000000e+00,
                       2.587669070352078826147135259816423059e-01,
                      -1.324366873994503035483205621858360246e-01,
                       5.055601231460399302974906277086120099e-02,
                       5.670552807902877745505065831821411848e-01])
    gamma2 = np.array([0.000000000000000000000000000000000000e+00,
                       1.000000000000000000000000000000000000e+00,
                       5.528418745102160469784280394378583878e-01,
                       6.731844400389673799267598042206373066e-01,
                       2.803103804507635077314375848800409585e-01,
                       5.521508873507393611035354297200683504e-01])
    gamma3 = np.array([0.000000000000000000000000000000000000e+00,
                       0.000000000000000000000000000000000000e+00,
                       0.000000000000000000000000000000000000e+00,
                       0.000000000000000000000000000000000000e+00,
                       2.752585813446636886503426921990467235e-01,
                      -8.950548709279785297709963742818217725e-01])
    beta   = np.array([0.000000000000000000000000000000000000e+00,
                       2.300285062878154318521950472131720744e-01,
                       3.021457892454169624762982948595890775e-01,
                       8.025601039472703979171797072922345251e-01,
                       4.362158997637629598287389853794593364e-01,
                       1.129268494470295342013699269045901019e-01])
    delta  = np.array([1.000000000000000000000000000000000000e+00,
                       3.407687209321454968602438384550623596e-01,
                       3.414399280584625162582312896120129153e-01,
                       7.229302732875589887484579776355531067e-01,
                       0.000000000000000000000000000000000000e+00])
    RK[shortname] = TwoSRungeKuttaMethod(beta, [gamma1, gamma2, gamma3], delta, lstype, fullname, description=description, shortname=shortname)
    #================================================
    fullname  = 'RK49[3S*]'
    shortname = 'RK49[3S*]'
    description = 'A 3S* Method of Parsani, Ketcheson, Deconinck (2013)'
    lstype = "3S*"
    gamma1 = np.array([0.000000000000000000000000000000000000e+00,
                       0.000000000000000000000000000000000000e+00,
                      -4.655641301259180409033433534204959869e+00,
                      -7.720264924836064412971836645738221705e-01,
                      -4.024423213419724199013671750435605645e+00,
                      -2.129685246739018711359392455051420256e-02,
                      -2.435022519234470106397338895476423204e+00,
                       1.985627480986167786580764982318214606e-02,
                      -2.810790112885284131039043131750077009e-01,
                       1.689434895835535688224382511180010624e-01])
    gamma2 = np.array([0.000000000000000000000000000000000000e+00,
                       1.000000000000000000000000000000000000e+00,
                       2.499262752607826154616077474202029407e+00,
                       5.866820365436137274528505258786026388e-01,
                       1.205141365412670806378514498646836728e+00,
                       3.474793796700869075166906441154424101e-01,
                       1.321346140128723201101479389762971550e+00,
                       3.119636324379370662107646694494178519e-01,
                       4.351419055894087395408575957844732329e-01,
                       2.359698299440788349379261035210220143e-01])
    gamma3 = np.array([0.000000000000000000000000000000000000e+00,
                       0.000000000000000000000000000000000000e+00,
                       0.000000000000000000000000000000000000e+00,
                       0.000000000000000000000000000000000000e+00,
                       7.621037111138170283552994987985584885e-01,
                      -1.981182159087218341841918345380690880e-01,
                      -6.228960706317566708989375001692678779e-01,
                      -3.752246993432626354092462861444801092e-01,
                      -3.355436539000946627453458859235979617e-01,
                      -4.560963110717484308986868768442946021e-02])
    beta   = np.array([0.000000000000000000000000000000000000e+00,
                       2.836343531977826293299926874169614166e-01,
                       9.736497978646965201221519237151369452e-01,
                       3.382358566377620112675117525213863701e-01,
                      -3.584937820217850568127460064715705812e-01,
                      -4.113955814725134448039955969989023288e-03,
                       1.427968962196018987143020240182522684e+00,
                       1.808467712038742958302606211873353459e-02,
                       1.605771316794520897630604849837254733e-01,
                       2.952226811394310090896908604918280616e-01])
    delta  = np.array([1.000000000000000000000000000000000000e+00,
                       1.262923854387806521515358326723799109e+00,
                       7.574967177560872899633181987155694515e-01,
                       5.163591158111222600979317576275207102e-01,
                      -2.746333792042827265378335255263664294e-02,
                      -4.382674653941771025777995873795589432e-01,
                       1.273587103668392783717422389599960297e+00,
                      -6.294740045442794862395885502337478101e-01,
                       0.000000000000000000000000000000000000e+00])
    RK[shortname] = TwoSRungeKuttaMethod(beta, [gamma1, gamma2, gamma3], delta, lstype, fullname, description=description, shortname=shortname)
    #================================================
    fullname  = 'RK510[3S*]'
    shortname = 'RK510[3S*]'
    description = 'A 3S* Method of Parsani, Ketcheson, Deconinck (2013)'
    lstype = "3S*"
    gamma1 = np.array([0.000000000000000000000000000000000000e+00,
                       0.000000000000000000000000000000000000e+00,
                       4.043660078504696109291671746177598834e-01,
                      -8.503427464263184631931835610885173082e-01,
                      -6.950894167072419804753735661506652832e+00,
                       9.238765225328278152261418654234148562e-01,
                      -2.563178039957404230619886220665648580e+00,
                       2.545744869966347634360204210679512471e-01,
                       3.125831733863169148435190436430275440e-01,
                      -7.007114800567585399804215740005020052e-01,
                       4.839620970980726410992645014630397782e-01])
    gamma2 = np.array([0.000000000000000000000000000000000000e+00,
                       1.000000000000000000000000000000000000e+00,
                       6.871467069752346112920804444001987576e-01,
                       1.093024760468898737286735922680236399e+00,
                       3.225975382330161345123542560031637549e+00,
                       1.041153700841396467779986778623424470e+00,
                       1.292821488864702716981014418706763536e+00,
                       7.391462769297005852564552697003819048e-01,
                       1.239129257039300047171792584776994772e-01,
                       1.842753479366766866665017232662648894e-01,
                       5.712788942697077931853755217161960900e-02])
    gamma3 = np.array([0.000000000000000000000000000000000000e+00,
                       0.000000000000000000000000000000000000e+00,
                       0.000000000000000000000000000000000000e+00,
                       0.000000000000000000000000000000000000e+00,
                      -2.393405159342139487677059150882996619e+00,
                      -1.902854422095986652863075505592860281e+00,
                      -2.820042210583207253904447497916407883e+00,
                      -1.832698464130565030316688535094726831e+00,
                      -2.199094510750697895051786190379061736e-01,
                      -4.082430660384876452972946481168037280e-01,
                      -1.377669791121207965023387487235595472e-01])
    beta   = np.array([0.000000000000000000000000000000000000e+00,
                       2.597883575710995818219828379369573668e-01,
                       1.777008800169541796742933570385503117e-02,
                       2.481636637328140659874975426646415144e-01,
                       7.941736827560429423655818936822470278e-01,
                       3.885391296871822386371775337465805933e-01,
                       1.455051664264339350562948993683676235e-01,
                       1.587517379462528854805469791244831868e-01,
                       1.650605631567659548064597174743539654e-01,
                       2.118093299943235030546873076673364267e-01,
                       1.559392340339606219945522980196983553e-01])
    delta  = np.array([1.000000000000000000000000000000000000e+00,
                      -1.331778409133849705447971700777998194e-01,
                       8.260422785246029908634568528214003891e-01,
                       1.513700430513332362281175846874248236e+00,
                      -1.305810063177048174765104704420082271e+00,
                       3.036678789342507567283746539033018053e+00,
                      -1.449458267074592576761915552197024226e+00,
                       3.834313873320957632984118390595540404e+00,
                       4.122293971923324917838726832997053862e+00,
                       0.000000000000000000000000000000000000e+00])
    RK[shortname] = TwoSRungeKuttaMethod(beta, [gamma1, gamma2, gamma3], delta, lstype, fullname, description=description, shortname=shortname)

    if which=='All':
        return RK
    else:
        return RK[which]

if __name__ == "__main__":
    import doctest
    doctest.testmod()
