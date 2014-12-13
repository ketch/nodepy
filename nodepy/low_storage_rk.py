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
For a review of low-storage methods, see [ketcheson2010]_ .

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
    * 2R embedded pairs
    * 3R embedded pairs

**Examples**::

    >>> from nodepy import lsrk
    >>> myrk = lsrk.load_2R('DDAS47')
    >>> print myrk
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
        

"""
from runge_kutta_method import *

#=====================================================
class TwoRRungeKuttaMethod(ExplicitRungeKuttaMethod):
#=====================================================
    """ Class for 2R/3R/4R low-storage Runge-Kutta pairs.

        These were developed by van der Houwen, Wray, and Kennedy et. al.
        Only 2R and 3R methods have been implemented so far.

        References:
            * [kennedy2000]_
            * [ketcheson2010]_
    """
    def __init__(self,a,b,bhat=None,regs=2,
            name='2R Runge-Kutta Method',description=''):
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
        m=len(b)
        self.A=np.zeros([m,m])
        for i in range(1,m):
            if regs==2:
                self.A[i,i-1]=a[i-1]
                for j in range(i-1):
                    self.A[i,j]=b[j]
            elif regs==3:
                self.A[i  ,i-1]=a[0,i-1]
                for j in range(i-2):
                    self.A[i,j]=b[j]
                if i<m-1:
                    self.A[i+1,i-1]=a[1,i-1]
            elif regs==4:
                #NEED TO FILL IN
                pass
        self.c=np.sum(self.A,1)
        if bhat is not None:
            self.bhat=bhat
            self.embedded_method=ExplicitRungeKuttaMethod(self.A,self.bhat)
        self.name=name
        self.info=description
        self.lstype=str(regs)+'R+_pair'

    def __step__(self,f,t,u,dt,errest=False,x=None):
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

            TODO: Write a version of this for non-embedded methods
        """
        m=len(self); b=self.b; a=self.a
        S2=u[-1]+0.
        S1=u[-1]+0. # by adding zero we get a copy; is there a better way?
        S1=dt*f(t[-1],S1)
        uhat = u[-1]+0.
        if self.lstype.startswith('2'):
            S2=S2+self.b[0]*S1
            uhat = uhat + self.bhat[0]*S1
            for i in range(1,m):
                S1 = S2 + (self.a[i-1]-self.b[i-1])*S1
                S1=dt*f(t[-1]+self.c[i]*dt,S1)
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
                S1=dt*f(t[-1]+self.c[i]*dt,S1)
                S3=S3+self.b[i]*S1
                uhat = uhat + self.bhat[i]*S1
                S1=S3 + (self.a[0,i]-b[i])*S1 + (self.a[1,i-1]-b[i-1])*S2
                S2=(S1-S3+(self.b[i-1]-self.a[1,i-1])*S2)/(self.a[0,i]-self.b[i])
            S1=dt*f(t[-1]+self.c[m-1]*dt,S1)
            S3=S3+self.b[m-1]*S1
            uhat=uhat+self.bhat[m-1]*S1
            if errest: return S3, np.max(np.abs(S3-uhat))
            else: return S3
        else: raise Exception('Error: only 2R and 3R methods implemented so far!')

#=====================================================
# End of class TwoRRungeKuttaMethod
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
        follow the notation of [ketcheson2010]_ .

        The argument *lstype* must be one of the following values:

            * 2S
            * 2S*
            * 3S*
    """
    def __init__(self,betavec,gamma,delta,lstype,
            name='Low-storage Runge-Kutta Method',description=''):
        r"""
            Initializes the low-storage method by storing the
            low-storage coefficients and computing the Butcher
            coefficients.
       """
        self.betavec=betavec
        self.gamma=gamma
        self.delta=delta
        self.lstype=lstype

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
            self.A,self.b=shu_osher_to_butcher(alpha,beta)
            # Change type of A to float64
            # This can be a problem if A is symbolic
            self.A=np.tril(self.A.astype(np.float64),-1)
            self.c=np.sum(self.A,1)

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
            self.A,self.b=shu_osher_to_butcher(alpha,beta)
            # Change type of A to float64
            # This can be a problem if A is symbolic
            self.A=np.tril(self.A.astype(np.float64),-1)
            self.c=np.sum(self.A,1)
        self.name=name
        self.info=description

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
        follow the notation of [ketcheson2010]_ .

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
            name='Low-storage Runge-Kutta Pair',description=''):
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
 
        self.embedded_method=ExplicitRungeKuttaMethod(self.A,self.bhat)

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
    elif lstype=='2S*': delta=[1.]+[0.]*len(range(m-1,2*m-3))+[0.]
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
       

def load_2R(name):
    """
        Loads 2R low-storage methods from the literature.
    """
    bhat=None
    regs=2
    if name=='DDAS47':
        fullname='DDAS4()7[2R]'
        descript='2R Method of Tselios \& Simos (2007)'
        b=np.array([0.0941840925477795334,
                    0.149683694803496998,
                    0.285204742060440058,
                    -0.122201846148053668,
                    0.0605151571191401122,
                    0.345986987898399296,
                    0.186627171718797670])
        g=np.array([0.241566650129646868,
                    0.0423866513027719953,
                    0.215602732678803776,
                    0.232328007537583987,
                    0.256223412574146438,
                    0.0978694102142697230])
        a=b[:-1]+g
    elif name=='LDDC46':
        fullname='LDDC4()6[2R]'
        b=np.array([0.10893125722541,
                    0.13201701492152,
                    0.38911623225517,
                    -0.59203884581148,
                    0.47385028714844,
                    0.48812405426094])
        g=np.array([0.17985400977138,
                    0.14081893152111,
                    0.08255631629428,
                    0.65804425034331,
                    0.31862993413251])
        descript='2R Method of Calvo'
        a=b[:-1]+g
    elif name=='RK45[2R]C':
        fullname='RK4(3)5[2R+]C'
        descript='A 2R Method of Kennedy et. al.'
        regs=2
        a=np.array([970286171893./4311952581923,
                    6584761158862./12103376702013,
                    2251764453980./15575788980749,
                    26877169314380./34165994151039])
        b=np.array([1153189308089./22510343858157,
                    1772645290293./4653164025191,
                    -1672844663538./4480602732383,
                    2114624349019./3568978502595,
                    5198255086312./14908931495163])
        bhat=np.array([1016888040809./7410784769900,
                       11231460423587./58533540763752,
                      -1563879915014./6823010717585,
                       606302364029./971179775848,
                       1097981568119./3980877426909])
    elif name=='RK58[3R]C':
        fullname='RK5(4)8[3R+]C'
        descript='A 3R Method of Kennedy et. al.'
        regs=3
        a=np.array([[141236061735./3636543850841,
                    7367658691349./25881828075080,
                    6185269491390./13597512850793,
                    2669739616339./18583622645114,
                    42158992267337./9664249073111,
                    970532350048./4459675494195,
                    1415616989537./7108576874996], #1st subdiagonal
                  [-343061178215./2523150225462,
                     -4057757969325./18246604264081,
                      1415180642415./13311741862438,
                     -93461894168145./25333855312294,
                      7285104933991./14106269434317,
                     -4825949463597./16828400578907,0.]]) #2nd subdiagonal
        b=np.array([514862045033./4637360145389,
                    0.,
                    0.,
                    0.,
                    2561084526938./7959061818733,
                    4857652849./7350455163355,
                    1059943012790./2822036905401,
                    2987336121747./15645656703944])
        bhat=np.array([1269299456316./16631323494719,
                       0.,
                       2153976949307./22364028786708,
                       2303038467735./18680122447354,
                       7354111305649./15643939971922,
                       768474111281./10081205039574,
                       3439095334143./10786306938509,
                       -3808726110015./23644487528593])
    elif name=='RK59[2R]C':
        fullname='RK5(4)9[2R+]C'
        descript='A 2R Method of Kennedy et. al.'
        regs=2
        a=np.array([1107026461565./5417078080134,
         38141181049399./41724347789894,
         493273079041./11940823631197,
         1851571280403./6147804934346,
         11782306865191./62590030070788,
         9452544825720./13648368537481,
         4435885630781./26285702406235,
         2357909744247./11371140753790])
        b=np.array([ 2274579626619./23610510767302,
                     693987741272./12394497460941,
                   - 347131529483./15096185902911,
                     1144057200723./32081666971178,
                     1562491064753./11797114684756,
                     13113619727965./44346030145118,
                     393957816125./7825732611452,
                     720647959663./6565743875477,
                     3559252274877./14424734981077])
        bhat=np.array([ 266888888871./3040372307578,
                         34125631160./2973680843661,
                       - 653811289250./9267220972999,
                         323544662297./2461529853637,
                         1105885670474./4964345317203,
                         1408484642121./8758221613943,
                         1454774750537./11112645198328,
                         772137014323./4386814405182,
                         277420604269./1857595682219])
    return TwoRRungeKuttaMethod(a,b,bhat,regs,fullname,description=descript)
