"""
Class for low-storage Runge-Kutta methods, and various functions related to
them.

**Author**: David Ketcheson (10-07-2008)

**Examples**::

    >>>

**To Do**:
    - Add 2N (Williamson) methods
"""
from runge_kutta_method import *

#=====================================================
class TwoRRungeKuttaMethod(ExplicitRungeKuttaMethod):
#=====================================================
    """
        Class for 2R low-storage Runge-Kutta methods.
        (van der Houwen)
    """
    def __init__(self,a,b,
            name='2R Runge-Kutta Method',description=''):
        """
            Initializes the 2R method by storing the
            low-storage coefficients and computing the Butcher
            array.
        """
        self.b=b
        self.a=a
        m=len(b)
        self.A=np.zeros([m,m])
        for i in range(1,m):
            for j in range(i-1):
                self.A[i,j]=b[j]
            self.A[i,i-1]=a[i-1]
        self.c=np.sum(self.A,1)
        self.name=name
        self.info=description
        self.lstype='2R'

    def __step__(self,f,t,u,dt):
        """
            Take a time step on the ODE u'=f(t,u).

            INPUT:
                f  -- function being integrated
                t  -- array of previous solution times
                u  -- array of previous solution steps
                        (u[i,:] is the solution at time t[i])
                dt -- length of time step to take

            OUTPUT:
                unew -- approximate solution at time t[-1]+dt

            The implementation here is special for 2R low-storage methods
            But it's not really ultra-low-storage yet.
        """
        m=len(self)
        S2=u[-1]+0.
        S1=u[-1]+0. # by adding zero we get a copy; is there a better way?
        S1=f(t[-1],S1)
        S2=S2+self.b[0]*dt*S1
        for i in range(1,m):
            S1 = S2 + (self.a[i-1]-self.b[i-1])*dt*S1
            S1=f(t[-1]+self.c[i]*dt,S1)
            S2=S2+self.b[i]*dt*S1
        return S2

#=====================================================
# End of class TwoRRungeKuttaMethod
#=====================================================


#=====================================================
class LowStorageRungeKuttaMethod(ExplicitRungeKuttaMethod):
#=====================================================
    """
        Class for low-storage Runge-Kutta methods
        that use Ketcheson's assumption (2S, 2S*, and 3S* methods).
    """
    def __init__(self,betavec,gamma,delta,type,
            name='Low-storage Runge-Kutta Method',description=''):
        """
            Initializes the low-storage method by storing the
            low-storage coefficients and computing the Butcher
            coefficients.
        """
        self.betavec=betavec
        self.gamma=gamma
        self.delta=delta
        self.lstype=type
        if type=='2S' or type=='2S*':
            m=len(betavec)-1
            alpha=np.zeros([m+1,m])
            beta =np.zeros([m+1,m])
            for i in range(0,m): beta[i+1,i] = betavec[i+1]
            for i in range(1,m):
                beta[ i+1,i-1] = -beta[i,i-1]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i-1] = -gamma[0][i]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i  ] = 1. - alpha[i+1,i-1]
            self.A,self.b=ShuOsher2Butcher(alpha,beta)
            self.c=np.sum(self.A,1)
        elif type.startswith('3S*'):
            m=len(betavec)-1
            alpha=np.zeros([m+1,m])
            beta =np.vstack([np.zeros(m),np.diag(betavec[1:])])
            for i in range(1,m):
                beta[ i+1,i-1] = -beta[i,i-1]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,  0] =  gamma[2][i+1]-gamma[2][i]* \
                                    gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i-1] = -gamma[0][i]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i  ] = 1. - alpha[i+1,i-1]-alpha[i+1,0]
            self.A,self.b=ShuOsher2Butcher(alpha,beta)
            self.c=np.sum(self.A,1)
        self.name=name
        self.info=description

    def __step__(self,f,t,u,dt):
        """
            Take a time step on the ODE u'=f(t,u).

            INPUT:
                f  -- function being integrated
                t  -- array of previous solution times
                u  -- array of previous solution steps
                        (u[i,:] is the solution at time t[i])
                dt -- length of time step to take

            OUTPUT:
                unew -- approximate solution at time t[-1]+dt

            The implementation here is special for 2S low-storage methods
            But it's not really ultra-low-storage yet.
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
# End of class LowStorageRungeKuttaMethod
#=====================================================

#=====================================================
class LowStorageRungeKuttaPair(ExplicitRungeKuttaPair):
#=====================================================
    """
        Class for low-storage embedded Runge-Kutta pairs.
    """
    def __init__(self,betavec,gamma,delta,type,
            name='Low-storage Runge-Kutta Pair',description=''):
        """
            Initializes the low-storage pair by storing the
            low-storage coefficients and computing the Butcher
            coefficients.
        """
        self.betavec=betavec
        self.gamma=gamma
        self.delta=delta
        self.lstype=type
        if type=='2S_pair':
            m=len(betavec)-1
            alpha=np.zeros([m+1,m])
            beta =np.zeros([m+1,m])
            for i in range(0,m): beta[i+1,i] = betavec[i+1]
            for i in range(1,m):
                beta[ i+1,i-1] = -beta[i,i-1]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i-1] = -gamma[0][i]*gamma[1][i+1]/gamma[1][i]
                alpha[i+1,i  ] = 1. - alpha[i+1,i-1]
            self.A,self.b=ShuOsher2Butcher(alpha,beta)
            self.c=np.sum(self.A,1)
            self.bhat=np.dot(delta,np.vstack([self.A,self.b]))/sum(delta)
        self.name=name
        self.info=description

    def __step__(self,f,t,u,dt):
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
        for i in range(1,m+1):
            S2 = S2 + self.delta[i-1]*S1
            S1 = self.gamma[0][i]*S1 + self.gamma[1][i]*S2 \
                 + self.betavec[i]*dt*f(t[-1]+self.c[i-1]*dt,S1)
        S2=1./sum(delta[1:m+1])*(S2+delta[m+1]*S1)
        return S1,abs(S1-S2)

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
        gamma=np.zeros([p,len(self)])
        for j in range(1,p):
            gamma[j,:]=(c**j/j-np.dot(A,c**(j-1)))/factorial(j-1)
        for i in range(len(code)):
            exec('z[i]='+code[i])
        z[-1]=np.dot(b,c**(p-1))-1./p
        return z



#=====================================================
#End of LowStorageRungeKuttaPair class
#=====================================================

def load_LSRK(file,lstype='2S'):
    f=open(file,'r')
    coeff=[]
    for line in f:
        coeff.append(float(line))
    if lstype=='2S' or lstype=='2S*': m=int(len(coeff)/3+1) # Number of stages
    elif lstype=='2S_pair': m=int((len(coeff)+1)/3)
    elif lstype.startswith('3S*'): m=int((len(coeff)+6.)/4.)
    beta=[0.]
    for i in range(m): beta.append(coeff[2*m-3+i])
    gamma=[[0.],[0.,1.]+coeff[0:m-1]]
    if lstype.startswith('3S*'): gamma.append([0,0,0,0]+coeff[3*m-3:4*m-6])
    if lstype=='2S' or lstype.startswith('3S*'):  delta=[1.]+coeff[m-1:2*m-3]+[0.]
    elif lstype=='2S*': delta=[1.]+[0.]*len(range(m-1,2*m-3))+[0.]
    elif lstype=='2S_pair': delta=[1.]+     coeff[m-1:2*m-3] +[coeff[-2],coeff[-1]]
    if lstype=='2S' or lstype=='2S*': 
        for i in range(1,m+1): gamma[0].append(1.-gamma[1][i]*sum(delta[0:i]))
        meth = LowStorageRungeKuttaMethod(beta,gamma,delta,lstype)
    elif lstype=='2S_pair':
        for i in range(1,m+1): gamma[0].append(1.-gamma[1][i]*sum(delta[0:i]))
        meth = LowStorageRungeKuttaPair(beta,gamma,delta,lstype)
    elif lstype.startswith('3S*'):
        for i in range(1,m+1): gamma[0].append(1.-gamma[2][i]
                                        -gamma[1][i]*sum(delta[0:i]))
        meth = LowStorageRungeKuttaMethod(beta,gamma,delta,lstype)
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

def load_2R(name):
    """
        Loads the 2R method of Tselios & Simos (2007).
    """
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
        a=np.array([970286171893./4311952581923,
                    6584761158862./12103376702013,
                    2251764453980./15575788980749,
                    26877169314380./34165994151039])
        b=np.array([1153189308089./22510343858157,
                    1772645290293./4653164025191,
                    -1672844663538./4480602732383,
                    2114624349019./3568978502595,
                    5198255086312./14908931495163])
    return TwoRRungeKuttaMethod(a,b,fullname,description=descript)
