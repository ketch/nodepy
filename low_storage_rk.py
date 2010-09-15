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
class TwoRRungeKuttaMethod(ExplicitRungeKuttaPair):
#=====================================================
    """
        Class for 2R/3R/4R low-storage Runge-Kutta methods.
        (van der Houwen, Wray, Kennedy)
    """
    def __init__(self,a,b,bhat,regs=2,
            name='2R Runge-Kutta Method',description=''):
        """
            Initializes the 2R method by storing the
            low-storage coefficients and computing the Butcher
            array.
        """
        self.b=b
        self.a=a
        self.bhat=bhat
        m=len(b)
        self.A=np.zeros([m,m])
        for i in range(1,m):
            for j in range(i-2):
                self.A[i,j]=b[j]
            if regs==2:
                self.A[i,i-1]=a[i-1]
            elif regs==3:
                self.A[i  ,i-1]=a[0,i-1]
                if i<m-1:
                    self.A[i+1,i-1]=a[1,i-1]
            elif regs==4:
                #NEED TO FILL IN
                pass
        self.c=np.sum(self.A,1)
        self.embmeth=ExplicitRungeKuttaMethod(self.A,self.bhat)
        self.name=name
        self.info=description
        self.lstype='2R+_pair'

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
            self.A,self.b=shu_osher_to_butcher(alpha,beta)
            self.A=np.tril(self.A,-1)
            print self.A
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
            self.A,self.b=shu_osher_to_butcher(alpha,beta)
            self.A=np.tril(self.A,-1)
            print self.A
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
            Need to add initialization for 3S*emb methods.
        """
        self.name=name
        self.info=description

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
            self.A,self.b=shu_osher_to_butcher(alpha,beta)
            self.A=np.tril(self.A,-1)
            self.c=np.sum(self.A,1)
            self.bhat=np.dot(delta,np.vstack([self.A,self.b]))/sum(delta)

        elif type=='3S*_pair':
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
            self.A=np.tril(self.A,-1)
            self.c=np.sum(self.A,1)
            self.bhat=np.dot(delta[:m+1],np.vstack([self.A,self.b]))/sum(delta)
        self.embmeth=ExplicitRungeKuttaMethod(self.A,self.bhat)

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
    elif lstype=='3S*': m=int((len(coeff)+6.)/4.)
    elif lstype=='3S*_pair': m=int((len(coeff)+3.)/4.)
    beta=[0.]
    for i in range(m): beta.append(coeff[2*m-3+i])
    gamma=[[0.],[0.,1.]+coeff[0:m-1]]
    if lstype.startswith('3S*'): gamma.append([0,0,0,0]+coeff[3*m-3:4*m-6])
    if lstype=='2S' or lstype=='3S*':  delta=[1.]+coeff[m-1:2*m-3]+[0.]
    elif lstype=='2S*': delta=[1.]+[0.]*len(range(m-1,2*m-3))+[0.]
    elif lstype=='2S_pair': delta=[1.]+     coeff[m-1:2*m-3] +[coeff[-2],coeff[-1]]
    elif lstype=='3S*_pair': delta=[1.]+     coeff[m-1:2*m-3] +[coeff[-3],coeff[-2],coeff[-1]]
    if lstype=='2S' or lstype=='2S*': 
        for i in range(1,m+1): gamma[0].append(1.-gamma[1][i]*sum(delta[0:i]))
        meth = LowStorageRungeKuttaMethod(beta,gamma,delta,lstype)
    elif lstype=='2S_pair':
        for i in range(1,m+1): gamma[0].append(1.-gamma[1][i]*sum(delta[0:i]))
        meth = LowStorageRungeKuttaPair(beta,gamma,delta,lstype)
    elif lstype.startswith('3S*'):
        for i in range(1,m+1): gamma[0].append(1.-gamma[2][i]
                                        -gamma[1][i]*sum(delta[0:i]))
        if lstype=='3S*':
            meth = LowStorageRungeKuttaMethod(beta,gamma,delta,lstype)
        elif lstype=='3S*_pair':
            meth = LowStorageRungeKuttaPair(beta,gamma,delta,lstype)
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
        Loads 2R low-storage methods.

        TODO: fix broken methods below.
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
    return TwoRRungeKuttaMethod(a,b,bhat,regs,fullname,description=descript)
