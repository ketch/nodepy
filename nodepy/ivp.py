"""
The principal objects in NodePy are ODE solvers. The object upon which a solver
acts is an initial value problem. Mathematically, an initial value problem
(IVP) consists of one or more ordinary differential equations and an initial
condition:

\\begin{align*} u(t) & = F(u) & u(0) & = u_0. \\end{align*}

This module implements the initial value problem as a class.

**Examples**::

    >>> from nodepy import ivp, rk
    >>> myivp = ivp.detest('A1')
    >>> myivp
    Problem Name:  A1
    Description:   Problem A1 of the non-stiff DETEST suite.

    # Integrate this problem with a Runge-Kutta method
    >>> rk4 = rk.loadRKM('RK44')
    >>> t,y = rk4(myivp)
    >>> y[-1][0] # doctest: +ELLIPSIS
    2.06115362252...e-09
"""
import numpy as np

class IVP(object):
    """
        In NodePy, an initial value problem is an object with the following
        properties:

            rhs: The right-hand-side function; i.e. F where u'=F(u).
            u0:  The initial condition.
            T:   The (default) final time of solution.

        Optionally an IVP may possess the following:
            exact: a function that takes one argument (t) and returns
                   the exact solution (Should we make this a function of
                   u0 as well?)
            dt0: The default initial timestep when a variable step size 
                 integrator is used.
            Any other problem-specific parameters.

    """
    def __init__(self,f=None,u0=1.,T=1.):
        self.u0  = u0
        self.rhs = f
        self.T   = T

    def __repr__(self):
        try:
            return 'Problem Name:  '+self.name+'\n'+'Description:   '+self.description
        except:
            try:
                return 'Problem Name: '+self.name
            except:
                return 'No name specified for this problem.'

def load_ivp(ivpname):
    """Load some very simple initial value problems."""
    import numpy as np

    ivp=IVP()
    if ivpname=='test':
        ivp.u0=1.
        ivp.rhs = lambda t,u: u
        ivp.exact = lambda t : ivp.u0*np.exp(t)
        ivp.T = 5.
        ivp.description = 'The linear scalar test problem'
    elif ivpname=='zoltan':
        ivp.u0=1.
        ivp.rhs = lambda t,u: -100*abs(u)
        ivp.exact = lambda t : ivp.u0*np.exp(-100*t)
        ivp.T = 10.
        ivp.description = 'The linear scalar test problem with abs value'
    elif ivpname=='nlsin':
        ivp.u0=1.
        ivp.rhs = lambda t,u: 4.*u*float(np.sin(t))**3*np.cos(t)
        ivp.exact = lambda t: ivp.u0*np.exp((np.sin(t))**4)
        ivp.T = 5.
        ivp.dt0=1.e-2
        ivp.description = 'A simple nonlinear scalar problem'
    elif ivpname=='ode1':
        ivp.u0=1.
        ivp.rhs = lambda t,u: 4.*t*np.sqrt(u)
        ivp.exact = lambda t: (1.+t**2)**2
        ivp.T = 5.
    elif ivpname=='ode2':
        ivp.u0=np.exp(1.)
        ivp.t0=0.5
        ivp.rhs = lambda t,u: u/t*np.log(u)
        ivp.exact = lambda t: np.exp(2.*t)
        ivp.T = 5.
    elif ivpname=='2odes':
        ivp.u0=np.array([1.,1.])
        ivp.rhs = lambda t,u: np.array([u[0], 2.*u[1]])
        ivp.exact = lambda t: np.array([np.exp(t), np.exp(2.*t)])
        ivp.T = 5.
    elif ivpname=='vdp':
        ivp.eps=0.1
        ivp.u0=np.array([2.,-0.65])
        ivp.rhs = lambda t,u: np.array([u[1], 1./ivp.eps*(-u[0]+(1.-u[0]**2)*u[1])])
        ivp.T = 5.
        ivp.dt0=1.e-2
        ivp.description = 'The van der Pol oscillator'
    else: print 'Unknown IVP name; returning empty IVP'
    ivp.name=ivpname
    return ivp

def detest(testkey):
    """
        Load problems from the non-stiff DETEST problem set.
        The set consists of six groups of problems, as follows:

            * A1-A5 -- Scalar problems
            * B1-B5 -- Small systems (2-3 equations)
            * C1-C5 -- Moderate size systems (10-50 equations)
            * D1-D5 -- Orbit equations with varying eccentricities
            * E1-E5 -- Second order equations
            * F1-F5 -- Problems with discontinuities
	    * SB1-SB3 -- Periodic Orbit problem from Shampine Baca paper pg.11,13

        .. note::
            Although this set of problems was not intended to become a
            standard, and although there are certain dangers in accepting
            any particular set of problems as a universal standard, it
            is nevertheless sometimes useful to try a new method on this
            test set due to the availability of published results for 
            many existing methods.
        
        Reference: [enright1987]_ 
    """
    import numpy as np
    ivp=IVP()
    if testkey=='A1':
        ivp.u0=1.
        ivp.T=20.
        ivp.rhs = lambda t,u: -u
        ivp.dt0 = 1.e-2
    elif testkey=='A2':
        ivp.u0=1.
        ivp.T=20.
        ivp.rhs = lambda t,u: -0.5*u**3
        ivp.dt0 = 1.e-2
    elif testkey=='A3':
        ivp.u0=1.
        ivp.T=20.
        ivp.rhs = lambda t,u: u*np.cos(t)
        ivp.dt0 = 1.e-2
    elif testkey=='A4':
        ivp.u0=1.
        ivp.T=20.
        ivp.rhs = lambda t,u: 0.25*u*(1.-0.05*u)
        ivp.dt0 = 1.e-5
    elif testkey=='A5':
        ivp.u0=4.
        ivp.T=20.
        ivp.rhs = lambda t,u: (u-t)/(u+t)
        ivp.dt0 = 1.e-2
    elif testkey=='B1':
        ivp.u0=np.array([1.,3.])
        ivp.T=20.
        ivp.rhs = lambda t,u: np.array([2.*(u[0]-u[0]*u[1]),-(u[1]-u[0]*u[1])])
        ivp.dt0 = 1.e-2
    elif testkey=='B2':
        ivp.u0=np.array([2.,0.,1.])
        ivp.T=20.
        ivp.rhs = lambda t,u: np.array([-u[0]+u[1],u[0]-2.*u[1]+u[2],u[1]-u[2]])
        ivp.dt0 = 1.e-2
    elif testkey=='B3':
        ivp.u0=np.array([1.,0.,0.])
        ivp.T=20.
        ivp.rhs = lambda t,u: np.array([-u[0],u[0]-u[1]**2,u[1]**2])
        ivp.dt0 = 1.e-2
    elif testkey=='B4':
        ivp.u0=np.array([3.,0.,0.])
        ivp.T=20.
        ivp.rhs = _B4rhs
        ivp.dt0 = 1.e-2
    elif testkey=='B5':
        ivp.u0=np.array([0.,1.,1.])
        ivp.T=20.
        ivp.rhs = lambda t,u: np.array([u[1]*u[2],-u[0]*u[2],-0.51*u[0]*u[1]])
        ivp.dt0 = 1.e-2
    elif testkey=='C1':
        ivp.u0=np.zeros(10); ivp.u0[0]=1.
        ivp.T=20.
        e=np.ones(10); e[-1]=0.
        ivp.L_rhs = np.diag(-e)+np.diag(e[:-1],-1);
        ivp.rhs = lambda t,u: np.dot(ivp.L_rhs,u)
        ivp.dt0 = 1.e-2
    elif testkey=='C2':
        ivp.u0=np.zeros(10); ivp.u0[0]=1.
        ivp.T=20.
        e=np.arange(1,11); e[-1]=0.
        ivp.L_rhs = np.diag(-e)+np.diag(e[:-1],-1);
        ivp.rhs = lambda t,u: np.dot(ivp.L_rhs,u)
        ivp.dt0 = 1.e-2    
    elif testkey=='C3':
        ivp.u0=np.zeros(10); ivp.u0[0]=1.
        ivp.T=20.
        e=np.ones(10)
        ivp.L_rhs = np.diag(-2*e)+np.diag(e[:-1],-1)+np.diag(e[:-1],1);
        ivp.rhs = lambda t,u: np.dot(ivp.L_rhs,u)
        ivp.dt0 = 1.e-2
    elif testkey=='C4':
        ivp.u0=np.zeros(51); ivp.u0[0]=1.
        ivp.T=20.
        e=np.ones(51)
        ivp.L_rhs = np.diag(-2*e)+np.diag(e[:-1],-1)+np.diag(e[:-1],1);
        ivp.rhs = lambda t,u: np.dot(ivp.L_rhs,u)
        ivp.dt0 = 1.e-2     
    elif testkey=='C5':
        ivp.u0=np.zeros(30)
	ivp.u0 = np.array([3.42947415189,3.35386959711,1.35494901715,6.64145542550,5.97156957878,2.18231499728,11.2630437207,14.6952576794,6.27960525067,-30.1552268759,1.65699966404,1.43785752721,-21.1238353380,28.4465098142,15.3882659679,-.557160570446,.505696783289,.230578543901,-.415570776342,.365682722812,.169143213293,-.325325669158,.189706021964,.0877265322780,-.0240476254170,-.287659532608,-.117219543175,-.176860753121,-.216393453025,-.0148647893090])	
        ivp.T=20.
        ivp.rhs = _C5rhs
        ivp.dt0 = 1.e-2
    elif testkey=='D1':
        eps=0.1
        ivp.u0=np.array([1.-eps,0.,0.,np.sqrt((1.+eps)/(1.-eps))])
        ivp.T=20.
        ivp.rhs = lambda t,u: np.array([u[2],u[3],-u[0]/(u[0]**2+u[1]**2)**1.5,-u[1]/(u[0]**2+u[1]**2)**1.5])
        ivp.dt0 = 1.e-2
    elif testkey=='D2':
        eps=0.3
        ivp.u0=np.array([1.-eps,0.,0.,np.sqrt((1.+eps)/(1.-eps))])
        ivp.T=20.
        ivp.rhs = lambda t,u: np.array([u[2],u[3],-u[0]/(u[0]**2+u[1]**2)**1.5,-u[1]/(u[0]**2+u[1]**2)**1.5])
        ivp.dt0 = 1.e-2
    elif testkey=='D3':
        eps=0.5
        ivp.u0=np.array([1.-eps,0.,0.,np.sqrt((1.+eps)/(1.-eps))])
        ivp.T=20.
        ivp.rhs = lambda t,u: np.array([u[2],u[3],-u[0]/(u[0]**2+u[1]**2)**1.5,-u[1]/(u[0]**2+u[1]**2)**1.5])
        ivp.dt0 = 1.e-2
    elif testkey=='D4':
        eps=0.7
        ivp.u0=np.array([1.-eps,0.,0.,np.sqrt((1.+eps)/(1.-eps))])
        ivp.T=20.
        ivp.rhs = lambda t,u: np.array([u[2],u[3],-u[0]/(u[0]**2+u[1]**2)**1.5,-u[1]/(u[0]**2+u[1]**2)**1.5])
        ivp.dt0 = 1.e-2
    elif testkey=='D5':
        eps=0.9
        ivp.u0=np.array([1.-eps,0.,0.,np.sqrt((1.+eps)/(1.-eps))])
        ivp.T=20.
        ivp.rhs = lambda t,u: np.array([u[2],u[3],-u[0]/(u[0]**2+u[1]**2)**1.5,-u[1]/(u[0]**2+u[1]**2)**1.5])
        ivp.dt0 = 1.e-2
    elif testkey=='E1':
        ivp.u0=np.array([0.6713967071418030,0.09540051444747446])
        ivp.T=20.
        ivp.rhs = lambda t,u: np.array([u[1],-(u[1]/(t+1.) + (1.-0.25/(t+1.)**2)*u[0])])
        ivp.dt0 = 1.e-2
    elif testkey=='E2':
        ivp.u0=np.array([2.,0.])
        ivp.rhs = lambda t,u: np.array([u[1],(1.-u[0]**2)*u[1]-u[0]])
        ivp.T=20.
        ivp.dt0 = 1.e-2
    elif testkey=='E3':
        ivp.u0=np.array([0.,0.])
        ivp.rhs = lambda t,u: np.array([u[1],u[0]**3 /6. - u[0] + 2.*np.sin(2.78535*t)])
        ivp.T=20.
        ivp.dt0 = 1.e-2
    elif testkey=='E4':
        ivp.u0=np.array([30.,0.])
        ivp.rhs = lambda t,u: np.array([u[1],0.032-0.4*u[1]**2])
        ivp.T=20.
        ivp.dt0 = 1.e-2
    elif testkey=='E5':
        ivp.u0=np.array([0.,0.])
        ivp.rhs = lambda t,u: np.array([u[1],np.sqrt(1.+u[1]**2)/(25.-t)])
        ivp.T=20.
        ivp.dt0 = 1.e-2
    elif testkey=='F1':
        ivp.u0=np.array([0.,0.])
        ivp.rhs = _F1rhs
        ivp.T=20.
        ivp.dt0 = 1.e-2
    elif testkey=='F2':
        ivp.u0=np.array([110.])
        ivp.rhs = _F2rhs
        ivp.T=20.
        ivp.dt0 = 1.e-2
    elif testkey=='F3':
        ivp.u0=np.array([0.,0.])
        ivp.rhs = lambda t,u: np.array([u[1],0.01*u[1]*(1.-u[0]**2)-u[0]-np.abs(np.sin(np.pi*t))])
        ivp.T=20.
        ivp.dt0 = 1.e-2
    elif testkey=='F4':
        ivp.u0=np.array([1.])
        ivp.rhs = _F4rhs
        ivp.T=20.
        ivp.dt0 = 1.e-2
    elif testkey=='F5':
        ivp.u0=np.array([1.])
        ivp.rhs = _F5rhs
        ivp.T=20.
        ivp.dt0 = 1.e-2
    elif testkey=='SB1':
        mu=0.01212856276531231
        mudash = 1.0-mu
        ivp.u0= np.array([1.2,0.,0.,-1.049357509830319])
        ivp.T=6.192169331319639
        ivp.rhs = lambda t,u: np.array([u[2],u[3],u[0]+2.*u[3]-mudash*(u[0]+mu)/((u[0]+mu)**2+u[1]**2)**1.5-mu*(u[0]-mudash)/((u[0]-mudash)**2+u[1]**2)**1.5,u[1]-2.*u[2]-mudash*(u[1])/((u[0]+mu)**2+u[1]**2)**1.5-mu*(u[1])/((u[0]-mudash)**2+u[1]**2)**1.5])
        ivp.dt0 = 1.e-1
    elif testkey=='SB2':
        mu=0.012277471
        mudash = 1.0-mu
        ivp.u0= np.array([0.994,0,0,-2.0317326955734])
        ivp.T=11.1243403372661
        ivp.rhs = lambda t,u: np.array([u[2],u[3],u[0]+2.*u[3]-mudash*(u[0]+mu)/((u[0]+mu)**2+u[1]**2)**1.5-mu*(u[0]-mudash)/((u[0]-mudash)**2+u[1]**2)**1.5,u[1]-2.*u[2]-mudash*(u[1])/((u[0]+mu)**2+u[1]**2)**1.5-mu*(u[1])/((u[0]-mudash)**2+u[1]**2)**1.5])
        ivp.dt0 = 1.e-1
    elif testkey=='SB3':
        mu=0.012277471
        mudash = 1.0-mu
        ivp.u0 = np.array([0.994,0.,0.,-2.11389879669450])
        ivp.T=5.43679543926019
        ivp.rhs = lambda t,u: np.array([u[2],u[3],u[0]+2.*u[3]-mudash*(u[0]+mu)/((u[0]+mu)**2+u[1]**2)**1.5-mu*(u[0]-mudash)/((u[0]-mudash)**2+u[1]**2)**1.5,u[1]-2.*u[2]-mudash*(u[1])/((u[0]+mu)**2+u[1]**2)**1.5-mu*(u[1])/((u[0]-mudash)**2+u[1]**2)**1.5])
        ivp.dt0 = 1.e-1

    else: raise Exception('Unknown Detest problem')
    ivp.name=testkey
    ivp.description='Problem '+testkey+' of the non-stiff DETEST suite.'
    return ivp

def _F1rhs(t,u):
    a=0.1
    du=np.zeros(2)
    du[0]=u[1]
    if np.mod(np.floor(t),2):     #Odd
      du[1] = 2.*a*u[1] - (np.pi**2+a**2)*u[0] - 1
    else:
      du[1] = 2.*a*u[1] - (np.pi**2+a**2)*u[0] + 1
    return du

def _F2rhs(t,u):
    du=np.zeros(1)
    if np.mod(np.floor(t),2):     #Odd
      du[0] = 55.-u[0]/2.
    else:
      du[0] = 55.-3.*u[0]/2.
    return du

def _F4rhs(t,u):
    du=np.zeros(1)
    if t<=10.:
      du[0] = -2./21 - 120.*(t-5.)/(1.+4.*(t-5.)**2)
    else:
      du[0] = -2.*u[0]
    return du

def _F5rhs(t,u):
    du=np.zeros(1)
    c=np.sum([i**(4./3) for i in range(1,20)])
    du[0] = 4./3/c * np.sum([(t-i+0j)**(4./3) for i in range(1,20)])*u[0]
    return du

def _B4rhs(t,u):
    du=np.zeros(3)
    du[0]=-u[1] - (u[0]*u[2])/np.sqrt(u[0]**2+u[1]**2)
    du[1]= u[0] - (u[1]*u[2])/np.sqrt(u[0]**2+u[1]**2)
    du[2]=        (u[0]     )/np.sqrt(u[0]**2+u[1]**2)
    return du
def _C5rhs(t,u):
    r = np.zeros(5); u1 = np.zeros(15); u2 = np.zeros(15); y1 = np.zeros((3,5)); y2 = np.zeros((3,5)); d = np.zeros((5,5)); du=np.zeros(30); ynew1 = np.zeros((3,5)); ynew2 = np.zeros((3,5));
    k2 = 2.95912208286; m0 = 1.00000597682; 
    m = np.array([.000954786104043,.000285583733151,.0000437273164546,.0000517759138449,.00000277777777778])
    k=0
    for j in range(0,5):
	for i in range(0,3):
	    y1[i][j] = u[k]
	    y2[i][j] = u[k+15]
	    k+=1

    for j in range(0,5):
	r[j] = np.sqrt(y1[0][j]**2+y1[1][j]**2+y1[2][j]**2)

    for k in range(0,5):
	for j in range(0,5):
	    d[k][j] = np.sqrt((y1[0][k]-y1[0][j])**2+(y1[1][k]-y1[1][j])**2+(y1[2][k]-y1[2][j])**2)
	      
    for j  in range(0,5):
	for i in range(0,3):
	    term=0.;
	    for k in range(0,5):
		if (k != j):
		    term = term+m[k]*((y1[i][k]-y1[i][j])/d[j][k]**3 - y1[i][k]/r[k]**3)
	    ynew1[i][j] = y2[i][j] 
	    ynew2[i][j] = k2*(-(m0+m[j])*y1[i][j]/r[j]**3+term)
    k=0
    for j in range(0,5):
	for i in range(0,3):
	    du[k] = ynew1[i][j]
	    du[k+15] = ynew2[i][j] 
	    k+=1
    
    return du

def detest_suite():
    """The entire non-stiff DETEST suite of problems."""
    detestkeys=['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5','D1','D2','D3','D4','D5','E1','E2','E3','E4','E5','F1','F2','F3','F4','F5']
    return [detest(dtkey) for dtkey in detestkeys]

def detest_suite_plus():
    """The entire non-stiff DETEST suite of problems plus the problems from Shampine-Baca paper. More problems can be added"""
    detestkeys=['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5','D1','D2','D3','D4','D5','E1','E2','E3','E4','E5','F1','F2','F3','F4','F5','SB1','SB2','SB3']
    return [detest(dtkey) for dtkey in detestkeys]

def detest_suite_minus():
    """The entire non-stiff DETEST suite of problems except the F (discontinous) problems."""
    detestkeys=['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5','D1','D2','D3','D4','D5','E1','E2','E3','E4','E5']
    return [detest(dtkey) for dtkey in detestkeys]


def detest_stiff(testkey):
    """The stiff DETEST suite.  Only a few problems have been implemented."""
    ivp=IVP()
    if testkey=='A1':
        ivp.u0=np.array([1.,1.,1.,1.])
        ivp.T=20.
        ivp.rhs = lambda t,u: np.array([-0.5*u[0], -u[1], -100.*u[2], -90.*u[3]])
        ivp.dt0 = 1.e-2
    elif testkey=='A2':
        ivp.u0=np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.])
        ivp.T=120.
        ivp.rhs = _A2rhs_stiff
        ivp.dt0 = 5.e-4
    elif testkey=='A3':
        ivp.u0=np.array([1.,1.,1.,1.])
        ivp.T=20.
        ivp.rhs = _A3rhs_stiff
        ivp.dt0 = 1.e-5
    elif testkey=='A4':
        ivp.u0=np.zeros(10)
        ivp.T=1.
        ivp.rhs = _A4rhs_stiff
        ivp.dt0 = 1.e-5
    else: print 'Unknown Detest problem; returning empty IVP'
    return ivp

def _A2rhs_stiff(t,u):
    du=np.zeros(9)
    du[0]=-1800.*u[0] + 900.*u[1]
    for i in range(1,8):
        du[i]=u[i-1]-2.*u[i]+u[i+1]
    du[8]=1000.*u[7]-2000.*u[8]+1000.
    return du

def _A3rhs_stiff(t,u):
    du=np.zeros(4)
    du[0]=-1.e4*u[0] + 100.*u[1] - 10.*u[2] + u[3]
    du[1]=-1.e3*u[1] + 10.*u[2] - 10.*u[3]
    du[2]=-u[2]+10.*u[3]
    du[3]=-0.1*u[3]
    return du

def _A4rhs_stiff(t,u):
    du=np.zeros(10)
    for i in range(10):
        du[i]=-(i+1)**5. * u[i]
    return du


if __name__ == "__main__":
    import doctest
    doctest.testmod()
