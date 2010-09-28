from numpy import exp, sin, cos, sqrt, log, array

class IVP:
    """
        Class for ordinary differential equations initial value
        problems.

        This is a pure virtual class, used only for inheritance.
        Any IVP should possess the following:

        rhs: The right-hand-side function; i.e. F where u'=F(u).
    """
    def __init__(self,u0=1,f=None):
        self.u0=u0
        self.rhs=f

def load_ivp(ivpname):
    ivp=IVP()
    if ivpname=='test':
        ivp.u0=1.
        ivp.rhs = lambda t,u: u
        ivp.exact = lambda t : ivp.u0*exp(t)
    elif ivpname=='nlsin':
        ivp.u0=1.
        ivp.rhs = lambda t,u: 4.*u*float(sin(t))**3*cos(t)
        ivp.exact = lambda t: ivp.u0*exp((sin(t))**4)
    elif ivpname=='ode1':
        ivp.u0=1.
        ivp.rhs = lambda t,u: 4.*t*sqrt(u)
        ivp.exact = lambda t: (1.+t**2)**2
    elif ivpname=='ode2':
        ivp.u0=exp(1.)
        ivp.t0=0.5
        ivp.rhs = lambda t,u: u/t*log(u)
        ivp.exact = lambda t: exp(2.*t)
    elif ivpname=='2odes':
        ivp.u0=array([1.,1.])
        ivp.rhs = lambda t,u: array([u[0], 2.*u[1]])
        ivp.exact = lambda t: array([exp(t), exp(2.*t)])
    elif ivpname=='vdp':
        ivp.eps=0.1
        ivp.u0=array([2.,-0.65])
        ivp.rhs = lambda t,u: array([u[1], 1./ivp.eps*(-u[0]+(1.-u[0]**2)*u[1])])
    else: print 'Unknown IVP name; returning empty IVP'
    return ivp
