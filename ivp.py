from numpy import exp, sin, cos

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
        ivp.u0=1
        ivp.rhs = lambda t,u: u
        ivp.exact = lambda t : ivp.u0*exp(t)
    elif ivpname=='nlsin':
        ivp.u0=1
        ivp.rhs = lambda t,u: 4.*u*float(sin(t))**3*cos(t)
        ivp.exact = lambda t: ivp.u0*exp((sin(t))**4)
    else: disp('Unknown IVP name; returning empty IVP')
    return ivp
