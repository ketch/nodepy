from numpy import exp, sin, cos

class IVP:
    """
        Class for ordinary differential equations initial value
        problems.
    """

# Some simple functions for convergence testing
class exp_fun(IVP):
    def __init__(self,u0=1):
        self.u0=u0
    def rhs(self,t,u):
        return u
    def exact(self,t):
        return self.u0*exp(t)

class nlsin_fun(IVP):
    def __init__(self,u0=1):
        self.u0=u0
    def rhs(self,t,u):
        return 4.*u*float(sin(t))**3*cos(t)
    def exact(self,t):
        return self.u0*exp((sin(t))**4)


