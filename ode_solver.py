#=====================================================
class ODESolver:
#=====================================================
    """ Top-level class for numerical ODE solvers """
    def __init__(self):
        pass

    def __call__(self,f,u0,T,N=100,dt=None,x=None):
        """
            Calling an ODESolver numerically integrates the function 
            f(t,u(t)) up to t=T using the solver.
        """
        u=[u0]
        if not isinstance(T,list): T=[T]
        if len(T)==2: t0=T[0]
        else: t0=0
        t=[t0]
        t1=T[-1]
        if dt is None: dt=(t1-t0)/N
        while abs(t[-1]+dt-t1)>1.e-13: 
            if x is not None: u.append(self.__step__(f,t,u,dt,x=x))
            else: u.append(self.__step__(f,t,u,dt))
            t.append(t[-1]+dt)
        dt_final=t1-t[-1]
        if x is not None: u.append(self.__step__(f,t,u,dt_final,x=x))
        else: u.append(self.__step__(f,t,u,dt_final))
        t.append(t[-1]+dt)
        return t,u
