#=====================================================
class ODESolver:
#=====================================================
    """ Top-level class for numerical ODE solvers """
    def __init__(self):
        pass

    def __call__(self,ivp,t0=0,N=100,dt=None,errtol=None,x=None):
        """
            Calling an ODESolver numerically integrates the ODE
            u'(t) = f(t,u(t)) with initial value u(0)=u0 from time
            t0 up to time T using the solver.

            The timestep can be controlled in any of three ways:

                1. By specifying N, the total number of steps to take.
                   Then dt = (T-t0)/N.
                2. By specifying dt directly.
                3. For methods with an error estimate (e.g., RK pairs),
                   by specifying an error tolerance.  Then the step
                   size is adjusted using a PI-controller to achieve
                   the requested tolerance.  In this case, dt should
                   also be specified and determines the value of the
                   initial timestep.

            The argument x is used to pass any additional arguments required
            for the RHS function f.

            TODO: 

                * Implement an option to not keep all output (for efficiency).
                * Option to keep timestep history
                * Option to keep error estimate history
        """
        f=ivp.f; u0=ivp.u0; T=ivp.T
        u=[u0]; t=[t0]

        if errtol is None:      # Fixed-timestep mode
            if dt is None: dt=(T-t0)/N
            while t[-1]<T:
                if t[-1]+dt>T: dt=T-t[-1]
                if x is not None: u.append(self.__step__(f,t,u,dt,x=x))
                else: u.append(self.__step__(f,t,u,dt))
                t.append(t[-1]+dt)

        else:                   # Error-control mode
            p=self.embedded_method.order()
            alpha = 0.7/p; beta = 0.4/p; kappa = 0.9
            errestold = errtol
            errest=1.

            if dt is None: print 'ERROR: Must specify initial timestep for error-control mode'
            while t[-1]<T:
                if t[-1]+dt>T: dt=T-t[-1]
                unew,errest = self.__step__(f,t,u,dt,errest=True,x=x)
                if errest<=errtol:      # Step accepted
                    u.append(unew)
                    t.append(t[-1]+dt)
                    errestold = errest  #Should this happen if step is rejected?
                else: print 'step rejected'

                #Compute new dt using PI-controller
                dt = kappa*dt*(errtol/errest)**alpha * (errestold/errtol)**beta

        return t,u
