#=====================================================
class ODESolver:
#=====================================================
    """ Top-level class for numerical ODE solvers """
    def __init__(self):
        pass

    def __step__(self):
        raise NotImplementedError

    def __call__(self,ivp,t0=0,N=5000,dt=None,errtol=None,controllertype='P',x=None,diagnostics=False):
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

            **Input**:
                - ivp -- An IVP instance (the initial value problem to be solved)
                - t0 -- The initial time from which to integrate
                - N -- The # of steps to take (using a fixed step size)
                - dt -- The step size to use
                - errtol -- The local error tolerance to be observed (using
                  adaptive stepping).  This requires that the method have an
                  error estimator, such as an embedded Runge-Kutta method.
                - controllerType -- The type of adaptive step size control
                  to be used; available options are 'P' and 'PI'.  See
                  [hairer1993b]_ for details.
                - diagnostics -- if True, return the number of rejected steps
                  and a list of step sizes used, in addition to the solution
                  values and times.

            **Output**:
                - t -- A list of solution times
                - u -- A list of solution values

            TODO: 

                * Implement an option to not keep all output (for efficiency).
                * Option to keep error estimate history
        """
        numself = self.__num__()
        f=ivp.rhs; u0=ivp.u0; T=ivp.T
        u=[u0]; t=[t0]; dthist=[]
        rejected_steps=0

        if errtol is None:      # Fixed-timestep mode
            if dt is None: dt=(T-t0)/N
            while t[-1]<T:
                if t[-1]+dt>T: dt=T-t[-1]
                if x is not None: u.append(numself.__step__(f,t,u,dt,x=x))
                else: u.append(numself.__step__(f,t,u,dt))
                t.append(t[-1]+dt)

        else:                   # Error-control mode
            p=numself.embedded_method.order()
            alpha = 0.7/p; beta = 0.4/p; kappa = 0.9
            facmin = 0.2; facmax = 5.
            errestold = errtol
            errest=1.

            if dt is None: print 'ERROR: Must specify initial timestep for error-control mode'
            while t[-1]<T:
                if t[-1]+dt>T: dt=T-t[-1]
                unew,errest = numself.__step__(f,t,u,dt,errest=True,x=x)
                if errest<=errtol:      # Step accepted
                    u.append(unew)
                    t.append(t[-1]+dt)
                    errestold = errest  #Should this happen if step is rejected?
                    dthist.append(dt)
                else: rejected_steps+=1

                if controllertype=='P':
                  #Compute new dt using P-controller
                  facopt = (errtol/errest)**alpha 

                elif controllertype=='PI':
                  #Compute new dt using PI-controller
                  facopt = ((errtol/errest)**alpha
                            *(errestold/errtol)**beta)
                else: print 'Unrecognized time step controller type'

                dt = dt * min(facmax,max(facmin,kappa*facopt))

        if diagnostics==False: return t,u
        else: return t,u,rejected_steps,dthist
