from __future__ import print_function
from __future__ import division

from __future__ import absolute_import
import numbers
from six.moves import range

#=====================================================
class ODESolver(object):
#=====================================================
    """ Top-level class for numerical ODE solvers """
    def __init__(self):
        pass

    def __step__(self):
        raise NotImplementedError

    def __call__(self,ivp,t0=0,N=5000,dt=None,errtol=None,controllertype='P',
                 x=None,diagnostics=False,use_butcher=False,max_steps=7500):
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
                  :cite:`hairer1993` for details.
                - diagnostics -- if True, return the number of rejected steps
                  and a list of step sizes used, in addition to the solution
                  values and times.

            **Output**:
                - t -- A list of solution times
                - u -- A list of solution values

            If ivp.T is a scalar, the solution is integrated to that time
            and all output step values are returned.  If ivp.T is a list/array,
            the solution is integrated to T[-1] and the solution is returned
            for the times specified in T.

            TODO:

                * Implement an option to not keep all output (for efficiency).
                * Option to keep error estimate history
        """
        # If a list of output times is specified and the method supports dense output,
        # then we don't have to stop exactly at the output times.
        if hasattr(self, 'b_dense') and not isinstance(ivp.T, numbers.Number):
            dense_output = True
        else:
            dense_output = False

        numself = self.__num__()
        f=ivp.rhs; u0=ivp.u0; T=ivp.T
        u=[u0]; t=[t0];
        dthist=[]; errest_hist=[]
        uu = u0
        rejected_steps=0

        if isinstance(T, numbers.Number):
            t_final = T
            t_out = None
        else:
            t_final = T[-1]
            t_out = T

        if t_out is not None:
            iout = 0
            next_output_time = T[iout]
        else:
            next_output_time = t_final

        t_current = t0

        out_now = False

        if not hasattr(self, 'b_dense'):
            if errtol is None:      # Fixed-timestep mode
                if dt is None: dt = (t_final-t0)/float(N)
                dt_standard = dt + 0
                max_steps = max(max_steps, int(round(2*(t_final-t0)/dt)))
                for istep in range(max_steps):

                    if t_current+dt >= next_output_time:
                        dt = next_output_time - t_current
                        out_now = True

                    if not hasattr(self, 'b_dense'):
                        uu = numself.__step__(f,t_current,uu,dt,x=x,use_butcher=use_butcher)
                    else:
                        uu, _ = numself.__step__(f,t_current,uu,dt,[],x=x,use_butcher=use_butcher)

                    t_current += dt
                    if (out_now) or (t_out is None):
                        u.append(uu)
                        t.append(t_current)
                    if t_current >= t_final: break
                    if out_now:
                        iout += 1
                        next_output_time = T[iout]
                        dt = dt_standard

                    out_now = False

            else:                   # Error-control mode
                p=self.embedded_method.p
                alpha = 0.7/p; beta = 0.4/p; kappa = 0.9
                facmin = 0.2; facmax = 5.0
                errestold = errtol
                errest=1.


                for istep in range(max_steps):
                    # Hit next output time exactly:
                    if t_current+dt >= next_output_time: 
                        dt = next_output_time - t_current
                        out_now = True

                    unew,errest = numself.__step__(f,t_current,uu,dt,estimate_error=True,x=x,use_butcher=use_butcher)

                    if errest<=errtol:      # Step accepted
                        t_current += dt
                        if (out_now) or (t_out is None):
                            u.append(unew)
                            t.append(t_current)
                            uu = unew.copy()
                        # Stop if final time reached:
                        if t_current >= t_final: break
                        if out_now:
                            iout += 1
                            next_output_time = T[iout]
                        errestold = errest #Should this happen if step is rejected?
                        dthist.append(dt)
                        errest_hist.append(errest)
                    else: 
                        rejected_steps+=1

                    out_now = False

                    if controllertype=='P':
                      #Compute new dt using P-controller
                      facopt = (errtol/(errest+1.e-6*errtol))**alpha 

                    elif controllertype=='PI':
                      #Compute new dt using PI-controller
                      facopt = ((errtol/errest)**alpha
                                *(errestold/errtol)**beta)
                    else: print('Unrecognized time step controller type')

                    # Set new step size
                    dt = dt * min(facmax,max(facmin,kappa*facopt))

                if istep==max_steps-1:
                    print('Maximum number of steps reached; giving up.')

        else:  # dense output
            if errtol is None:      # Fixed-timestep mode
                if dt is None: dt = (t_final-t0)/float(N)
                dt_standard = dt + 0
                for istep in range(max_steps):

                    thetas = []
                    if t_current+dt >= next_output_time:
                        out_now = True
                        while next_output_time <= t_current + dt:
                            thetas.append( (next_output_time - t_current)/dt )
                            iout += 1
                            if iout >= len(T): break
                            next_output_time = T[iout]

                    uu, output = numself.__step__(f,t_current,uu,dt,thetas,
                                                  x=x,use_butcher=use_butcher)

                    if output:
                        for i, outsol in enumerate(output):
                            u.append(outsol)
                            t.append(t_current + dt*thetas[i])
                    t_current += dt
                    if t_current >= t_final: break

                    out_now = False

            else:                   # Error-control mode
                raise NotImplementedError

        if diagnostics==False:
            return t, u
        else:
            if errtol is None:
                return t,u,rejected_steps,dthist
            else:
                return t,u,rejected_steps,dthist,errest_hist
