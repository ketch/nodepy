"""
Script for running convergence tests.
"""
import pylab as pl

def ctest(methods,ode,T,grids=[20,40,80,160,320,640]):
    """
        Runs a convergence test and creates a plot of the results.

        INPUTS:
            methods -- a list of ODEsolver instances
            ode     -- an ODE instance
            T       -- a list of two numbers indicating the
                       initial and final times of integration.
                       If the first is omitted, it is assumed zero.
            grids   -- a list of grid sizes for integration.
                       optional; defaults to [20,40,80,160,320,640]

        EXAMPLES:

            sage: import runge_kutta_method as rk
            sage: from ode import *
            sage: rk44=rk.loadRKM('RK44')
            sage: myode=nlsin_fun()
            sage: T=[0.,5.]
            sage: ctest(methods,myode,T)

    """
    pl.clf()
    pl.draw()
    pl.hold(True)
    if not isinstance(methods,list): methods=[methods]
    err=[]
    for method in methods:
        err0=[]
        for i in range(len(grids)):
            N=grids[i]
            t,u=method(ode.rhs,ode.u0,T,N=N)
            err0.append(abs(u[-1]-ode.exact(t[-1])))
        err.append(err0)
        work=[grid*len(method) for grid in grids]
        pl.loglog(work,err0,label=method.name)
        pl.xlabel('Function evaluations')
        pl.ylabel('Error at $t_{final}$')
    pl.legend()
    pl.hold(False)
    pl.draw()
    pl.ioff()
    return err
