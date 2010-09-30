"""
Functions for running convergence and performance tests.
"""
import pylab as pl
import numpy as np
from NodePy import runge_kutta_method as rk

def ctest(methods,ivp,grids=[20,40,80,160,320,640]):
    """
        Runs a convergence test and creates a plot of the results.

        INPUTS:
            methods -- a list of ODEsolver instances
            ivp     -- an IVP instance
            grids   -- a list of grid sizes for integration.
                       optional; defaults to [20,40,80,160,320,640]

        EXAMPLES:

            import runge_kutta_method as rk
            from ivp import *
            rk44=rk.loadRKM('RK44')
            myivp=nlsin_fun()
            ctest(methods,myivp)

            TODO: 
                - Option to plot versus f-evals or dt

    """
    pl.clf(); pl.draw(); pl.hold(True)
    # In case just one method is passed in (and not as a list):
    if not isinstance(methods,list): methods=[methods]
    err=[]
    try:
        exsol = ivp.exact(ivp.T)
    except:
        bs5=rk.loadRKM('BS5')
        bigN=grids[-1]*4
        print 'solving on fine grid with '+str(bigN)+' points'
        t,u=bs5(ivp.rhs,ivp.u0,ivp.T,N=grids[-1]*4)
        print 'done'
        exsol = u[-1].copy()
    for method in methods:
        print method.name
        err0=[]
        for i,N in enumerate(grids):
            t,u=method(ivp.rhs,ivp.u0,ivp.T,N=N)
            err0.append(np.linalg.norm(u[-1]-exsol))
        err.append(err0)
        work=[grid*len(method) for grid in grids]
        pl.loglog(work,err0,label=method.name,linewidth=3)
        pl.xlabel('Function evaluations')
        pl.ylabel('Error at $t_{final}$')
    pl.legend()
    pl.hold(False)
    pl.draw()
    pl.ioff()
    return err


def ptest(methods,ivps,tols=[1.e-1,1.e-2,1.e-3,1.e-4]):
    """
        Runs a performance test and creates a plot of the results.

        INPUTS:
            methods -- a list of ODEsolver instances
                       Note that all methods must have error estimators.
            ivps    -- a list of IVP instances
            tols    -- a specified list of error tolerances (optional)

        EXAMPLES:

            import runge_kutta_method as rk
            from ivp import *
            bs5=rk.loadRKM('BS5')
            myivp=nlsin_fun()
            ptest(methods,myivp,T)

    """
    pl.clf(); pl.draw(); pl.hold(True)
    # In case just one method is passed in (and not as a list):
    if not isinstance(methods,list): methods=[methods]
    if not isinstance(ivps,list): ivps=[ivps]
    err=np.ones([len(methods),len(tols)])
    work=np.zeros([len(methods),len(tols)])
    for ivp in ivps:
        try:
            exsol = ivp.exact(ivp.T)
        except:
            bs5=rk.loadRKM('BS5')   #Use Bogacki-Shampine RK for fine solution
            lowtol=min(tols)/100.
            print 'solving for "exact" solution with tol= '+str(lowtol)
            t,u=bs5(ivp.rhs,ivp.u0,ivp.T,errtol=lowtol,dt=ivp.dt0)
            print 'done'
            exsol = u[-1].copy()
        for imeth,method in enumerate(methods):
            print 'Solving with method '+method.name
            for jtol,tol in enumerate(tols):
                t,u=method(ivp.rhs,ivp.u0,ivp.T,errtol=tol,dt=ivp.dt0)
                err[imeth,jtol]*= np.max(np.abs(u[-1]-exsol))
                work[imeth,jtol]+= len(t)*len(method)
    for imeth,method in enumerate(methods):
        for jtol,tol in enumerate(tols):
            err[imeth,jtol]=err[imeth,jtol]**(1./len(ivps))
    for imeth,method in enumerate(methods):
        pl.loglog(work[imeth,:],err[imeth,:],label=method.name,linewidth=3)
    pl.xlabel('Function evaluations')
    pl.ylabel('Error at $t_{final}$')
    pl.legend(loc='best')
    pl.hold(False)
    pl.draw()
    pl.ioff()
    return work,err
