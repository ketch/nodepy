"""
Functions for running convergence and performance tests.

**Examples**::

    >>> from nodepy import rk, convergence, ivp

    # Load some methods:
    >>> rk4=rk.loadRKM('RK44')
    >>> SSP2=rk.loadRKM('SSP22')
    >>> SSP104=rk.loadRKM('SSP104')

    # Define an initial value problem:
    >>> myivp=ivp.load_ivp('test')

    >>> work, error = convergence.ctest([rk4,SSP2,SSP104],myivp)
"""
import numpy as np
import runge_kutta_method as rk

def ctest(methods,ivp,grids=[20,40,80,160,320,640],verbosity=0,parallel=False):
    """
        Runs a convergence test, integrating a single initial value problem
        using a sequence of fixed step sizes and a set of methods.
        Creates a plot of the resulting errors versus step size for each method.

        **Inputs**:
            - methods -- a list of ODEsolver instances
            - ivp     -- an IVP instance
            - grids   -- a list of grid sizes for integration.
                      optional; defaults to [20,40,80,160,320,640]
            - parallel -- to exploit possible parallelization (optional)

        **Example**::

            >>> import runge_kutta_method as rk
            >>> from ivp import load_ivp
            >>> rk44=rk.loadRKM('RK44')
            >>> myivp=load_ivp('nlsin')
            >>> work, err=ctest(rk44,myivp)

        TODO: 
            - Option to plot versus f-evals or dt

    """
    import matplotlib.pyplot as pl
    pl.clf(); pl.hold(True)
    # In case just one method is passed in (and not as a list):
    if not isinstance(methods,list): methods=[methods]
    err=[]
    try:
        exsol = ivp.exact(ivp.T)
    except:
        bs5=rk.loadRKM('BS5')
        bigN=grids[-1]*4
        if verbosity>0: print 'solving on fine grid with '+str(bigN)+' points'
        t,u=bs5(ivp,N=bigN)
        if verbosity>0: print 'done'
        exsol = u[-1] + 0.
    for method in methods:
        if verbosity>0: print "solving with %s" % method.name
        err0=[]
        for i,N in enumerate(grids):
            t,u=method(ivp,N=N)
            err0.append(np.linalg.norm(u[-1]-exsol))
        err.append(err0)
        work=np.array([grid*len(method) for grid in grids])
	if parallel:
		speedup = len(method)/float(method.num_seq_dep_stages())
		work = work/speedup
	pl.loglog(work,err0,label=method.name,linewidth=3)
    pl.xlabel('Function evaluations')
    pl.ylabel('Error at $t_{final}$')
    pl.legend(loc='best')
    pl.hold(False)
    pl.draw()
    return work, err


def ptest(methods,ivps,tols=[1.e-1,1.e-2,1.e-4,1.e-6],verbosity=0,parallel=False):
    """
        Runs a performance test, integrating a set of problems with a set
        of methods using a sequence of error tolerances.  Creates a plot 
        of the error achieved versus the amount of work done (number of
        function evaluations) for each method.

        **Input**:
            * methods -- a list of ODEsolver instances
                      Note that all methods must have error estimators.
            * ivps    -- a list of IVP instances
            * tols    -- a specified list of error tolerances (optional)
            * parallel -- to exploit possible parallelization (optional)

        **Example**::

            >>> import runge_kutta_method as rk
            >>> from ivp import load_ivp
            >>> bs5=rk.loadRKM('BS5')
            >>> myivp=load_ivp('nlsin')
            >>> work,err=ptest(bs5,myivp)

    """
    import matplotlib.pyplot as pl
    pl.clf(); pl.draw(); pl.hold(True)
    # In case just one method is passed in (and not as a list):
    if not isinstance(methods,list): methods=[methods]
    if not isinstance(ivps,list): ivps=[ivps]
    err=np.ones([len(methods),len(tols)])
    work=np.zeros([len(methods),len(tols)])
    for ivp in ivps:
        if verbosity>0: print "solving problem %s" % ivp
        try:
            exsol = ivp.exact(ivp.T)
        except:
            bs5=rk.loadRKM('BS5')   #Use Bogacki-Shampine RK for fine solution
            lowtol=min(tols)/100.
            if verbosity>0: print 'solving for "exact" solution with tol= '+str(lowtol)
            t,u=bs5(ivp,errtol=lowtol,dt=ivp.dt0)
            if verbosity>0: print 'done'
            exsol = u[-1] + 0.
        for imeth,method in enumerate(methods):
            if verbosity>0: print 'Solving with method '+method.name
            if parallel:
                speedup = len(method)/float(method.num_seq_dep_stages())
            else:
                speedup = 1.
            workperstep = len(method)-method.is_FSAL()
            for jtol,tol in enumerate(tols):
                t,u,rej,dt,errhist=method(ivp,errtol=tol,dt=ivp.dt0,diagnostics=True,controllertype='P')
                if verbosity>1: print str(rej)+' rejected steps'
                err[imeth,jtol]*= np.max(np.abs(u[-1]-exsol))
                #FSAL methods save on accepted steps, but not on rejected:
                work[imeth,jtol]+= (len(t)*workperstep+rej*len(method))/speedup
    for imeth,method in enumerate(methods):
        for jtol,tol in enumerate(tols):
            err[imeth,jtol]=err[imeth,jtol]**(1./len(ivps))
    for imeth,method in enumerate(methods):
        pl.semilogy(work[imeth,:],err[imeth,:],label=method.name,linewidth=3)
    pl.xlabel('Function evaluations')
    pl.ylabel('Error at $t_{final}$')
    pl.legend(loc='best')
    pl.hold(False)
    pl.draw()
    return work,err



if __name__ == "__main__":
    import doctest
    doctest.testmod()
