def bisect(rlo, rhi, acc, tol, fun, **kwargs):
    """ 
        Performs a bisection search.

        **Input**:
            - fun -- a function such that fun(r)==True iff x_0>r, where x_0 is the value to be found.
    """
    while rhi-rlo>acc:
        r=0.5*(rhi+rlo)
        if kwargs: isvalid=fun(r,tol,**kwargs)
        else: isvalid=fun(r,tol)
        if isvalid:
            rlo=r
        else:
            rhi=r
    return rlo

def permutations(str):
    if len(str) <=1:
        yield str
    else:
        for perm in permutations(str[1:]):
            for i in range(len(perm)+1):
                yield perm[:i] + str[0:1] + perm[i:]

def shortstring(x,printzeros=False):
    import numpy as np
    import sympy.core.numbers
    if x==0 and printzeros==False:
        return ''
    elif x.__class__ is np.float64 or x.__class__ is sympy.Float:
        return '%6.3f' % x
    else: 
        return ' '+str(x)

def array2strings(x,printzeros=False):
    import numpy as np
    return np.reshape([shortstring(xi,printzeros) for xi in x.reshape(-1)],x.shape)

def find_plot_bounds(f,guess,N=101,zmax=1000):
    r"""Find reasonable area to plot for stability regions.
    
    Tries to find an area that contains the entire stability region
    but isn't too big.  Makes lots of assumptions.  Obviously can't
    work for unbounded stability regions.

    f should return True if f(z) is in the stability region.

    N should be odd in order to catch very small stability regions.
    """
    import numpy as np

    bounds = guess
    old_bounds = []

    while bounds != old_bounds:
        old_bounds = bounds
        y=np.linspace(bounds[2],bounds[3],N)

        #Check boundaries
        bounds = list(bounds)
        close = False
        while abs(bounds[0])<zmax:
            x=np.linspace(bounds[0],bounds[1],N)
            Z=x[0]+y*1j

            left = f(Z)
            if np.any(left):
                bounds[0] = 1.5*bounds[0]
                close = True
            else:
                if close == True:
                    break
                bounds[0] = bounds[0]/1.5
                if bounds[0] > -1.e-15:
                    return bounds

        bounds[1] = -0.1*bounds[0]
        x=np.linspace(bounds[0],bounds[1],N) + 0.*1j

        close = False
        while abs(bounds[2])<zmax:
            y=np.linspace(bounds[2],bounds[3],N)
            Z=x + y[0]*1j

            bottom = f(Z)
            if np.any(bottom):
                bounds[2] = 1.5*bounds[2]
                close = True
            else:
                if close == True:
                    break
                bounds[2] = bounds[2]/1.5
                if bounds[2] > -1.e-15:
                    return bounds

        bounds[3] = -bounds[2]

    return bounds

 
