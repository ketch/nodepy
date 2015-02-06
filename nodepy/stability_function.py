import numpy as np

def imaginary_stability_interval(p,q=None,eps=1.e-14):
    r"""
        Length of imaginary axis half-interval contained in the
        method's region of absolute stability.

        **Examples**::

            >>> from nodepy import rk
            >>> rk4 = rk.loadRKM('RK44')
            >>> rk4.imaginary_stability_interval() #doctest: +ELLIPSIS
            2.8284271247...
    """
    if q is None: 
        q = np.poly1d([1.])

    c = p.c[::-1].copy()
    c[1::2] = 0      # Zero the odd coefficients to get real part
    c[::2][1::2] = -1*c[::2][1::2]  # Negate coefficients with even powers of i
    p1 = np.poly1d(c[::-1])

    c = p.c[::-1].copy()
    c[::2] = 0      # Zero the even coefficients to get imaginary part
    c[1::2][1::2] = -1*c[1::2][1::2]  # Negate coefficients with even powers of i
    p2 = np.poly1d(c[::-1])

    c = q.c[::-1].copy()
    c[1::2] = 0      # Zero the odd coefficients to get real part
    c[::2][1::2] = -1*c[::2][1::2]  # Negate coefficients with even powers of i
    q1 = np.poly1d(c[::-1])

    c = q.c[::-1].copy()
    c[::2] = 0      # Zero the even coefficients to get imaginary part
    c[1::2][1::2] = -1*c[1::2][1::2]  # Negate coefficients with even powers of i
    q2 = np.poly1d(c[::-1])

    ppq = p1**2 + p2**2 + q1**2 + q2**2
    pmq = p1**2 + p2**2 - q1**2 - q2**2

    ppq_roots = np.array([x.real for x in ppq.r if abs(x.imag)<eps and x.real>0])
    pmq_roots = np.array([x.real for x in pmq.r if abs(x.imag)<eps and x.real>0])

    if len(pmq_roots)>0: pmqr = np.min(pmq_roots)
    else: pmqr = np.inf
    if len(ppq_roots)>0: ppqr = np.min(ppq_roots)
    else: ppqr = np.inf

    mr = min(pmqr,ppqr)

    if mr == np.inf:
        # Stability region boundary does not touch/cross the imaginary axis
        # If it's explicit, it must be unstable for all imaginary values
        if len(q.coeffs==1): return 0
        z = 1j/2.
    else:
        z = mr*1j/2.

    val = 1
    while val==1:
        # Check whether it is stable between 0 and mr
        # This could be wrong if the stability boundary is tangent to the imaginary axis at mr
        val = np.abs(p(z)/q(z))
        if val<1:
            return mr
        elif val==1:
            z = z/2.
        else:
            return 0
        if np.abs(z)<1.e-10:
            print "Warning: unable to determine exact imaginary stability interval"
            return mr


def real_stability_interval(p,q=None,eps=1.e-12):
    r"""
        Length of negative real axis interval contained in the
        method's region of absolute stability.

        **Examples**::

            >>> from nodepy import rk
            >>> rk4 = rk.loadRKM('RK44')
            >>> I = rk4.real_stability_interval()
            >>> print "%.10f" % I
            2.7852935634
            >>> rkc = rk.RKC1(2)
            >>> rkc.real_stability_interval()
            8.0
    """
    if q is None: q = np.poly1d([1.])

    # Find points where p = +/- q
    pmq = p-q
    ppq = p+q
    pmq_roots = np.array([-x.real for x in pmq.r if abs(x.imag)<eps and x.real<0])
    ppq_roots = np.array([-x.real for x in ppq.r if abs(x.imag)<eps and x.real<0])
    roots = np.hstack((pmq_roots,ppq_roots))
    roots = np.unique(roots)

    z = -roots[0]/2.
    if np.abs(p(z)/q(z))>1.+eps:
        return 0.

    for i in range(len(roots)-1):
        z = -(roots[i]+roots[i+1])/2
        if np.abs(p(z)/q(z))>1.+eps:
            return roots[i]
    z = -roots[-1]*2 
    if np.abs(p(z)/q(z))>1.+eps:
        return roots[-1]
    return np.inf


def plot_stability_region(p,q,N=200,color='r',filled=True,bounds=None,
                          plotroots=False,alpha=1.,scalefac=1.,fignum=None):
    r""" 
        Plot the region of absolute stability of a rational function; i.e. the set

        `\{ z \in C : |\phi (z)|\le 1 \}`

        Unless specified explicitly, the plot bounds are determined automatically, attempting to
        include the entire region.  A check is performed beforehand for
        methods with unbounded stability regions.
        Note that this function is not responsible for actually drawing the 
        figure to the screen (or saving it to file).

        **Inputs**: 
            - p, q  -- Numerator and denominator of the stability function 
            - N       -- Number of gridpoints to use in each direction
            - color   -- color to use for this plot
            - filled  -- if true, stability region is filled in (solid); otherwise it is outlined
            - plotroots -- if True, plot the roots and poles of the function
            - alpha -- transparency of contour plot
            - scalefac -- factor by which to scale region (often used to normalize for stage number)
            - fignum -- number of existing figure to use for plot

    """
    import matplotlib.pyplot as plt

    # Convert coefficients to floats for speed
    if p.coeffs.dtype=='object':
        p = np.poly1d([float(c) for c in p.coeffs])
    if q.coeffs.dtype=='object':
        q = np.poly1d([float(c) for c in q.coeffs])

    if bounds is None:
        from utils import find_plot_bounds
        # Check if the stability region is bounded or not
        m,n = p.order,q.order
        if (m < n) or ((m == n) and (abs(p[m])<abs(q[n]))):
            print 'The stability region is unbounded'
            bounds = (-10*m,m,-5*m,5*m)
        else:
            stable = lambda z : np.abs(p(z)/q(z))<=1.0
            bounds = find_plot_bounds(stable,guess=(-10,1,-5,5))
            if np.min(np.abs(np.array(bounds)))<1.e-14:
                print 'No stable region found; is this method zero-stable?'

        if (m == n) and (abs(p[m])==abs(q[n])):
            print 'The stability region may be unbounded'

    # Evaluate the stability function over a grid
    x=np.linspace(bounds[0],bounds[1],N)
    y=np.linspace(bounds[2],bounds[3],N)
    X=np.tile(x,(N,1))
    Y=np.tile(y[:,np.newaxis],(1,N))
    Z=X+Y*1j
    R=np.abs(p(Z*scalefac)/q(Z*scalefac))

    # Plot
    h = plt.figure(fignum)
    plt.hold(True)
    if filled:
        plt.contourf(X,Y,R,[0,1],colors=color,alpha=alpha)
    else:
        plt.contour(X,Y,R,[0,1],colors=color,alpha=alpha,linewidths=3)
    if plotroots: plt.plot(np.real(p.r),np.imag(p.r),'ok')
    if len(q)>1: plt.plot(np.real(q.r),np.imag(q.r),'xk')
    plt.plot([0,0],[bounds[2],bounds[3]],'--k',linewidth=2)
    plt.plot([bounds[0],bounds[1]],[0,0],'--k',linewidth=2)
    plt.axis('Image')
    plt.hold(False)
    return h


def plot_order_star(p,q,N=200,bounds=[-5,5,-5,5], plotroots=False,
                color=('w','b'),filled=True,subplot=None):
    r""" Plot the order star of a rational function
        i.e. the set
        
        $$ \{ z \in C : |R(z)/exp(z)|\le 1 \} $$

        where $R(z)=p(z)/q(z)$ is the stability function of the method.

        **Input**: (all optional)
            - N       -- Number of gridpoints to use in each direction
            - bounds  -- limits of plotting region
            - color   -- color to use for this plot
            - filled  -- if true, order star is filled in (solid); otherwise it is outlined
    """
    # Convert coefficients to floats for speed
    if p.coeffs.dtype=='object':
        p = np.poly1d([float(c) for c in p.coeffs])
    if q.coeffs.dtype=='object':
        q = np.poly1d([float(c) for c in q.coeffs])


    import matplotlib.pyplot as plt
    x=np.linspace(bounds[0],bounds[1],N)
    y=np.linspace(bounds[2],bounds[3],N)
    X=np.tile(x,(N,1))
    Y=np.tile(y[:,np.newaxis],(1,N))
    Z=X+Y*1j
    R=np.abs(p(Z)/q(Z)/np.exp(Z))
    if subplot is not None:
        plt.subplot(subplot[0],subplot[1],subplot[2])
    else:
        plt.clf()
    plt.contourf(X,Y,R,[0,1,1.e299],colors=color)
    plt.hold(True)
    if plotroots: plt.plot(np.real(p.r),np.imag(p.r),'ok')
    plt.plot([0,0],[bounds[2],bounds[3]],'--k')
    plt.plot([bounds[0],bounds[1]],[0,0],'--k')
    plt.axis('Image')
    plt.hold(False)
    plt.draw()


def pade_exp(k,j):
    """
    Return the Pade approximation to the exponential function
    with numerator of degree k and denominator of degree j.
    """
    Pcoeffs=[1]
    Qcoeffs=[1]
    for n in range(1,k+1):
        newcoeff=Pcoeffs[0]*(k-n+1.)/(j+k-n+1.)/n
        Pcoeffs=[newcoeff]+Pcoeffs
    P=np.poly1d(Pcoeffs)
    for n in range(1,j+1):
        newcoeff=-1.*Qcoeffs[0]*(j-n+1.)/(j+k-n+1.)/n
        Qcoeffs=[newcoeff]+Qcoeffs
    Q=np.poly1d(Qcoeffs)
    return P,Q

def taylor(p):
    r"""
        Return the Taylor polynomial of the exponential, up to order p.
    """
    from sympy import factorial

    coeffs = np.array( [1./factorial(k) for k in range(p+1) ] )

    return np.poly1d(coeffs[::-1])

if __name__ == "__main__":
    import doctest
    doctest.testmod()

