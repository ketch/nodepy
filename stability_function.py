import numpy as np
import matplotlib.pyplot as pl

def plot_stability_region(p,q,N=200,bounds=[-10,1,-5,5],
                color='r',filled=True,scaled=False,plotroots=True,
                alpha=1.,scalefac=None):
    r""" 
        The region of absolute stability
        of a rational function; i.e. the set

        `\{ z \in C : |\phi (z)|\le 1 \}`

        **Input**: (all optional)
            - N       -- Number of gridpoints to use in each direction
            - bounds  -- limits of plotting region
            - color   -- color to use for this plot
            - filled  -- if true, stability region is filled in (solid); otherwise it is outlined
    """
    m=len(p)
    x=np.linspace(bounds[0],bounds[1],N)
    y=np.linspace(bounds[2],bounds[3],N)
    X=np.tile(x,(N,1))
    Y=np.tile(y[:,np.newaxis],(1,N))
    Z=X+Y*1j
    if scaled: 
        if scalefac==None: scalefac=m
    else: scalefac=1.
    R=np.abs(p(Z*scalefac)/q(Z*scalefac))
    #pl.clf()
    if filled:
        pl.contourf(X,Y,R,[0,1],colors=color,alpha=alpha)
    else:
        pl.contour(X,Y,R,[0,1],colors=color,alpha=alpha)
    pl.title('Absolute Stability Region')
    pl.hold(True)
    if plotroots: pl.plot(np.real(p.r),np.imag(p.r),'ok')
    if len(q)>1: pl.plot(np.real(q.r),np.imag(q.r),'xk')
    pl.plot([0,0],[bounds[2],bounds[3]],'--k',linewidth=2)
    pl.plot([bounds[0],bounds[1]],[0,0],'--k',linewidth=2)
    pl.axis('Image')
    pl.hold(False)
    pl.draw()

def plot_order_star(p,q,N=200,bounds=[-5,5,-5,5],
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
    x=np.linspace(bounds[0],bounds[1],N)
    y=np.linspace(bounds[2],bounds[3],N)
    X=np.tile(x,(N,1))
    Y=np.tile(y[:,np.newaxis],(1,N))
    Z=X+Y*1j
    R=np.abs(p(Z)/q(Z)/np.exp(Z))
    if subplot is not None:
        pl.subplot(subplot[0],subplot[1],subplot[2])
    else:
        pl.clf()
    pl.contourf(X,Y,R,[0,1,1.e299],colors=color)
    pl.hold(True)
    pl.plot([0,0],[bounds[2],bounds[3]],'--k')
    pl.plot([bounds[0],bounds[1]],[0,0],'--k')
    pl.axis('Image')
    pl.hold(False)
    pl.draw()


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
    P=pl.poly1d(Pcoeffs)
    for n in range(1,j+1):
        newcoeff=-1.*Qcoeffs[0]*(j-n+1.)/(j+k-n+1.)/n
        Qcoeffs=[newcoeff]+Qcoeffs
    Q=pl.poly1d(Qcoeffs)
    return P,Q

def taylor(p):
    r"""
        Return the Taylor polynomial of the exponential, up to order p.
    """
    from scipy import factorial

    coeffs=1./factorial(np.arange(p+1))

    return np.poly1d(coeffs[::-1])

def plot_stability_region(p,q,N=200,bounds=[-10,1,-5,5],
                color='r',filled=True,scaled=False,plotroots=True,
                alpha=1.,scalefac=None):
    r""" 
        The region of absolute stability
        of a Runge-Kutta method, is the set

        `\{ z \in C : |\phi (z)|\le 1 \}`

        where $\phi(z)$ is the stability function of the method.

        **Input**: (all optional)
            - N       -- Number of gridpoints to use in each direction
            - bounds  -- limits of plotting region
            - color   -- color to use for this plot
            - filled  -- if true, stability region is filled in (solid); otherwise it is outlined
    """
    m=len(p)
    x=np.linspace(bounds[0],bounds[1],N)
    y=np.linspace(bounds[2],bounds[3],N)
    X=np.tile(x,(N,1))
    Y=np.tile(y[:,np.newaxis],(1,N))
    Z=X+Y*1j
    if scaled: 
        if scalefac==None: scalefac=m
    else: scalefac=1.
    R=np.abs(p(Z*scalefac)/q(Z*scalefac))
    pl.clf()
    if filled:
        pl.contourf(X,Y,R,[0,1],colors=color,alpha=alpha)
    else:
        pl.contour(X,Y,R,[0,1],colors=color,alpha=alpha)
    pl.title('Absolute Stability Region')
    pl.hold(True)
    if plotroots: pl.plot(np.real(p.r),np.imag(p.r),'ok')
    if len(q)>1: pl.plot(np.real(q.r),np.imag(q.r),'xk')
    pl.plot([0,0],[bounds[2],bounds[3]],'--k',linewidth=2)
    pl.plot([bounds[0],bounds[1]],[0,0],'--k',linewidth=2)
    pl.axis('Image')
    pl.hold(False)
    pl.draw()


