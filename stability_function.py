import numpy as np
import pylab as pl

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
