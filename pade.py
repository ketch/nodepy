import pylab as pl
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
