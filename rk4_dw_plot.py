import runge_kutta_method as rk
import pylab as pl

wvals=pl.linspace(-1,1,1078)
vals=pl.zeros(pl.size(wvals))
for i in range(len(wvals)):
    rk4=rk.RK44_family(wvals[i])
    try:
        r,d,a,at=rk4.optimal_perturbed_splitting()
        vals[i]=r
    except:
        print 'oops', wvals[i]
        vals[i]=0

pl.plot(wvals,vals)
pl.draw()
pl.show()

