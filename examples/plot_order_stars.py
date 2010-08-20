from NodePy.stability_function import pade_exp
from NodePy.stability_function import plot_order_star
import pylab as pl

P=range(5)
Q=range(5)
k=(5,4,4,0)
j=(5,6,7,10)
for ip in range(4):
    P[ip],Q[ip]=pade_exp(k[ip],j[ip])
    plot_order_star(P[ip],Q[ip],subplot=(2,2,ip+1),bounds=[-10,10,-10,10])
    pl.hold(True)
    pl.plot(P[ip].r.real,P[ip].r.imag,'ok')
    pl.plot(Q[ip].r.real,Q[ip].r.imag,'ok')
    pl.hold(False)
    pl.title('k='+str(k[ip])+',j='+str(j[ip]))
