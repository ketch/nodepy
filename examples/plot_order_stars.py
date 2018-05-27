from nodepy.stability_function import pade_exp
from nodepy.stability_function import plot_order_star
import matplotlib.pyplot as pl
import numpy as np

P=list(range(5))
Q=list(range(5))
k=(5,4,4,0)
j=(5,6,7,10)
for ip in range(4):
    P[ip],Q[ip]=pade_exp(k[ip],j[ip])
    plot_order_star(P[ip],Q[ip],bounds=[-10,10,-10,10])
    pl.plot(P[ip].r.real,P[ip].r.imag,'ok')
    pl.plot(Q[ip].r.real,Q[ip].r.imag,'ok')
    pl.title('k='+str(k[ip])+',j='+str(j[ip]))
    pl.show()
