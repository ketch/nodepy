#Python script to solve Nick Cain's system

import pylab as pl
from runge_kutta_method import *
import rhs

dx=0.05
x=pl.arange(0,1.,dx)
u0=x*0.
tend=30.
dt=1.95
rk=loadRKM('Mid22')

t,u=rk(rhs.ncfun,u0,[0,tend],1,dt=dt,x=x)

X,T=pl.meshgrid(x,t)
#pl.pcolor(X,T,u)
pl.plot(t,u)
