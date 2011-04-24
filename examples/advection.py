"""
Example showing how to set up a semi-discretization and advect it.
"""
from nodepy import semidisc
from nodepy import *
import matplotlib.pyplot as pl

upwind=semidisc.load_semidisc('upwind advection')
rk4=rk.loadRKM('RK44')

t,y=rk4(upwind)

pl.plot(upwind.x,y[0])
pl.plot(upwind.x,y[-1])
pl.show()
