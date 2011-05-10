"""
Example showing how to set up a semi-discretization and advect it.
"""

# Import libraries
##################
from nodepy import semidisc
from nodepy import *
import matplotlib.pyplot as pl


# Create spatial operator L (i.e. u' = L u)
###########################################
upwind = semidisc.load_semidisc('upwind advection')


# Create time marching
######################
rk4=rk.loadRKM('RK44')


# Solve the problem
###################
t,y=rk4(upwind)


# Plot the soution
##################
pl.plot(upwind.xExact,upwind.uExactInit,label = 'Initial condition')
pl.plot(upwind.xExact,upwind.uExact,label = 'Exact solution')

pl.plot(upwind.xCenter,y[-1],label = 'Upwind solution')
pl.title('1D linear advection equation')
pl.xlabel('x')
pl.ylabel('u')
pl.legend()
pl.show()

