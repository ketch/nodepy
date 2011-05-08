"""
Example showing how to set up a semi-discretization with the spectral difference method and advect it.
"""

# Import libraries
from nodepy import semidisc
from nodepy import *

import numpy as np
import matplotlib.pyplot as pl


# Create spatial operator L (i.e. u' = L u)
spectralDifference = semidisc.load_semidisc('spectral difference advection',order=2)

# Create time marching
rk4=rk.loadRKM('RK44')

# Solve the problem
t,y=rk4(spectralDifference)

# Plot the soution
pl.plot(spectralDifference.xExact,spectralDifference.uExact,label = 'Exact solution')
pl.plot(spectralDifference.xSol,y[-1],label = 'Spectral difference solution')
pl.title('1D linear advection equation')
pl.xlabel('x')
pl.ylabel('u')
pl.legend()
pl.show()

