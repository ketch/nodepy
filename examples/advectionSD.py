"""
Example showing how to set up a semi-discretization with the spectral difference method and advect it.
"""

# Import libraries
##################
from nodepy import semidisc
from nodepy import *

import numpy as np
import matplotlib.pyplot as pl


# Create spatial operator L (i.e. u' = L u)
###########################################
orderAcc = 1
spectralDifference = semidisc.load_semidisc('spectral difference advection',order=orderAcc)


# Create time marching
######################
rk4=rk.loadRKM('RK44')


# Solve the problem
###################
t,y=rk4(spectralDifference)


# Plot the soution
##################
pl.plot(spectralDifference.xExact,spectralDifference.uExact,label = 'Exact solution')

# Check if we want a 1st-order spectral difference solution. If we want that, prepare some arrays
# for pretty plots
if orderAcc == 1:
    # Copy the last element of the list y in temporary array. 
    # The element is a numpy array.
    tmp = y[-1]
    
    # Solution is constant in a cell. Thus two points are enough for plotting a pice-wise constant
    # function
    nbrPlotPnts = 2*spectralDifference.xCenter.size
    x1stSD=np.zeros(nbrPlotPnts)
    u1stSD=np.zeros(nbrPlotPnts)
    dx = spectralDifference.xCenter[1] - spectralDifference.xCenter[0] # Assume uniform grid spacing

    for i in range(0,spectralDifference.xCenter.size):
        for j in range(0,2):
            # Compute x coordinate
            x1stSD[i*2]   = spectralDifference.xCenter[i] - 1./2.*dx
            x1stSD[i*2+1] = spectralDifference.xCenter[i] + 1./2.*dx
            
            # Set solution
            u1stSD[i*2]   = tmp[i]
            u1stSD[i*2+1] = tmp[i]
    
    # Plot 1st-order numerical solution
    pl.plot(x1stSD,u1stSD,label = 'Spectral difference solution')

else:
    # Plot orderAcc-order numerical solution
    pl.plot(spectralDifference.xSol,y[-1],label = 'Spectral difference solution')


pl.title('1D linear advection equation')
pl.xlabel('x')
pl.ylabel('u')
pl.legend()
pl.show()

