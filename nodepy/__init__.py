"""
NodePy (Numerical ODE solvers in Python) is...
"""

__version__="0.6"

import sys
mypath= '/Users/ketch/Research/Projects/nodepy'
if mypath not in sys.path: sys.path.append(mypath)

import runge_kutta_method as rk
import linear_multistep_method as lm
import rooted_trees as rt
import ivp
import convergence as conv
import low_storage_rk as lsrk
