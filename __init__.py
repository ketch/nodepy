"""
NodePy (Numerical ODE solvers in Python) is...
"""

__version__="0.3"

import sys
mypath= '/Users/ketch/Research/Projects/nodepy'
if mypath not in sys.path: sys.path.append(mypath)

import runge_kutta_method as rkm
import linear_multistep_method as lmm
import rooted_trees as rt
