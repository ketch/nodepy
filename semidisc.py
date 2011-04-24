#ISSUES:
#    - Need a way to pass things like dx
#    - Need to call boundary conditions separately since they aren't always
#        written as ODEs

#Should have a special class for linear semidiscretizations?
"""
In NodePy a semi-discretization is a family of IVPs parameterized by grid size.
For now, only semi-discretizations of one-dimensional PDEs are supported.
"""

import numpy as np
from ivp import IVP

class LinearSemiDiscretization(IVP):
    """
        Class for linear semi-discretizations of PDEs.
        Inherits from IVP, but possesses a grid and
        is parameterized by grid size.

        Any instance should provide:

        - L: the matrix representation of the right-hand-side
        - N, xmin, xmax (describing the grid)
    """

def load_semidisc(sdname,N=50,xmin=0.,xmax=1.):
    sd=LinearSemiDiscretization()
    #Set up grid
    dx=(xmax-xmin)/N;         #Grid spacing
    sd.x=np.linspace(xmin,xmax,N)
    sd.N=N
    if sdname=='upwind advection':
        sd.L = upwind_advection_matrix(N,dx)
    else: print 'unrecognized sdname'
    sd.rhs=lambda t,u : np.dot(sd.L,u)
    sd.u0 = np.sin(2*np.pi*sd.x)
    sd.T = 1.
    return sd

def upwind_advection_matrix(N,dx):
    from scipy.sparse import spdiags
    e=np.ones(N)
    #L=spdiags([-e,e,e],[0,-1,N-1],N,N)/dx
    L=(np.diag(-e)+np.diag(e[1:],-1))/dx
    L[0,-1]=1./dx
    return L
