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

class SemiDiscretization(IVP):
    """
        Class for semi-discretizations of PDEs.
        Inherits from IVP, but possesses a grid and
        is parameterized by grid size.
    """

def load_semidisc(sdname,N=50,xmin=0.,xmax=1.,nghost=2,bctype=periodic):
    sd=SemiDiscretization()
    #Set up grid
    sd.dx=1./N;         #Grid spacing
    N2=N+2*nghost;      #Total number of points, including ghosts
    sd.x=np.linspace(-(nghost-0.5)*dx,1.+(nghost-0.5)*dx,N2)
    if sdname=='upwind advection':
        sd.rhs = upwind_advection_rhs
    else: print 'unrecognized sdname'
    sd.bc=bc
    sd.bctype=bctype
    return sd

def upwind_advection_rhs(t,u,sd):
    N=len(u)
    du = zeros(N)
    du[1:] = - sd.dx * (u[1:]-u[:-1])

def bc(t,u,bctype):
    if bctype=='periodic':
        u[0:nghost]  = u[-2*nghost:-nghost] # Periodic boundary
        u[-nghost:]  = u[nghost:2*nghost]   # Periodic boundary
    else: print 'Unrecognized bctype'
    return u
