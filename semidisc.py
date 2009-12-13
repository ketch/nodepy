"""
Functions related to semi-discretizations of PDEs
"""

import numpy as np

class SemiDiscretization(ODE):
    """
        Class for semi-discretizations of PDEs.
        Inherits from ODE, but also has boundary conditions.
    """

class Advection(SemiDiscretization):
    def __init__(self,x,u0):
        self.x=x
        self.u0=u0
    def exact(self,t):
        return self.u0(self.x-t)

class UpwindAdvection(Advection):
    """
        First order upwind semi-discretization of the advection equation:

        u_t + u_x = 0
    """
    def rhs(self,t,u):
        return np.array([u[i]-u[i-1] for i in range(len(u))])
        
    def rhs_matrix(self):
        """
            Returns a matrix L such that the ODE is given by

            u' = Lu.
        """
        N=len(self.x)
        dx=self.x[1]=self.x[0]
        return 1./dx*(np.diag(np.ones(N-1),-1)-np.diag(np.ones(N)))

def hatfunction(x,x1,x2):
    return 1.*(x>x1)*(x<x2)
