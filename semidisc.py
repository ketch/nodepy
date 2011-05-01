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

def load_semidisc(sdname,order=1,N=50,xmin=0.,xmax=1.):
    sd=LinearSemiDiscretization()
    #Set up grid
    dx=(xmax-xmin)/N;               # Grid spacing
    sd.x=np.linspace(xmin,xmax,N)   # Grid
    sd.N=N
    if sdname=='upwind advection':
        sd.L = upwind_advection_matrix(N,dx)
        print sd.L
    elif sdname == 'spectral difference advection':
        sd.flxPnts = spectral_difference_matrix(N,dx,order)
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

def spectral_difference_matrix(N,dx,order):
    import sys

    # Set coordinates of flux and solutions points according to the order of the scheme
    if order == 1:
        # 1st order
        fluxPnts = np.array([ -1.0 , 1.0 ])
        solPnts = np.array([ 0 ])
    elif order == 2:
        # 2nd order
        fluxPnts = np.array([ -1.0 , 0.0 , 1.0 ])
        solPnts = np.array([ -1.0 ,       1.0 ])
            
    elif order == 3:
        # 3rd order
        fluxPnts = np.array([ -1.0 , -0.58 , 0.58 , 1.0 ])
        solPnts = np.array([ -1.0 ,         0.58 , 1.0 ])
    
    elif order == 4:
        # 4th order
        fluxPnts = np.array([ -1.0 , -0.78 , 0.0 , 0.78 , 1.0 ])
        solPnts = np.array([ -1.0 , -0.78 ,       0.78 , 1.0 ])

    elif order == 5:
        # 5th order
        fluxPnts = np.array([ -1.0 , -0.83 , -0.36 , 0.36 , 0.83 , 1.0 ])
        solPnts = np.array([ -1.0 , -0.83 ,         0.36 , 0.83 , 1.0 ])
    
    elif order == 6:
        # 6th order
        fluxPnts = np.array([ -1.0 , -0.88 , -0.53 , 0.0 , 0.53 , 0.88 , 1.0 ])
        solPnts = np.array([ -1.0 , -0.88 , -0.53 ,       0.53 , 0.88 , 1.0 ])
    else:
       print "Error: min order 1, max order 6"
       sys.exit()


    #print fluxPnts
    #print solPnts

    # Number of solution and flux points
    ####################################
    nbrFluxPnts = fluxPnts.size
    nbrSolPnts = solPnts.size

    # Matrices for the calculation of the solution at the flux points 
    #################################################################
    extrSolToFlux = np.ones((nbrFluxPnts,nbrSolPnts))
    #print extrSolToFlux


    for iFlux in range(0,nbrFluxPnts):
        for iSol in range(0,nbrSolPnts):
            for iCon in range(0,nbrSolPnts):
                if iCon != iSol:
                    extrSolToFlux[iFlux,iSol] = extrSolToFlux[iFlux,iSol]*(fluxPnts[iFlux]-solPnts[iCon])/(solPnts[iSol]-solPnts[iCon])

    #print extrSolToFlux
            
           
  
    # Riemann flux settings
    #######################
    upwindPar = 1.0 # If fully upwind (upwindPar =1). For advection equation upwindPar = 1.0

    faceLeft  = 0.5*(1.0 + upwindPar); # Contribution of the left solution 
    faceRight = 0.5*(1.0 - upwindPar); # Contribution of the right solution


    # Construction of the flux polynomial
    #####################################
    fluxCurrCell = np.zeros((nbrFluxPnts,nbrSolPnts));
    fluxLeftCell = np.zeros((nbrFluxPnts,nbrSolPnts));
    fluxRightCell = np.zeros((nbrFluxPnts,nbrSolPnts));
    
    # Flux at the internal flux points
    for iFlux in range(1,nbrFluxPnts-1):
        for iSol in range(0,nbrSolPnts):
            fluxCurrCell[iFlux,iSol] = extrSolToFlux[iFlux,iSol]  # We assume convective velocity = 1, i.e. the flux function is 
                                                                  # f= a*u

    #print fluxCurrCell


    # Flux at the boundaries flux points
    for iSol in range(0,nbrSolPnts):
        # Left face
        fluxLeftCell[0,iSol] = faceLeft*extrSolToFlux[nbrFluxPnts-1,iSol] 
        fluxCurrCell[0,iSol] = faceRight*extrSolToFlux[0,iSol]
    
        # Right face
        fluxCurrCell[nbrFluxPnts-1,iSol] = faceLeft*extrSolToFlux[nbrFluxPnts-1,iSol]
        fluxRightCell[nbrFluxPnts-1,iSol] = faceRight*extrSolToFlux[0,iSol]

    #print fluxCurrCell
    #print fluxRightCell
    #print fluxLeftCell
    

    # Matrix for the calculation of the flux derivative at the solution points
    ##########################################################################
    derivFluxInSolPnts = np.zeros((nbrSolPnts,nbrFluxPnts))

    for iSol in range(0,nbrSolPnts):
        for iFlux in range(0,nbrFluxPnts):
            for iCon in range(0,nbrFluxPnts):
                if iCon != iFlux:
                    contribution = 1.0/(fluxPnts[iFlux]-fluxPnts[iCon])
                    for iFac in range(0,nbrFluxPnts):
                        if (iFac != iCon) & (iFac != iFlux):
                            contribution = contribution*(solPnts[iSol]-fluxPnts[iFac])/(fluxPnts[iFlux]-fluxPnts[iFac])
                    derivFluxInSolPnts[iSol,iFlux] = derivFluxInSolPnts[iSol,iFlux] + contribution
            # Factor 2 comes from Jacobian determinant
            derivFluxInSolPnts[iSol,iFlux] = derivFluxInSolPnts[iSol,iFlux]*2.0

    #print derivFluxInSolPnts


    # Compute the products between the matrix of the derivatives and the solution contributions
    ###########################################################################################
    DMm1 = np.dot(derivFluxInSolPnts,fluxLeftCell)
    #print DMm1
    DM0  = np.dot(derivFluxInSolPnts,fluxCurrCell)
    #print DM0
    DMp1 = np.dot(derivFluxInSolPnts,fluxRightCell)
    #print DMp1

    #print type(DMm1)
    #print type(fluxLeftCell)
    #print type(fluxLeftCell)
    #print type(fluxCurrCell)
    #print type(fluxRightCell)

    # Create block tridiagonal matrix
    #################################
    dimL = nbrSolPnts*3
    L = np.zeros((dimL,dimL))

    # Main block diagonal
    for iBlock in range(0,3):
        L[nbrSolPnts*iBlock:nbrSolPnts*(iBlock+1),nbrSolPnts*iBlock:nbrSolPnts*(iBlock+1)] = DM0[:,:] 
    
    # Lower and upper blocks diagonal
    for iBlock in range(0,2):
        L[nbrSolPnts*(iBlock+1):nbrSolPnts*(iBlock+2),nbrSolPnts*iBlock:nbrSolPnts*(iBlock+1)] = DMm1[:,:]
        L[nbrSolPnts*iBlock:nbrSolPnts*(iBlock+1),nbrSolPnts*(iBlock+1):nbrSolPnts*(iBlock+2)] = DMp1[:,:]


    

    return




