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

def load_semidisc(sdName,N=100,xMin=0.,xMax=1.,order=1):
    sd=LinearSemiDiscretization()

    #Set up grid
    ############
    # Number of cells
    sd.N=N

    # Grid spacing
    dx=(xMax-xMin)/N;                               

    # Cell centers position
    sd.xCenter = np.zeros(N)
    sd.xCenter[0] = xMin + dx/2.0
    for i in range(1,N):
        sd.xCenter[i] = sd.xCenter[0]+dx*i

    # The spatial discretization sets also the initial condition
    # because the number of DOF depends on the scheme
    if sdName=='upwind advection':
        sd.L  = upwind_advection_matrix(N,dx)
        sd.u0 = np.sin(2*np.pi*sd.xCenter)
    if sdName=='weno5':
        sd.L  = weno5_linearized_matrix(N,dx)
        sd.u0 = np.sin(2*np.pi*sd.xCenter)
    elif sdName == 'spectral difference advection':
        sd.L,sd.xSol,sd.u0 = spectral_difference_matrix(N,sd.xCenter,dx,order)
    else: print 'unrecognized sdname'
    
    # Define RHS
    # In this case is simply L*u
    sd.rhs=lambda t,u : np.dot(sd.L,u)
    
    # Set final time
    sd.T = 5.9


    # Construct exact solution by using a large number of cells. 
    # Probably this is not the best place to construct it because if one want
    # to change the the initial solution he must also change the intial condition
    # in the spatial discretization.
    #############################################################################
    nbrCellsExact = 2000
    sd.xExact = np.linspace(xMin,xMax,nbrCellsExact)   # Position of the interfaces
    sd.uExactInit = np.zeros(nbrCellsExact)
    sd.uExactInit = np.sin(2*np.pi*sd.xExact)

    # Compute the real final position of the last point. 
    # One point is enough because it is a rigid translation in the x direction
    xExactTmp = sd.xExact+sd.T
    xLastPnt = xExactTmp[nbrCellsExact-1]
  
    from math import modf

    b,a = modf(xLastPnt)

    
    # 1st approach: valid for classical periodic initial condition, e.g. sin(), cos(), etc.
    sd.uExact = np.zeros((nbrCellsExact))
    sd.uExact = np.sin(2*np.pi*(sd.xExact-b))
    
    # 2nd approach: general
    ##############
    #if b != 0.:
    #    # Copy part of the solution on the left
    #    for i in range(0,nbrCellsExact):
    #        if  xExactTmp[i] > a:
    #            index = i
    #            break
    #    # New exact solution with periodic BC
    #    sd.uExact[0] = sd.uExactInit[index-1]
    #    sd.uExact[1:nbrCellsExact-index+1] = sd.uExactInit[index:nbrCellsExact]
    #    sd.uExact[nbrCellsExact-index+1:nbrCellsExact+1] = sd.uExactInit[0:index]
    #        
    #    sd.xExact = np.linspace(xMin,xMax,nbrCellsExact+1)
    #else:
    #    sd.uExact = sd.uExact
    #    sd.xExact = np.linspace(xMin,xMax,nbrCellsExact)

    return sd


def upwind_advection_matrix(N,dx):
    #from scipy.sparse import spdiags
    e=np.ones(N)
    #L=spdiags([-e,e,e],[0,-1,N-1],N,N)/dx
    L=(np.diag(-e)+np.diag(e[1:],-1))
    L[0,-1]=1.
    L=L/dx
    return L

def weno5_linearized_matrix(N,dx=0):
    import scipy.linalg
    if dx == 0: dx = 1./N
    e=np.zeros(N)
    e[0]  =  1./3
    e[-1] =  5./6
    e[-2] = -1./6
    L2a = scipy.linalg.circulant(e)
    e     =  np.zeros(N)
    e[1]  = -1./6
    e[0]  =  5./6
    e[-1] =  1./3
    L2b = scipy.linalg.circulant(e)
    e     =  np.zeros(N)
    e[2]  =  1./3
    e[1]  = -7./6
    e[0]  = 11./6
    L2c = scipy.linalg.circulant(e)
    L2 = 3./10*L2a + 3./5*L2b + 1./10*L2c
    last_column = L2[:,0].reshape(N,1)
    L1 = np.hstack((L2[:,1:],last_column))
    L = (L1-L2)/dx
    return L
    

def centered_advection_matrix(N,dx=0):
    "3-point centered difference approximation of first derivative."
    if dx==0: dx=1./N
    e=np.ones(N)
    L=(np.diag(-0.5*e[1:],1)+np.diag(0.5*e[1:],-1))
    L[0,-1]=0.5
    L[-1,0]=-0.5
    L=L/dx
    return L

def centered_diffusion_matrix(N,dx=0):
    "3-point centered difference approximation of second derivative."
    if dx==0: dx=1./N
    e=np.ones(N)
    L=(np.diag(e[1:],1)-2.*np.diag(e,0)+np.diag(e[1:],-1))
    L[0,-1]=1.
    L[-1,0]=1.
    L=L/dx**2
    return L

def centered_advection_diffusion_matrix(a,b,N,dx=0):
    r"3-point centered difference approximation for `a u_x + b u_{xx}`."
    if dx==0: dx=1./N
    L1 = centered_advection_matrix(N,dx)
    L2 = centered_diffusion_matrix(N,dx)
    L = a*L1 + b*L2
    return L


def spectral_difference_matrix(nbrCells,xCenter,dx,order):
    np.set_printoptions(threshold=np.nan)


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
        fluxPnts = np.array([ -1.0 , -0.57 , 0.57 , 1.0 ])
        solPnts = np.array([ -1.0 ,         0.57 , 1.0 ])
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
       raise Exception("Error: min order 1, max order 6")


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



    # Flux at the boundaries flux points
    for iSol in range(0,nbrSolPnts):
        # Left face
        fluxLeftCell[0,iSol] = faceLeft*extrSolToFlux[nbrFluxPnts-1,iSol] 
        fluxCurrCell[0,iSol] = faceRight*extrSolToFlux[0,iSol]
    
        # Right face
        fluxCurrCell[nbrFluxPnts-1,iSol] = faceLeft*extrSolToFlux[nbrFluxPnts-1,iSol]
        fluxRightCell[nbrFluxPnts-1,iSol] = faceRight*extrSolToFlux[0,iSol]


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
            derivFluxInSolPnts[iSol,iFlux] = derivFluxInSolPnts[iSol,iFlux]*2.0/dx



    # Compute the products between the matrix of the derivatives and the solution contributions
    ###########################################################################################
    DMm1 = np.dot(derivFluxInSolPnts,fluxLeftCell)
    DM0  = np.dot(derivFluxInSolPnts,fluxCurrCell)
    DMp1 = np.dot(derivFluxInSolPnts,fluxRightCell)

    # Create block tridiagonal matrix
    #################################  
    dimL = nbrSolPnts*nbrCells
    L = np.zeros((dimL,dimL))

    # Main block diagonal
    for iBlock in range(0,nbrCells):
        L[nbrSolPnts*iBlock:nbrSolPnts*(iBlock+1),nbrSolPnts*iBlock:nbrSolPnts*(iBlock+1)] = DM0[:,:] 
    
    # Lower and upper blocks diagonal
    for iBlock in range(0,nbrCells-1):
        L[nbrSolPnts*(iBlock+1):nbrSolPnts*(iBlock+2),nbrSolPnts*iBlock:nbrSolPnts*(iBlock+1)] = DMm1[:,:]
        L[nbrSolPnts*iBlock:nbrSolPnts*(iBlock+1),nbrSolPnts*(iBlock+1):nbrSolPnts*(iBlock+2)] = DMp1[:,:]


    # Add periodic boundary condition contributions to the operator L
    #################################################################
    # Apply BC to the first cell --> DMm1
    L[0:nbrSolPnts,nbrSolPnts*(nbrCells-1):nbrSolPnts*nbrCells] = DMm1[:,:]    

    # Apply BC to the last cell --> DMp1
    # Actually here we are doing nothing because it is a fully upwind scheme with 1D advection equation 
    L[nbrSolPnts*(nbrCells-1):nbrSolPnts*nbrCells,0:nbrSolPnts] = DMp1[:,:]  


    # Bring spatial discretization on the RHS of the equation
    #########################################################
    L = -1*L
  
    # Construct initial solution
    ############################
    u0 = np.zeros((dimL))

    # Create solution points position
    # Here the solution points at the interface are reated two times because of the
    # spectral difference method
    xSolPnts = np.zeros(dimL)
    for i in range(0,nbrCells):
        for j in range(0,nbrSolPnts):
            xSolPnts[i*nbrSolPnts+j] = xCenter[i] + 1./2.*dx*solPnts[j]

    u0 = np.sin(2*np.pi*xSolPnts)

   
    return L,xSolPnts,u0




