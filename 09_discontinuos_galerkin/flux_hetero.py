"""
Created on Wed Feb 17 20:46:08 2016
"""
from numpy import * 
          
def flux(Q, N, ne, Ap, Am):
    """
    calculates the flux between two boundary sides of 
    connected elements for element i
    """
    # for every element we have 2 faces to other elements (left and right)
    out = zeros((ne,N+1,2))

    # Calculate Fluxes inside domain
    for i in range(1, ne-1):
        out[i,0,:] = Ap[i,:,:] @ (-Q[i-1,N,:]) + Am[i,:,:] @ (-Q[i,0,:])                 
        out[i,N,:] = Ap[i,:,:] @ Q[i,N,:] + Am[i,:,:] @ Q[i+1,0,:]
        
    # Boundaries
    # Left
    out[0,0,:] = Ap[0,:,:] @ array([0,0]) + Am[0,:,:] @ (-Q[i,0,:])        
    out[0,N,:] = Ap[0,:,:] @ Q[0,N,:] + Am[0,:,:] @ Q[1,0,:]

    # Right
    out[ne-1,0,:] = Ap[ne-1,:,:] @ (-Q[ne-2,N,:]) + Am[ne-1,:,:] @ -Q[ne-1,0,:]    
    out[ne-1,N,:] = Ap[ne-1,:,:] @ (Q[ne-1,N,:]) + Am[ne-1,:,:] @ array([0, 0])

    return out

##*****************************************************************************
## FLUX TEST
##*****************************************************************************
#N = 2
#ne = 4
#Q = ones([ne, N+1, 2])
#
## Inialize Flux relates matrices
##c = 10
#rho0 = 2
#c0 = 3
#mu0 = rho0*c0**2
#
##c = zeros(ne)
#rho = zeros(ne)
#mu = zeros(ne)
#
#Ap = zeros((ne,2,2))
#Am = zeros((ne,2,2))
#
## initialize c, rho, mu, and Z
##c = c + c0
#rho = rho + rho0
#rho[ne/2:ne] = .25*rho[ne/2:ne]
#mu = mu + mu0
#c = sqrt(mu/rho)
#Z = rho*c
#
### Initialize flux matrices
#for i in range(1,ne-1):
#    #Z[i]=rho[i]*sqrt(mu[i]/rho[i])
#    # Left side positive direction    
#    R = array([[Z[i], -Z[i]], [1, 1]])
#    L1 = array([[0, 0], [0, c[i]]])
#    Ap[i,:,:] = R @ L1 @ linalg.inv(R)
#    
#    # Right side negative direction    
#    R = array([[Z[i], -Z[i]], [1, 1]])
#    L2 = array([[-c[i], 0 ], [0, 0]])    
#    Am[i,:,:] = R @ L2 @ linalg.inv(R)
#    
#Flux = flux(Q, N, ne, Ap, Am)
#
##*****************************************************************************