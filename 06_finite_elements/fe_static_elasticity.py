"""
Created on Mon Feb  1 10:08:31 2016
"""
#------------------------------------------------------------------------------
#CHAPTER 6:  The Finite-Element Method
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

#%% CODE 07: Listing 6.1 1D static elasticity - Pag 140
#------------------------------------------------------------------------------
# FINITE ELEMENTS SOLUTION
#------------------------------------------------------------------------------
# [...]
# Basic parameters 
nx = 20             # Number of boundary points
u = np.zeros(nx)    # Solution vector 
f = np.zeros(nx)    # Source vector 
mu = 1              # Constant shear modulus 
# Element boundary points
x = np.linspace(0, 1, nx) # x in [0,1]
h = x[2] - x[1] # Constant element size 
# Assemble stiffness matrix K_ij
K = np.zeros((nx, nx))
for i in range(1, nx-1):
    for j in range(1, nx-1):
        if i == j:
            K[i, j] = 2*mu/h
        elif i == j + 1:
            K[i, j] = -mu/h
        elif i + 1 == j:
            K[i, j] = -mu/h
        else:
            K[i, j] = 0
# Souce term is a spike at i = 15
f[15] = 1
# Boundary condition at x = 0
u[0] = 0.15 ; f[1] = u[0]/h
# Boundary condition at x = 1
u[nx-1] = 0.05 ; f[nx-2] = u[nx-1]/h
# finite element solution 
u[1:nx-1] = np.linalg.inv(K[1:nx-1, 1:nx-1]) @ f[1:nx-1].T
# [...]

# Plotting
plt.plot(x,u, color='r', lw=3.)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.axis([0, 1, 0.04, .28])

#%% CODE 08: Listing 6.2 1D static elasticity with finite differences - Pag 141 
#------------------------------------------------------------------------------
# FINITE DIFFERENCES SOLUTION
#------------------------------------------------------------------------------
# Poisson's equation with relaxation method
# non-zero boundary conditions
u  = np.zeros(nx) # set u to zero
du = np.zeros(nx) # du/dx
f  = np.zeros(nx) # forcing

nt = 500
isnap = 25

# [...]
# Forcing
f[15] = 1/h  # force vector
for it in range(nt):
    # Calculate the average of u (omit boundaries)
    for i in range(1, nx-1):
        du[i] =u [i+1] + u[i-1]
    u = 0.5*( f*h**2/mu + du )
    u[0] = 0.15    # Boundary condition at x=0
    u[nx-1] = 0.05 # Boundary condition at x=1
# [...]    
    
    # visualization    
    fd = u
    xfd = np.arange(0, nx)*h
    if np.mod(it+1, isnap) == 0:
        plt.plot(xfd, fd, color='k', ls='-.')

plt.savefig('Fig_6.7.png')
plt.show()
