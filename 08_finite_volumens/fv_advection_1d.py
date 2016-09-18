"""
Created on Thu Feb  4 09:56:10 2016
"""
#------------------------------------------------------------------------------
#CHAPTER 8:  The Finite-Volume Method
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Initialization of setup
# ---------------------------------------------------------------
nx = 6000         # number of grid points
xmax = 75000      # in m
c = 2500          # Advected speed
eps = 0.5         # CFL
tmax = 2          # simulation time in s
sig = 200         # in m
x0 = 1000         # in m
method = 'Lax-Wendroff' # Select: 'Lax-Wendroff' or 'upwind'
isnap = 10 

# Space
x = np.linspace(0,xmax,nx)
dx = min(np.diff(x))

# use wave based CFL criterion
dt = eps*dx/c # calculate tim step from stability criterion

# Simulation time
nt = int(np.floor(tmax/dt))

# Initialize shape of fields
Q = np.zeros(nx)
dQ = np.zeros(nx)
dQ1 = np.zeros(nx)
dQ2 = np.zeros(nx)
Qa = np.zeros(nx)

#Spatial initial condition
sx = np.exp(-1./sig**2 * (x-x0)**2)
# Set Initial condition 
Q = sx

#---------------------------------------------------------------------------
# Initialize animated plot
plt.figure()
plt.ion() # set interective mode
plt.show()
#---------------------------------------------------------------------------

#%% CODE 15: Listing 8.1 Finite volume method: scalar advection - Pag 194
# [...]
# Time extrapolation
for j in range(nt): 
    # upwind
    if method == 'upwind': 
        for i in range(1, nx-1):
            # Forward (upwind) (c>0)
            dQ[i] = Q[i] - Q[i-1]
        # Time extrapolation 
        Q = Q - dt/dx*c*dQ        
    # Lax wendroff
    if method == 'Lax-Wendroff': 
        for i in range(1, nx-1):
            # Forward (upwind) (c>0)
            dQ1[i] = Q[i+1] - 2*Q[i] + Q[i-1]
            dQ2[i] = Q[i+1] - Q[i-1]
        # Time extrapolation 
        Q = Q - 0.5*c*dt/dx*dQ2 + 0.5*(c*dt/dx)**2 *dQ1        
    # Boundary condition     
    Q[0] = Q[nx-2] # Periodic    
    Q[nx-1] = Q[nx-2] # Absorbing
# [...]
        
    if not j % isnap:           
        # Analytical solution
        xd = c*j*dt+x0
        Qa = np.exp(-1./sig**2 * (x - xd)**2)
        
        # Plotting
        plt.clf() # clear current figure 
        plt.plot(x, Q, color="black", lw = 1.5)
        plt.plot(x, Qa, color="red", lw = 1.5)
        plt.xlim([xd-600, xd+600]) # window size 1200 m
        plt.text(xd+300, 0.8, '$x$ =%d m'% xd)        
               
        plt.title('Scalar Advection')
        plt.xlabel(' x (m)')
        plt.ylabel(' Amplitude ')
        plt.draw() 