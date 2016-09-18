"""
Created on Mon Feb  1 10:08:31 2016
"""
#------------------------------------------------------------------------------
#CHAPTER 6:  The Finite-Element Method
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Basic parameters
nt = 1000    # number of time steps
vs = 3000    # acoustic velocity
ro0 = 2500   # Density
isnap = 250  # snapshot frequency
nx = 1000    # number of grid points 
isx = 500    # source location
xmax = 10000.
eps = 0.5    # stability limit

dx = xmax/(nx-1)  # calculate space increment
x = np.arange(0, nx)*dx   # initialize space coordinates
x = x.T

h = np.diff(x)  # Element sizes

# parameters
ro = x*0 + ro0
mu = x*0 + ro*vs**2

# time step from stabiity criterion
dt = 0.5*eps*dx/np.max(np.sqrt(mu/ro))

# source time function
pt = 20*dt
t = np.arange(1, nt+1)*dt  # initialize time axis
t0 = 3*pt
src = -1/pt**2*(t-t0)*np.exp(-1/pt**2*(t-t0)**2)

# Source vector
f = np.zeros(nx); f[isx:isx+1] = f[isx:isx+1] + 1.

# Stiffness matrix Kij
K = np.zeros((nx,nx))
for i in range(1, nx-1):
    for j in range(1, nx-1):
        if i==j:
            K[i,j] = mu[i-1]/h[i-1] + mu[i]/h[i]
        elif i==j+1:
            K[i,j] = -mu[i-1]/h[i-1]
        elif i+1==j:
            K[i,j] = -mu[i]/h[i]
        else:
            K[i,j] = 0
# Corner element
K[0,0] = mu[0]/h[0]
K[nx-1,nx-1] = mu[nx-1]/h[nx-2]

#%% CODE 10: Listing 6.4 Mass matrix with varying element size - Pag 147
# Mass matrix M_ij
M = np.zeros((nx,nx))
for i in range(1, nx-1):
    for j in range (1, nx-1):
        if j==i:
            M[i,j] = (ro[i-1]*h[i-1] + ro[i]*h[i])/3
        elif j==i+1:
            M[i,j] = ro[i]*h[i]/6
        elif j==i-1:
            M[i,j] = ro[i-1]*h[i-1]/6
        else:
            M[i,j] = 0
# Corner element
M[0,0] = ro[0]*h[0]/3
M[nx-1,nx-1] = ro[nx-1]*h[nx-2]/3
# Invert M
Minv = np.linalg.inv(M)

# Initialize FD matrices for comparison in the regular grid case
Mf = np.zeros((nx,nx), dtype=float)
D = np.zeros((nx,nx), dtype=float)
dx = h[1]

for i in range(nx):
    Mf[i,i] = 1./ro[i]
    if i>0:
        if i<nx-1:
            D[i+1,i] =1
            D[i-1,i] =1
            D[i,i] = -2
            
D = ro0*vs**2*D/dx**2

# Initialize fields
u = np.zeros(nx)
uold = np.zeros(nx)
unew = np.zeros(nx)

U = np.zeros(nx)
Uold = np.zeros(nx)
Unew = np.zeros(nx)

fig = plt.figure(figsize=(14,8), dpi=80) 
fig.suptitle("1D Elastic wave solution", fontsize=16)
iplot = 0
#%% CODE 09: Listing 6.3 Time extrapolation - Pag 147
#   CODE 11: Listing 6.5 1D elastic case _ Pag 148
# Time extrapolation
for it in range(nt):
    # Finite Difference Method
    Unew = (dt**2)*Mf @ (D @ U + f/dx*src[it]) + 2*U - Uold 
    Uold, U = U, Unew
    
    # Finite Element Method
    unew = (dt**2)*Minv @ (f*src[it] - K @ u) + 2*u - uold                             
    uold, u = u, unew
    
    # Display both
    if np.mod(it+1, isnap) == 0:
        # extract window
        xc = 500*dx + it*dt*vs - 150
        xd = 300
        iplot += 1
        
        plt.subplot(4,1,iplot)
        L1 = plt.plot(x, u, label='FEM')
        L2 = plt.plot(x, U, label='FDM')
        plt.legend()        
        plt.text(xc+1.5*xd, 0.00000002, '%d m' %(xc-500*dx))

plt.savefig('Fig_6.10.png')        
plt.show()
