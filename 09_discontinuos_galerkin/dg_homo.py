"""
Created on Wed Feb 17 18:44:25 2016
"""
#------------------------------------------------------------------------------
#CHAPTER 9:  The Discontinuous Galerkin Method 
#------------------------------------------------------------------------------
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from gll import gll
from lagrange1st import lagrange1st
from flux_homo import flux

#---------------------------------------------------------------
# Initialization of setup
# ---------------------------------------------------------------
c = 2500             # acoustic velocity in m/s
tmax = 2.0           # Length of seismogram
xmax = 10000         # Length of domain
vs = 2500            # Advection velocity
rho = 2500
mu = rho*vs**2
N = 4                # Order of Lagrange polynomials
ne = 200             # Number of elements
sig = 200            # width of Gaussian initial condition
x0 = 5000            # x locartion of Gauss 
eps = 0.4            # Courant criterion
iplot = 20           # Plotting frequency 
imethod = 'Euler'       # 'Euler', 'RK'
#--------------------------------------------------------------------

# Initialization of GLL points integration weights
[xi,w] = gll(N)     # xi -> N+1 coordinates [-1 1] of GLL points
                    # w Integration weights at GLL locations
# Space domain
le = xmax/ne       # Length of elements, here equidistent
ng = ne*N + 1
# Vector with GLL points  
k=0;
xg = zeros((N+1)*ne)
for i in range(0, ne):
    for j in range(0, N+1):
        k += 1
        xg[k-1] = i*le+.5*(xi[j]+1)*le
        
x = reshape(xg, (N+1, ne), order='F').T
# ---------------------------------------------------------------

# Calculation if time step acoording to Courant criterion
dxmin = min(diff(xg[1:N+1]))
dt = eps*dxmin/vs # Global time step
nt = int(floor(tmax/dt))

# Mapping - Jacobian
J = le/2  # Jacobian
Ji = 1/J  # Inverse Jacobian

# Initialization of 1st derivative of Lagrange polynomials
l1d = lagrange1st(N)   
# Array with GLL as columns for each N+1 polynomial

# -----------------------------------------------------------------
# Initialization of system matrices
# -----------------------------------------------------------------
# Elemental Mass matrix
M = zeros((N+1, N+1))
for i in range(0, N+1):
    M[i,i] = w[i]*J 
    
# Build inverse matrix (M is diagonal!)
Minv = identity(N+1)
for i in range(0, N+1):
    Minv[i,i] = 1./M[i,i]
# ---------------------------------------------------------------
    
# Elemental Stiffness Matrix
K = zeros((N+1, N+1))
for i in range(0, N+1):
    for j in range(0, N+1):
            K[i,j] = w[j]*l1d[i,j] # NxN matrix for every element
# ---------------------------------------------------------------

# Inialize Flux relates matrices
Z = rho*vs
R = array([[Z, -Z], [1, 1]])
Rinv = linalg.inv(R)

Lm= array([[-c, 0], [0, 0]])
Lp= array([[0, 0] , [0, c]])

Ap = R @ Lp @ Rinv
Am = R @ Lm @ Rinv

A = array([[0, -mu], [-1/rho, 0]])

#%%%%%%%%%%%%%%% Time extrapolation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# DG Solution
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Initalize solution vectors
Q = zeros([ne, N+1, 2])
Qa = zeros([ne, N+1, 2])
Qnew = zeros([ne, N+1, 2])

k1 = zeros([ne, N+1, 2])
k2 = zeros([ne, N+1, 2])

Q[:,:,0] = exp(-1/sig**2*((x-x0))**2)
Qs = zeros(xg.size)  # for plotting
Qv = zeros(xg.size)  # for plotting 

#Q0[:,:,0] = np.exp(-1/sig**2*((x-x0))**2)
#Q[...] = Q0

#*************************************************************************
FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=10, metadata=metadata, bitrate=1000)
#*************************************************************************
# Initialize animated plot
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
line1 = ax1.plot(xg, Qs, 'k')
line2 = ax2.plot(xg, Qv, 'r')  
ax1.set_ylabel('Stress')
ax2.set_ylabel('Velocity')
ax2.set_xlabel(' x ')
plt.suptitle('Homogeneous Disc. Galerkin  - %s method'%imethod, size=12)

plt.ion() # set interective mode
plt.show()
#***************************************************************
with writer.saving(fig, "Dg_homo.mp4", 200):
#***************************************************************
    for it in range(nt):
        if not it % iplot:
            print(' it = %i' %it)
            
        if imethod == 'Euler': # Euler    
            # Calculate Fluxes 
            Flux = flux(Q, N, ne, Ap, Am)        
            # Extrapolate each element using flux F 
            for i in range(1,ne-1):
                Qnew[i,:,0] = dt * Minv @ (-mu * K @ Q[i,:,1].T - Flux[i,:,0].T) + Q[i,:,0].T
                Qnew[i,:,1] = dt * Minv @ (-1/rho * K @ Q[i,:,0].T - Flux[i,:,1].T) + Q[i,:,1].T                
            
        elif imethod == 'RK':                  
            # Calculate Fluxes
            Flux = flux(Q, N, ne, Ap, Am)
            
            for i in range(1,ne-1):
                k1[i,:,0] = Minv @ (-mu * K @ Q[i,:,1].T - Flux[i,:,0].T)
                k1[i,:,1] = Minv @ (-1/rho * K @ Q[i,:,0].T - Flux[i,:,1].T)   
                  
            for i in range(1,ne-1):
                Qnew[i,:,0] = dt * Minv @ (-mu * K @ Q[i,:,1].T - Flux[i,:,0].T) + Q[i,:,0].T 
                Qnew[i,:,1] = dt * Minv @ (-1/rho * K @ Q[i,:,0].T - Flux[i,:,1].T) + Q[i,:,1].T    
                  
            Flux = flux(Qnew,N,ne,Ap,Am)
           
            for i in range(1,ne-1):
                k2[i,:,0] = Minv @ (-mu * K @ Qnew[i,:,1].T - Flux[i,:,0].T)
                k2[i,:,1] = Minv @ (-1/rho * K @ Qnew[i,:,0].T - Flux[i,:,1].T) 
            # Extrapolate       
            Qnew = Q + (dt/2) * (k1 + k2)
        else:
            raise NotImplementedError
            
        Q, Qnew = Qnew, Q
            
        #----------------------------------------------------------------------         
        # Plot             
        if not it % iplot: 
            for l in line1:
                l.remove()
                del l               
            for l in line2:
                l.remove()
                del l 

            # stretch for plotting
            k = 0
            for i in range(ne):
                for j in range(N+1):
                    Qs[k] = Q[i,j,0]
                    Qv[k] = Q[i,j,1] 
                    k = k + 1
                          
            line1 = ax1.plot(xg, Qs, 'k')
            line2 = ax2.plot(xg, Qv, 'r')            
            plt.draw()
            #**********************
            writer.grab_frame()
            #**********************  






                
#            # Analytical solution (stress i.c.)
#            Qa[:,:,0] = 1./2.*(np.exp(-1./sig**2 * (x-x0 + c*it*dt)**2)\
#            + np.exp(-1./sig**2 * (x-x0-c*it*dt)**2))
#            
#            Qa[:,:,1] = 1/(2*Z)*(np.exp(-1./sig**2 * (x-x0+c*it*dt)**2)\
#            - np.exp(-1./sig**2 * (x-x0-c*it*dt)**2))
#                 
#            line1 = ax1.plot(x, Q[:,:,0], 'k', x, Qa[:,:,0], 'r--')
#            line2 = ax2.plot(x, Q[:,:,1], 'k', x, Qa[:,:,1], 'r--')
#            plt.draw()
#            #**********************
#            writer.grab_frame()
#            #**********************
            
        
