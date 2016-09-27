"""
Created on Thu Feb  4 09:50:22 2016
"""
#------------------------------------------------------------------------------
#CHAPTER 8:  The Finite-Volume Method
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialization of setup
# --------------------------------------------------------------------------
nx = 800            # number of grid points 
c0 = 2500           # acoustic velocity in m/s
ro = 2500           # density in kg/m^3
Z0 = ro*c0          # impedance
mu = ro*c0**2       # shear modulus
rho0 = ro
mu0 = mu
imethod = 'Lax-Wendroff'  # 'Lax-Wendroff', 'upwind'

xmax = 10000        # in m 
eps = 0.5           # CFL
tmax = 1.5          # simulation time in s
isnap = 10
sig = 200
x0 = 2500
iplot = 20           # Plotting frequency
# --------------------------------------------------------------------------                                
# FD 
dx = xmax/(nx-1)  # calculate space increment
xfd = np.arange(0, nx)*dx
mufd = np.zeros(xfd.size) + mu0
rhofd = np.zeros(xfd.size) + rho0
mufd[(nx-1)/2+1:nx] = mufd[(nx-1)/2+1:nx]*4

s = np.zeros(xfd.size)
v = np.zeros(xfd.size)
dv = np.zeros(xfd.size)
ds = np.zeros(xfd.size)
s = np.exp(-1/sig**2*((xfd-x0))**2)
# -------------------------------------------------------------------------- 

A = np.zeros((2,2,nx))
Z = np.zeros((1,nx))
c = np.zeros((1,nx))

# initialize velocity
c = c + c0
c[nx/2:nx] = c[nx/2:nx]*2
Z = ro*c

# Initialize A for each cell
for i in range(1,nx):    
    A0 = np.array([[0, -mu], [-1/ro, 0]])
    if i > nx/2:
        A0= np.array([[0, -4*mu], [-1/ro, 0]])
    A[:,:,i] = A0 

# -------------------------------------------------------------------------- 
# Space
x, dx = np.linspace(0,xmax,nx,retstep=True)
# use wave based CFL criterion
dt = eps*dx/np.max(c) # calculate tim step from stability criterion
# Simulation time
nt = int(np.floor(tmax/dt))
# Initialize wave fields
Q = np.zeros((2,nx))
Qnew = np.zeros((2,nx))

# Source
sx = np.exp(-1./sig**2 * (x-x0)**2)
Q[0,:] = sx

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
line1 = ax1.plot(x, Q[0,:], 'k', x, s, 'r--')
line2 = ax2.plot(x, Q[1,:], 'k', x, v, 'r--')
ax1.axvspan(((nx-1)/2+1)*dx, 
            nx*dx, alpha=0.2, facecolor='b')
ax2.axvspan(((nx-1)/2+1)*dx, 
            nx*dx, alpha=0.2, facecolor='b')
ax1.set_xlim([0, xmax])
ax2.set_xlim([0, xmax])

ax1.set_ylabel('Stress')
ax2.set_ylabel('Velocity')
ax2.set_xlabel(' x ')
plt.suptitle('Heterogeneous F. volume - %s method'%imethod, size=12)

plt.ion() # set interective mode
plt.show()
#***************************************************************
with writer.saving(fig, "Fv_hetero.mp4", 200):
#***************************************************************
#%% CODE 17: Listing 8.3 1D Elastic wave propagation. Heterogeneous case. - Pag 202 
    # [...]
    # Time  Extrapolation    
    for j in range(nt):
        for i in range(1,nx-1): # Lax-Wendroff method
            dQl = Q[:,i] - Q[:,i-1]
            dQr = Q[:,i+1] - Q[:,i]      
            Qnew[:,i] = Q[:,i] - dt/(2*dx)*A[:,:,i] @ (dQl + dQr)\
                    + 1/2*(dt/dx)**2 *A[:,:,i] @ A[:,:,i] @ (dQr - dQl)                     
        # Absorbing boundary conditions
        Qnew[:,0] = Qnew[:,1]
        Qnew[:,nx-1] = Qnew[:,nx-2]
        # [...]
        Q, Qnew = Qnew, Q
        
        # FD Extrapolation scheme---------------------------------------------
        # Stress derivative
        for i in range(1, nx-1):
            ds[i] = (s[i+1] - s[i])/dx 
        # Velocity extrapolation
        v = v + dt*ds/rhofd
        # Velocity derivative
        for i in range(1, nx-1):
            dv[i] = (v[i] - v[i-1])/dx 
        # Stress extrapolation
        s = s + dt*mufd*dv 
        print('it = %i' %j)
        #---------------------------------------------------------------------        
        # Plot  
        if not j % isnap: 
            for l in line1:
                l.remove()
                del l               
            for l in line2:
                l.remove()
                del l 
                 
            line1 = ax1.plot(x, Q[0,:], 'k', x, s, 'r--')
            line2 = ax2.plot(x, Q[1,:], 'k', x, v, 'r--')
            plt.draw()
            #**********************
            writer.grab_frame()
            #**********************       
        