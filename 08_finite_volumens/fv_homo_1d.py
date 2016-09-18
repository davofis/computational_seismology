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
c = 2500            # acoustic velocity in m/s
ro = 2500           # density in kg/m^3
Z = ro*c            # impedance
mu = ro*c**2        # shear modulus
imethod = 'Lax-Wendroff'  # 'Lax-Wendroff', 'upwind'

xmax = 10000        # in m 
eps = 0.5           # CFL
tmax = 1.5          # simulation time in s
isnap = 10
sig = 200
x0 = 5000

# --------------------------------------------------------------------------                                
# Space
x, dx = np.linspace(0,xmax,nx,retstep=True)
# use wave based CFL criterion
dt = eps*dx/c # calculate tim step from stability criterion
# Simulation time
nt = int(np.floor(tmax/dt))

# Initialize wave fields
# [...]
# Specifications
Q = np.zeros((2,nx))
Qnew = np.zeros((2,nx))
Qa = np.zeros((2,nx))
# Initial condition
#Qnew[0,:] = np.exp(-1./sig**2 * (x-x0)**2)

sx = np.exp(-1./sig**2 * (x-x0)**2)
Q[0,:] = sx
#Qnew[0,:] = np.exp(-1./sig**2 * (x-x0)**2)

# Initialize all matrices
R = np.array([[Z, -Z],[1, 1]])
Rinv = np.linalg.inv(R)
Lp = np.array([[0, 0], [0, c]])
Lm = np.array([[-c, 0], [0, 0]])
Ap = R @ Lp @ Rinv     
Am = R @ Lm @ Rinv    
A = np.array([[0, -mu], [-1/ro, 0]]) 
    
#*************************************************************************
FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=20, metadata=metadata, bitrate=1000)
#*************************************************************************
# Initialize animated plot
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
line1 = ax1.plot(x, Q[0,:], 'k', x, -Qa[0,:], 'r--', x, .5*sx,'k--')
line2 = ax2.plot(x, Q[1,:], 'k', x, -Qa[1,:], 'r--')
ax1.set_ylabel('Stress')
ax2.set_ylabel('Velocity')
ax2.set_xlabel(' x ')
plt.suptitle('Homogeneous F. volume - %s method'%imethod, size=12)

plt.ion() # set interective mode
plt.show()
#***************************************************************
with writer.saving(fig, "Fv_homo_lax_wendroff.mp4", 200):
#***************************************************************
#%% CODE 16: Listing 8.2 1D Elastic wave propagation: Lax-Wendroff scheme. - Pag 200 
    # Time  Extrapolation    
    for i in range(nt): 
        if imethod =='Lax-Wendroff':        
            for j in range(1,nx-1):
                dQ1 = Q[:,j+1] - Q[:,j-1]
                dQ2 = Q[:,j-1] - 2*Q[:,j] + Q[:,j+1]
                Qnew[:,j] = Q[:,j] - 0.5*dt/dx*(A @ dQ1)\
                + 1./2.*(dt/dx)**2 * (A @ A) @ dQ2
            # Absorbing boundary conditions
            Qnew[:,0] = Qnew[:,1]
            Qnew[:,nx-1] = Qnew[:,nx-2]
    # [...]
        elif imethod == 'upwind': 
            for j in range(1,nx-1):
                dQl = Q[:,j] - Q[:,j-1]
                dQr = Q[:,j+1] - Q[:,j]
                Qnew[:,j] = Q[:,j] - dt/dx * (Ap @ dQl + Am @ dQr)                    
            # Boundary conditions 
            Qnew[:,0] = Qnew[:,1]
            Qnew[:,nx-1] = Qnew[:,nx-2]
        else:
            raise NotImplementedError
        
        Q, Qnew = Qnew, Q

        print('it = %i' %i)
        if not i % isnap: 
            for l in line1:
                l.remove()
                del l               
            for l in line2:
                l.remove()
                del l 
                
            # Analytical solution (stress i.c.)
            Qa[0,:] = 1./2.*(np.exp(-1./sig**2 * (x-x0 + c*i*dt)**2)\
            + np.exp(-1./sig**2 * (x-x0-c*i*dt)**2))
            
            Qa[1,:] = 1/(2*Z)*(np.exp(-1./sig**2 * (x-x0+c*i*dt)**2)\
            - np.exp(-1./sig**2 * (x-x0-c*i*dt)**2))
            
            line1 = ax1.plot(x, Q[0,:], 'k', x, Qa[0,:], 'r--', x, sx,'k--')
            line2 = ax2.plot(x, Q[1,:], 'k', x, Qa[1,:], 'r--')
            plt.draw()
            #**********************
            writer.grab_frame()
            #**********************