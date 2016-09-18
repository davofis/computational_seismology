"""
Created on Mon Feb  1 10:08:31 2016
"""

import numpy as np
import matplotlib.pyplot as plt
import os
#import matplotlib.pyplot as plt
#import fib3
#from lamb import lamb


# -----------------------------------------------------------------------------
# Initialization of setup
# -----------------------------------------------------------------------------
r   = 10.0

vp  = 8.0
vs  = 4.62
rho = 3.3
nt  = 512
dt  = 0.01
h   = 0.01
ti  = 0.0

var = [vp, vs, rho, nt, dt, h, r, ti]

# -----------------------------------------------------------------------------
# Execute fortran code
# -----------------------------------------------------------------------------
with open('input.txt', 'w') as f:
    for i in var:
        print(i, file=f, end='  ')

f.close()

os.system("./lamb.exe")

# -----------------------------------------------------------------------------
# Load the solution
# -----------------------------------------------------------------------------
G = np.genfromtxt('output.txt')

u_rx = G[:,0]    # Radial displacement owing to horizontal load
u_tx = G[:,1]    # Tangential displacement due to horizontal load
u_zx = G[:,2]    # Vertical displacement owing to horizontal load

u_rz = G[:,3]    # Radial displacement owing to a vertical load
u_zz = G[:,4]    # Vertical displacement owing to vertical load

t = np.linspace(dt, nt*dt, nt)    # Time axis
#%%
# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
seis = [u_rx, u_tx, u_zx, u_rz, u_zz]  # Collection of seismograms
labels = ['$u_{rx}(t)$','$u_{tx}(t)$','$u_{zx}(t)$','$u_{rz}(t)$','$u_{zz}(t)$']
cols = ['b','r','k','g','c']

# Initialize animated plot
fig = plt.figure(figsize=(14,8), dpi=80)

fig.suptitle("Green's Function for Lamb's problem", fontsize=16)
plt.ion() # set interective mode
plt.show()

for i in range(5):              
    st = seis[i]
    ax = fig.add_subplot(2, 3, i+1)
    ax.plot(t, st, lw = 1.5, color=cols[i])  
    ax.set_xlabel('Time(s)')
    ax.text(0.8*nt*dt, 0.8*max(st), labels[i], fontsize=16)
    
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

plt.show()
#np.savetxt('seis.csv', (t, u_rx, u_tx, u_zx, u_rz, u_zz))  # Export the data as seis.csv in the given order     
#plt.savefig('green_lamb.png')  # save the figure
  