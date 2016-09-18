# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 06:32:15 2016
"""

import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.integrate import quad
from numpy import sin, cos, arccos, arctan,  pi, sign, sqrt
from numpy import vectorize, linspace, asarray, outer, diff, savetxt

def sph2cart(r, th, phi):
    """
    Transform spherical coordinates to cartesian
    """
    x = r * sin(th) * cos(phi)
    y = r * sin(th) * sin(phi)
    z = r * cos(th)   
    return x, y, z
    
def cart2sph(x, y, z):
    '''
    Transform cartesian coordinates to spherical
    '''
    r = sqrt(x**2 + y**2 + z**2)
    th = arccos(z/r)
    phi = arctan(y/x)
    return r, th, phi 
    
#%% Initialization of setup
# -----------------------------------------------------------------------------
x = 2500   # x receiver coordinate 
y = 2500   # y receiver coodinate
z = 2500   # z receiver coodinate

rho = 2500                # Density kg/m^3 
beta = 3000               # S-wave velocity
alpha = sqrt(3)*beta      # p-wave velocity

stf = 'gauss'             # Set the desired source time function 'heaviside' , 'gauss'
Trise = 0.25              # Rise time used in the source time function 
Mo = 10E16                # Scalar Moment 
    
r, th, phi = cart2sph(x, y, z)     # spherical receiver coordinates  

tmin = r/alpha - 2*Trise           # Minimum observation time 
tmax = r/beta + Trise + 2*Trise    # Maximum observation time 

# SOURCE TIME FUNCTION
# -----------------------------------------------------------------------------   
if stf == 'heaviside':
    M0 = lambda t: 4*Mo*0.5*(sign(t) + 1)
if stf == 'gauss':
    M0 = lambda t: 4*Mo*(1 + erf(t/Trise))

#******************************************************************************
# COMPUTE AKI & RICHARDS SOLUTION
#******************************************************************************
# Scalar factors int the AKI & RICHARDS solution
# -----------------------------------------------------------------------------
CN  = (1/(4 * pi * rho)) 
CIP = (1/(4 * pi * rho * alpha**2))
CIS = (1/(4 * pi * rho * beta**2))
CFP = (1/(4 * pi * rho * alpha**3))
CFS = (1/(4 * pi * rho * beta**3))

# Radiation patterns: near(AN), intermedia(AIP,AIS), and far(AFP,AFS) fields  
# -----------------------------------------------------------------------------
def AN(th, phi):    
    AN = [[9*sin(2*th)*cos(phi), -6*cos(2*th)*cos(phi), 6*cos(th)*sin(phi)]]
    return asarray(AN)
    
def AIP(th, phi):    
    AIP = [[4*sin(2*th)*cos(phi), -2*cos(2*th)*cos(phi), 2*cos(th)*sin(phi)]]
    return asarray(AIP)
    
def AIS(th, phi):    
    AIS = [-3*sin(2*th)*cos(phi), 3*cos(2*th)*cos(phi), -3*cos(th)*sin(phi)]
    return asarray(AIS)
    
def AFP(th, phi):    
    AFP = [sin(2*th)*cos(phi), 0, 0 ]
    return asarray(AFP)
    
def AFS(th, phi):    
    AFS = [0, cos(2*th)*cos(phi), -cos(th)*sin(phi)]
    return asarray(AFS)

# Calculate integral in the right hand side of AKI & RICHARDS solution
# -----------------------------------------------------------------------------
integrand = lambda  tau, t: tau*M0(t - tau)

def integral(t):
    return quad(integrand, r/alpha, r/beta, args=(t))[0]

vec_integral = vectorize(integral)

# Assemble the total AKI & RICHARDS solution
# -----------------------------------------------------------------------------
t = linspace(tmin, tmax, 1000) 
UN =   CN * (1/r**4) * outer(AN(th, phi), vec_integral(t))
UIP = CIP * (1/r**2) * outer(AIP(th, phi), M0(t - r/alpha)) 
UIS = CIS * (1/r**2) * outer(AIS(th, phi), M0(t - r/beta))

t, dt = linspace(tmin, tmax, 1001, retstep=True) # diff() return N-1 size vector  
UFP = CFP * (1/r) * outer(AFP(th, phi), diff(M0(t - r/alpha))/dt)
UFS = CFS * (1/r) * outer(AFS(th, phi), diff(M0(t - r/beta))/dt)
t = linspace(tmin, tmax, 1000) 

U = UN + UIP + UIS + UFP + UFS

Ur, Uth, Uphi = U[0,:], U[1,:], U[2,:]  # spherical componets of the field u 
Ux, Uy, Uz = sph2cart(Ur, Uth, Uphi)    # spherical to cartesian coordinates
#******************************************************************************

# Plotting
# -----------------------------------------------------------------------------
seis = [Ux, Uy, Uz, Ur, Uth, Uphi]  # Collection of seismograms
labels = ['$U_x(t)$','$U_y(t)$','$U_z(t)$','$U_r(t)$','$U_\theta(t)$','$U_\phi(t)$']
cols = ['b','r','k','g','c','m']

# Initialize animated plot
fig = plt.figure(figsize=(14,8), dpi=80)

fig.suptitle("Seismic Wavefield of a Double-Couple Point Source", fontsize=16)
plt.ion() # set interective mode
plt.show()

for i in range(6):              
    st = seis[i]
    ax = fig.add_subplot(2, 3, i+1)
    ax.plot(t, st, lw = 1.5, color=cols[i])  
    ax.set_xlabel('Time(s)')
    ax.text(tmin+0.8*(tmax-tmin), 0.7*max(st), labels[i])
    
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

savetxt('seis.csv', (t, Ux, Uy, Uz, Ur, Uth, Uphi))  # Export the data as seis.csv in the given order     
plt.savefig('Fig_2.4.png')  # save the figure
plt.show()  
