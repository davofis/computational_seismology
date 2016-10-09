# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 22:24:26 2016

@author: david
"""
# This is a configuration step for the exercise. Please run it before calculating the derivative!
import numpy as np
import matplotlib.pyplot as plt

# Function for setting up the Chebyshev derivative matrix
def get_cheby_matrix(nx):
    cx = np.zeros(nx+1)
    x = np.zeros(nx+1)
    for ix in range(0,nx+1):
        x[ix] = np.cos(np.pi * ix / nx)
   
    cx[0] = 2.
    cx[nx] = 2.
    cx[1:nx] = 1.
   
    D = np.zeros((nx+1,nx+1))
    for i in range(0, nx+1):
        for j in range(0, nx+1):
            if i==j and i!=0 and i!=nx:
                D[i,i]=-x[i]/(2.0*(1.0-x[i]*x[i]))
            else:
                D[i,j]=(cx[i]*(-1)**(i+j))/(cx[j]*(x[i]-x[j]))
  
    D[0,0] = (2.*nx**2+1.)/6.
    D[nx,nx] = -D[0,0]
    return D  

# Initialize arbitrary test function on Chebyshev collocation points
nx = 199     # Number of grid points
x = np.zeros(nx+1)
for ix in range(0,nx+1):
    x[ix] = np.cos(ix * np.pi / nx) 
dxmin = min(abs(np.diff(x)))
dxmax = max(abs(np.diff(x)))

# Function example: Gaussian
# Width of Gaussian
s = .2
# Gaussian function (modify!)
f = np.exp(-1/s**2 * x**2)
# Analytical derivative
df_ana = -2/s**2 * x * np.exp(-1/s**2 * x**2)


# Calculate numerical derivative using differentiation matrix
# Initialize differentiation matrix
D = get_cheby_matrix(nx)
df_num = D @ f

# Calculate error between analytical and numerical solution
err = np.sum((df_num - df_ana)**2) / np.sum(df_ana**2) * 100


# Plot analytical and numerical result
plt.plot(x,f,'b',label='f(x)')
plt.plot(x,df_num,'r',label='d/dx f(x) - numerical',alpha=0.6)
plt.plot(x,df_ana,'r--',label='d/dx f(x) - analytical')
plt.xlabel('x')
plt.ylabel('f(x) and d/df f(x)')
plt.title('Error: %s %%'%err)
plt.legend(loc='upper right')
plt.show()