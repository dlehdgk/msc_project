# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 07:31:30 2023

@author: Dong Ha
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import zeta
from numba import jit, cuda

#constants
z3 = zeta(3)
pi = np.pi
c = 3e8
r0 = 2.82e-15
hc = 1.2398e-6
B = 13.6
me = 0.511e6
T0 = 2.4e-4
eta = 6.05e-10
h = 0.681
Orh2 = 2.473e-5
Om = 0.315
Ol = 0.6894
Obh2 = 0.023
planck = 4.135667696e-15

@jit(target_backend='cuda') 
def T(z):
    return T0*(1+z)
@jit(target_backend='cuda') 
def n_g(z):
    return (T(z)/planck)**3*16*pi*z3
@jit(target_backend='cuda') 
def Saha(z):
    return 4*np.sqrt(2/pi)*z3*eta*(T(z)/me)**1.5*np.exp(B/T(z))
@jit(target_backend='cuda') 
def X(z):
    return (-1+np.sqrt(1+4*Saha(z)))/(2*Saha(z))
@jit(target_backend='cuda') 
def Hu(z):
    H0 = 100*h*1e3/3.085e22
    return H0*np.sqrt(Orh2/h/h*(1+z)**4+Om*(1+z)**3+Ol)

@jit(target_backend='cuda')
def func(z, y):
    Gam = X(z)*eta*n_g(z)*8*pi/3*r0**2/(c**2)
    return 1/(Hu(z)*(1+z))*(Gam*(y**2-1)-3*Hu(z)*y)

#%%
z = np.linspace(1000, 1700, 500, dtype='float32')
Gam = X(z)*eta*n_g(z)*8*pi/3*r0**2/(c**2)
plt.plot(z, 1/Gam)
plt.plot(z, 1/Hu(z))
plt.yscale('log')

#%%
sol = solve_ivp(func, [900, 1300], [1])
z = sol.t
n = np.swapaxes(sol.y, 0, 1)
plt.plot(z, n)
plt.grid()
plt.xlabel('Redshift')
plt.ylabel('Electron number density')
plt.savefig('Images/Boltzmann.png', dpi=300)
plt.show()
