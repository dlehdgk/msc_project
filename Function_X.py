# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:13:35 2023

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
h = 0.673
Orh2 = 2.473e-5
Om = 0.315
Ol = 0.685
Obh2 = 0.023

@jit(target_backend='cuda') 
def T(z):
    return T0*(1+z)
@jit(target_backend='cuda') 
def n_g(z):
    return (T(z)/hc)**3*16*pi*z3
@jit(target_backend='cuda') 
def Saha(z):
    return 4*np.sqrt(2/pi)*z3*eta*(T(z)/me)**1.5*np.exp(B/T(z))
@jit(target_backend='cuda') 
def Hu(z, h):
    H0 = 100*h*1e3/3.085e22
    return H0*np.sqrt(Orh2/h/h*(1+z)**4+Om*(1+z)**3+Ol)

# finding excited state solutions
@jit(target_backend='cuda') 
def func(z, X):
    alpha = 9.78*r0**2*c*np.sqrt(B/T(z))*np.log(B/T(z))
    beta = n_g(z)*eta/Saha(z)
    return alpha*(beta*(1-X)-n_g(z)*eta*X**2)/(Hu(z, h)*(1+z))

#Differential
sol = solve_ivp(func, [0, 1700], [0])
z_diff = sol.t
X_diff = np.swapaxes(sol.y, 0, 1)

plt.plot(z_diff, X_diff, label = 'Solution to ODE')
plt.grid()
plt.legend()
plt.xlabel('Redshift [z]')
plt.ylabel('Ionisation Fraction [$X_e$]')
plt.savefig('Images/Precise X.png', dpi=300)
plt.show()