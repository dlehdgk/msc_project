# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:09:11 2023

@author: Dong Ha
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import zeta
from numba import jit


#need to have every combination of units (gauss, metric) and sign (-,+)
#general units

B = 13.6
T0 = 2.4e-4
z3 = zeta(3)
pi = np.pi
eta = 6.05e-10
me = 0.511e6
hc = 1.2398e-6

#unit params
#i0 = cgs i1 = metric
#0i = r0 1i = c

units = np.array([[2.82e-13, 2.82e-15], [3e10, 3e8]])

@jit(nopython=True) 
def func(z, X, i, j):
    T = T0*(1+z)
    S = 4*np.sqrt(2/pi)*z3*eta*(T/me)**1.5*np.exp(B/T)
    n_gamma = (T/hc)**3*16*pi*z3
    H = 2.2e-18*np.sqrt(5e-5*(1+z)**4+0.3*(1+z)**3+0.7)
    alpha = 9.78*units[0,j]**2*units[1,j]*np.sqrt(B/T)*np.log(B/T)
    beta = n_gamma*eta/S
    return (-1)**i*alpha*(beta*(1-X)-n_gamma*eta*X**2)/(H*(1+z))

z = np.empty([2,2], object)
X = np.empty([2,2], object)

fig, ax = plt.subplots(2,2, figsize = (12,12))
#sign
for i in range(2):
    #units
    for j in range(2):
        sol = solve_ivp(func, [800.0, 1700.0], [0], args = (i, j))
        z[i,j] = sol.t
        X[i,j] = np.swapaxes(sol.y, 0, 1)
        
        ax[i,j].plot(z[i,j], X[i,j])
        ax[i,j].set_ylabel('Ionisation fraction [X]')
        ax[i,j].set_xlabel('Redshift [z]')
        ax[i,j].grid()

ax[0,0].set_title('Positive cgs')
ax[1,0].set_title('Negative cgs')
ax[0,1].set_title('Positive metric')
ax[1,1].set_title('Negative metric')

fig.savefig('Images/Ionisation_units.png', dpi=300)