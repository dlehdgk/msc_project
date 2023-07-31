# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 13:10:26 2023

@author: Dong Ha
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op
from scipy.integrate import solve_ivp
from scipy.special import zeta, polygamma, factorial
import camb
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
def X_eq(z):
    return (-1+np.sqrt(1+4*Saha(z)))/(2*Saha(z))
def X_num(z):
    return 2.4e-3*np.sqrt(Om)*(z/1000)**12.75/(Obh2/h)
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

z = np.linspace(800, 1800, 500, dtype='float32')

#CAMB
#set_params is a shortcut routine for setting many things at once
pars = camb.set_params(H0=h*100, ombh2=Obh2, omch2=0.122, As=2e-9, ns=0.95)
data= camb.get_background(pars)
back_ev = data.get_background_redshift_evolution(z, ['x_e', 'visibility'], format='array')
X_camb = back_ev[:,0]

#Saha
X_Saha = X_eq(z)

#Paddy
z_valid = np.linspace(800, 1200, 300)
X_Paddy = X_num(z_valid)

#Differential
#sol = solve_ivp(func, [z[0], z[-1]], [0])
#z_diff = sol.t
#X_diff = np.swapaxes(sol.y, 0, 1)

#plots
#plt.plot(z, X_camb, label = 'CAMB')
plt.plot(z, X_Saha, label = 'Saha')
#plt.plot(z_valid, X_Paddy, label = 'Paddy Solution')
#plt.plot(z_diff, X_diff, label = 'Solution to ODE')
plt.grid()
#plt.legend()
plt.xlabel('Redshift')
plt.ylabel('Ionisation Fraction')
plt.savefig('Images/Saha.png', dpi=300)
#plt.savefig('Images/ionisation_fraction.png', dpi=300)
plt.show()
#%%

tau = back_ev[:,1]

plt.plot(z, tau)
plt.grid()
plt.legend()
plt.xlabel('redshift')
plt.ylabel('Optical Depth')
plt.show()
#%%
fig, axs= plt.subplots(1,2, figsize=(12,5))
for i, (ax, label), in enumerate(zip(axs, ['$x_e$','Visibility'])):
    ax.semilogx(z, back_ev[:,i])
    ax.set_xlabel('$z$')
    ax.set_ylabel(label)
    ax.set_xlim([500,1e4])
    