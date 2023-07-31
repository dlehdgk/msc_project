# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 05:41:56 2023

@author: Dong Ha
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, cuda

@jit(target_backend='cuda') 
def Hubble(z, h, Om, Ol):
    Or = 1-Om-Ol
    return 3.241e-18*h*np.sqrt(Or*(1+z)*(1+z)*(1+z)*(1+z)+Om*(1+z)*(1+z)*(1+z)+Ol)

z = np.linspace(0, 2000, 30000)
#z = 10**(np.linspace(0, 7, 5000))
#array of parameters [i, j] i = parameters h, Om, Ol j = min, max

h68 = np.array([[68.1-0.74, 68.1+0.74], [0.02234-0.00013+0.121-0.0013, \
                                        0.02234+0.00013+0.121+0.0013],\
               [0.6894-0.0076, 0.6894]])

h68[0,:] /= 100
h68[1,0] /= (h68[0,0]**2)
h68[1,1] /= (h68[0,1]**2)
h73 = np.array([[73.6-1.1, 73.6+1.1], [0.334-0.018, 0.334+0.018], [0.666+0.018, 0.666-0.018]])

h73[0,:] /= 100

H68 = np.empty(2, object)
H73 = np.empty(2, object)

for i in range(2):
    H68[i] = Hubble(z, h68[0,i], h68[1,i], h68[2,i])
    H73[i] = Hubble(z, h73[0,i], h73[1,i], h73[2,i])

#plt.plot(z, H67[1], label='SPT-3G CMB', color = 'green')
plt.fill_between(z, H68[0], H68[1], color='black', alpha=0.4, label='SPT-3G CMB')
#plt.plot(z, H73[1], label='Pantheon+ SNe 1a')
plt.fill_between(z, H73[0], H73[1], color='purple', alpha=0.4, label='Pantheon+ SNe 1a')
plt.xlabel('Redshift')
plt.ylabel('Hubble Parameter [$s^{-1}]$')
#plt.xscale('log')
plt.grid()
plt.legend(loc='upper left')
plt.savefig('Images/Hubble.png', dpi=300)
plt.show()

#%%
delta = Hubble(1100, h68[0,:], h68[1,:], h68[2, :])-Hubble(1100,  h73[0,:], h73[1,:], h73[2,:])
print(0.5*(delta[0]+delta[1]),'+-', 0.5*(delta[0]-delta[1]))
