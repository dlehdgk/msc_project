# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:49:39 2023

@author: Dong Ha
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import zeta
from numba import jit, cuda

#general units

B = 13.6
T0 = 2.4e-4
z3 = zeta(3)
pi = np.pi
eta = 6.05e-10
me = 0.511e6
hc = 1.2398e-6

@jit(target_backend='cuda') 
def neg_nuc(s, X):
    H = 1.1/(s**4)
    lnp = 0.29*(s**2+6*s+12)/(s**5)
    return -lnp*((1-X)*np.exp(-s)-X)/(H*s)

@jit(target_backend='cuda') 
def pos_nuc(s, X):
    H = 1.1/(s**4)
    lnp = 0.29*(s**2+6*s+12)/(s**5)
    return lnp*((1-X)*np.exp(-s)-X)/(H*s)

neg = solve_ivp(neg_nuc, [0.01, 2], [0.5])
pos = solve_ivp(pos_nuc, [0.01, 2], [0.5])

neg_z = neg.t
pos_z = pos.t

neg_X = np.swapaxes(neg.y, 0, 1)
pos_X = np.swapaxes(pos.y, 0, 1)

fig, ax = plt.subplots(1,2, figsize = (12,4))
ax[0].plot(neg_z, neg_X, color = 'black')
ax[0].set_title('negative',fontsize=13)
ax[0].set_ylabel('Neutron fraction [X]',fontsize=13)
ax[0].set_xlabel('Q/T',fontsize=13)
ax[0].grid()

ax[1].plot(pos_z, pos_X, color = 'orange')
ax[1].set_title('positive',fontsize=13)
ax[1].set_ylabel('Neutron fraction [X]',fontsize=13)
ax[1].set_xlabel('Q/T',fontsize=13)
ax[1].grid()
fig.savefig('Images/Nucleogenesis.png', dpi=300)

#shows notes are wrong
