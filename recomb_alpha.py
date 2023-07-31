# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 08:30:30 2023

@author: Dong Ha
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc

#constants
q = 4.8032e-10
m = 9.1095e-28
pi = np.pi
e4 = np.exp(4)
c=3e10
kb=1.3803e-16
hbar=6.262e-27/(2*pi)
B = m*q**4/(2*hbar**2)
a0 = hbar**2/(m*q**2)
T0 = 2.7255
fact = 2**10*pi**1.5/(3*e4)
sig0 = 2**9*(pi*q*a0)**2/(3*e4*hbar*c)
A = sig0*c/np.sqrt(pi)*(q*q/(hbar*c))**3

def T(z):
    return T0*(1+z)

def alph(T):
    return fact*(q*q/hbar/c)**3*a0**3*B/hbar*np.sqrt(B/(kb*T))
#approximation of sum to infty
def recombA(T):
    x1 = B/(kb*T)
    t1 = (x1+1)*np.exp(x1)*sc.exp1(x1)+np.log(x1)+0.577
    return 0.5*A*np.sqrt(x1)*t1
#manual sum to state n
def a_sum(T, start, n):
    tot = 0
    for i in range(start,n+1):
        xn = B/(i*i*kb*T)
        tot += xn**1.5*np.exp(xn)*sc.exp1(xn)
    return A*tot
def cal(T):
    return 4.2e-13*(1.6022e-12/(kb*T))**0.7

temp = np.linspace(500, 10000, 1000)
z = np.linspace(800, 1300, 1000)

ta1 = alph(temp)
ta_int = recombA(temp)
taa = cal(temp)
ta10 = a_sum(temp, 1, 10)
ta50 = a_sum(temp, 1, 50)

za1 = alph(T(z))
za_int = recombA(T(z))
zaa = cal(T(z))
za10 = a_sum(T(z), 1, 10)
za50 = a_sum(T(z), 1, 50)

fig, ax= plt.subplots(1,2, figsize=(12,5))
for i in range(2):
    ax[i].plot(temp, a1)

plt.plot(T, a1, label='n = 1 only')
plt.plot(T, aa, label='cal (2011)')
plt.plot(T, a_int, label='integrated solution')
plt.plot(T, a10, label='to n = 10')
plt.plot(T, a50, label='to n = 50')
plt.xscale('log')
plt.xlabel('Temperature [K]')
plt.ylabel('$\\alpha$ [$cm^3\cdot s^{-1}$]')
plt.grid()
plt.legend()
plt.savefig('Images/gs_sum.png', dpi=300)
plt.show()