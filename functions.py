# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 08:30:39 2023

@author: Dong Ha
"""

from constants import *
from numba import jit

#key functions
#temperature for z
@jit(nopython=True)
def Temp(z):
    return T0*(1+z)
#photon number density
@jit(nopython=True)
def ng(z):
    return (Temp(z)/hc)**3*16*pi*z3
#Hubble parameter
@jit(nopython=True)
def Hubble(z):
    H0 = 100*h/3.0857e19
    return H0*np.sqrt(Orh2/h/h*(1+z)**4+Omh2/h/h*(1+z)**3+Ol)
#Saha's ionisation fraction
@jit(nopython=True)
def saha(z):
    S = 4*np.sqrt(2/pi)*z3*eta*(Temp(z)/(m*c*c))**1.5*np.exp(B/Temp(z))
    X = (-1+np.sqrt(1+4*S))/(2*S)
    return X
#Padmanabhan fitting
@jit(nopython=True)
def rev_ex(z):
    amp = 2.4e-3*np.sqrt(Omh2)/0.02207
    return amp*(z/1000)**12.75
#%% recombination rates
#trapezium approximation of recombination rates to excited states
def a_ex(z):
    x2 = B/(4*Temp(z))
    expo = x2*np.exp(x2)*sc.exp1(x2)
    return A*np.sqrt(x2)*(expo*(0.5+1/x2)+0.577+np.log(x2))
#exponential fitting function
@jit(nopython=True)
def exp_fit(x, a, b):
    """
    a = rate coefficient at 1 eV
    b = power to fit
    """
    return a*(ev/Temp(x))**b

#%% functions for ionisation fraction function
@jit(nopython=True)
def K(z):
    return (c*planck/ly)**3/(8*pi*Hubble(z))
#beta/alpha
@jit(nopython=True)
def b_over_a(z):
    T = Temp(z)
    return (m*T/(2*pi*hbar**2))**1.5*np.exp(-B/(4*T))