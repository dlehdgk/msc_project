# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 08:30:39 2023

@author: Dong Ha
"""

import numpy as np
from scipy.special import zeta, polygamma, factorial

#constants
kb = 1.380649e-23
h = 6.62607015e-34
hbar = h/2/np.pi
c = 2.99792458e8
me = 9.1093837015e-31
e = 1.602176634e-19
eps0 = 8.8541878128e-12
mp = 1.67262192369e-27
B = 13.6*e
G = 6.674e-11
alp_fs = e*e/(2*eps0*h*c)
st = 8*np.pi*(alp_fs*hbar/me/c)**2/3

# convert z to T
def T(z):
    T = 2.725*(1+z)
    return T
#photon number density
def n_gamma(z):
    ng = (kb*T(z)/h/c)**3*16*np.pi*zeta(3)
    return ng

#eta = baryon/photon

def eta(Ob = 0.023):
    n_bary = Ob*3e10/(3.0857e22**2*mp*8*np.pi*G)
    e = n_bary/n_gamma(0)
    return e

#Saha equation
def Saha(z, Ob):
    S = 4*np.sqrt(2/np.pi)*zeta(3)*eta(Ob)*(kb*T(z)/me/c/c)**1.5*np.exp(B/kb/T(z))
    return S

#mass fraction

def m_frac(z, Ob = 0.023):
    X = (-1+np.sqrt(1+4*Saha(z, Ob)))/(2*Saha(z, Ob))
    return X

#Thompson scattering width

def Thomp(z, X):
    G = X*eta()*(kb*T(z)/h/c)**3*16*np.pi*zeta(3)*st*c
    return G

#hubble parameter
def Hubble(z, H0, Or, Om, Ok, OL):
    E_square = Or*(1+z)**4 + Om*(1+z)**3 + Ok*(1+z)**2 + OL
    E = np.sqrt(E_square)
    H = H0*E
    return H

#numerical Xe
def X_num(z, Onr, Ob):
    return 2.4e-3*np.sqrt(Onr)*(z/1000)**12.75/Ob

#Numerical probability
def prob(z):
    return 5.26e-3*(z/1000)**13.25*np.exp(-0.37*(z/1000)**14.25)