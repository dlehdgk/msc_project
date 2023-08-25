# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 18:02:28 2023

@author: Dong Ha
"""
#document to store all constants and parameters

#imports
import numpy as np
import scipy.special as sc

#mathematical constants
z3 = sc.zeta(3)
pi = np.pi
e4 = np.exp(4)

#cosmological params
eta = 6.05e-10
h = 0.673
Orh2 = 2.473e-5
Omh2 = 0.14187
Ol = 0.685

#Gaussian cgs units
q = 4.8032e-10
m = 9.1095e-28
c=3e10
kb=1.3803e-16
hbar=1.0546e-27
planck = 6.6262e-27
hc = 6.6262e-27*c
#Binding energy of H 13.6 eV
B = m*q**4/(2*hbar**2)
#Bohr radius
a0 = hbar**2/(m*q*q)
#ground state photoionisation x-section
sig0 = 512*(pi*q*a0)**2/(3*e4*hbar*c)
#x-section from TA vol1
#sig0 = 64*q*q*pi*a0**2/(3*np.sqrt(3)*hbar*c)
#recombination coefficient multiplicative constant term
A = sig0*c/np.sqrt(pi)*(q*q/(hbar*c))**3
#classical electron radius
r0 = q*q/(m*c*c)
#T0 in erg
T0 = 2.7255*kb
#1eV in erg
ev = 1.6022e-12
#constant factor in review
revA = 9.78*r0*r0*c
#2S decay rate
Lambda = 8.23
#Lyman-alpha photon energy
ly = 3*B/4
