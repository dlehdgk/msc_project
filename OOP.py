# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 07:52:08 2023

@author: Dong Ha
"""

import numpy as np
from scipy.special import zeta, polygamma, factorial

#constants
kb = 1.380649e-23
pi = np.pi
z3 = zeta(3)
r0 = 2.82e-15
#constants in eV
hc = 1.2398e-6
B = 13.6
me = 0.511e6

h = 6.62607015e-34
hbar = h/2/np.pi
c = 2.99792458e8
e = 1.602176634e-19
eps0 = 8.8541878128e-12
mp = 1.67262192369e-27
B = 13.6*e
G = 6.674e-11
alp_fs = e*e/(2*eps0*h*c)
st = 8*np.pi*(alp_fs*hbar/me/c)**2/3

#background
class BG():
    def __init__(self, start, end, num):
        self.z = np.linspace(start, end, num)
        #Temperature in eV
        self.__temp = 2.725*kb/e*(1+self.list)
    
    def n_gamma(self):
        return (self.__temp/hc)**3*16*pi*z3
    
    #Saha's mass fraction
    def Saha(self, eta):
        S = 4*np.sqrt(2/pi)*z3*eta*(self.__temp/me)**1.5*np.exp(B/self.__temp)
        return -1+np.sqrt(1+4*S)/(2*S)
    
    def Hubble(self, H0, Or = 2.5e-5, Om = 0.3, Ok = 0, Ol = 0.7):
        E2 = Or*(1+self.z)**4 + Om*(1+self.z)**3 + Ok*(1+self.z)**2 + Ol
        return H0*np.sqrt(E2)
    
    #Excited + non-equilibrium state
    def Xe(self, eta):
        def alph(T):
            return 9.78*r0**2*c*np.sqrt(B/T)*np.log(B/T)
        def func(z):
            Beta = self.n_gamma*eta*(self.Saha)**2/(1-self.Saha)
            
    
    