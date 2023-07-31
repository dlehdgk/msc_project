# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 09:48:26 2023

@author: Dong Ha
"""

from functions import *
import matplotlib.pyplot as plt
from scipy import optimize as op
from scipy.integrate import solve_ivp

hubble = 0.7
H0 = 100*hubble/3.0857e19
Ok = 0
Om = 0.3
Or = 2.5e-5/(hubble**2)
OL = 1-Om-Or
z = np.arange(800,1600, dtype = 'float32')

X = m_frac(z)

X_nu = X_num(z, Om*hubble**2, 0.023)
plt.plot(z, X, label = "Saha's equation")
#plt.plot(z, X_nu, label = 'excited states only')
plt.legend()
plt.grid()
plt.xlabel('redshift [z]')
plt.ylabel('Mass fraction [X]')
#plt.savefig('Images/mass_fraction.png', dpi=300)
plt.show()

#finding percentile of ionised H
ion_frac = 0.9

for i in range(len(z)):
    if X[i] >= ion_frac:
        print("fraction is:", X[i])
        print("at", z[i])
        break

"""
Shows the Neutral fraction is much greater than expected accounting for the varying
exponent factor found by numerical integration (valid range 800 ~ 1200) that normal Saha's equation =>
recombination and time of last scattering happens later (smaller z)
"""
#%%
#Thompson freeze out
Gam = Thomp(z, X)
Gam_nu = Thomp(z, X_nu)

H = Hubble(z, H0, Or, Om, Ok, OL)

plt.plot(z, 1/H, label = '1/H(z)')
plt.plot(z, 1/Gam, label = '1/$\Gamma_{Th}$(z) Saha')
plt.plot(z, 1/Gam_nu, label = '1/$\Gamma_{Th}$(z) Excited states')
plt.legend()
plt.grid()
plt.xlabel('redshift [z]')
plt.ylabel('time [s]')
plt.yscale('log')
#plt.savefig('Images/freeze-out.png', dpi=300)
plt.show()

#%%
#freeze-out z
def Saha1(z):
    X1 = m_frac(z)
    return 1/Thomp(z, X1) - 1/Hubble(z, H0, Or, Om, Ok, OL)

def Exc(z):
    X1 = X_num(z, Om*hubble**2, 0.023)
    return 1/Thomp(z, X1) - 1/Hubble(z, H0, Or, Om, Ok, OL)

freeze_saha = op.fsolve(Saha1, [1100])
print("Saha:", freeze_saha)

freeze_excited = op.fsolve(Exc, [900])
print("Excited", freeze_excited)

#plt.plot(z, Saha1(z), label = 'Saha')
plt.plot(z, Exc(z), label = 'Excited')
plt.grid()
plt.legend()
plt.xlabel('redshift [z]')
plt.ylabel('1/$\Gamma_{Th}$ - 1/H [s]')
#plt.savefig('Images/width-Hubble.png', dpi=300)
print (1/Thomp(freeze_excited, X_num(freeze_excited, Om*hubble**2, 0.023))/3600/24/365)


#%%
#optimise probability of last scattering

def func(z, Xe):
    alph = 9.78*(2.82e-15)**2*c*np.sqrt(B/(kb*T(z)))*np.log(B/(kb*T(z)))
    beta = n_gamma(z)*eta()/Saha(z, 0.023)
    H = Hubble(z, 100*hubble/3.0857e19, 2.5e-5, 0.3, 0, 0.7)
    inte = -alph*(beta*(1-Xe)-n_gamma(z)*eta()*Xe**2)/(H*(1+z))
    return inte

sol = solve_ivp(func, [600, 1400], [0])

z = sol.t
Xe = np.swapaxes(sol.y, 0, 1)

plt.plot(z, Xe)
plt.grid()
plt.ylabel('ionisation fraction [X]')
plt.xlabel('redshift [z]')
