# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 03:19:01 2023

@author: Dong Ha
"""

from functions import *
import matplotlib.pyplot as plt
import scipy.optimize as op
from scipy.integrate import odeint
import time

#%% fitting to obtain numerical solution for recomb rate

start = time.time()
#initial guess
a_guess = np.array([2e-13, 0.8])
#initialise values for recomb coefficient
z = np.linspace(700, 1700, 2000)
a_data = a_ex(z)

fit_a, cov_a = op.curve_fit(exp_fit, z, a_data, a_guess)

#@jit(nopython=True)
def X_diff(z,X):
    C1 = 1+Lambda*K(z)*(1-X)*ng(z)*eta
    C2 = 1+Lambda*K(z)*(1-X)*ng(z)*eta*(Lambda+b_over_a(z)*exp_fit(z, fit_a[0], fit_a[1]))
    rate_a = exp_fit(z, fit_a[0], fit_a[1])*X*X*eta*ng(z)
    rate_b = b_over_a(z)*exp_fit(z, fit_a[0], fit_a[1])*np.exp(-ly/Temp(z))*(1-X)
    prefactor = 1/((1+z)*Hubble(z))
    return prefactor*(rate_a-rate_b)*C1/C2

#numerical integration of dX/dz
t = np.linspace(1700.0, 800.0, 1000)
sol_X = odeint(X_diff, 1, t, tfirst=True)
X = sol_X[:,0]

end = time.time()
print("Elapsed time = %s" % (end - start))

plt.plot(t, X, label='excited X')
plt.plot(t, saha(t), label='saha')
plt.legend()
plt.show()
#%% dX/dz

def rate_b(z, X):
    return b_over_a(z)*exp_fit(z, fit_a[0], fit_a[1])*np.exp(-ly/Temp(z))*(1-X)
def X_by_z(z, X):
	alpha = exp_fit(z, fit_a[0], fit_a[1])
	C1 = 1+Lambda*K(z)*(1-X)*ng(z)*eta
	C2 = 1+Lambda*K(z)*(1-X)*ng(z)*eta*(Lambda+b_over_a(z)*alpha)
	pre = alpha/((1+z)*Hubble(z))
	inside = X*X*ng(z)*eta-b_over_a(z)*np.exp(-ly/Temp(z))
	#inside = X*X*ng(z)*eta
	return C1*inside*pre/C2

#numerical integration of dX/dz
t = np.linspace(700.0, 1600.0, 2000, dtype='float32')
sol_X = odeint(X_by_z, saha(700), t, tfirst=True)
X = sol_X[:,0]

#%%my X
trunc = np.linspace(800, 1200, 100)
review_fit = rev_ex(trunc)
s = saha(t)
#plt.plot(t, s, label='Saha ionisation fraction')
#plt.plot(trunc, review_fit, label='$\propto z^{12.75}$')
plt.plot(t, X, label='X')
#plt.yscale('log')
plt.grid()
plt.legend()
plt.show()

#%%

