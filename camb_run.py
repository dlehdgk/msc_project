# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:49:27 2023

@author: Dong Ha
"""

import matplotlib.pyplot as plt
import numpy as np
import camb
from camb import model, initialpower

#%%
pars=camb.CAMBparams()
pars.set_cosmology(70)
pars.InitPower.set_params(2e-9,0.965,0)
pars.set_for_lmax(2000)
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit = 'muK')


#%%
pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, As=2e-9, ns=0.95)
data= camb.get_background(pars)
z = np.linspace(1000, 1800, 300)
back_ev = data.get_background_redshift_evolution(z, ['x_e', 'visibility'], format='array')

plt.plot(z, back_ev[:,1])
plt.show()