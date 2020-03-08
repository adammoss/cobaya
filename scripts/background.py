import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

matplotlib.use('TkAgg')

sys.path.insert(0, '/Users/ppzam/work/code/cosmo/code/CAMB/')

import camb
from camb import model, initialpower
from camb.dark_energy import DarkEnergyPPF, DarkEnergyFluid, AxionEffectiveFluid

fig, ax = plt.subplots(2, 2, figsize=(12, 12))

nodes = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0])
num_a_vals = len(nodes)

a_vals = np.logspace(-5, 0, num_a_vals)
pars = camb.CAMBparams()
pars.set_cosmology(H0=68.33, ombh2=0.02239, omch2=0.1177, mnu=0.06, omk=0, tau=0.068)
results = camb.get_background(pars)
densities = results.get_background_densities(a_vals)
total_de_density = a_vals**(-4) * (densities['de'] + densities['tot'] * nodes)

w_vals = [-1.0]
for i in range(num_a_vals - 1):
    a1 = a_vals[num_a_vals - i - 2]
    a2 = a_vals[num_a_vals - i - 1]
    rho1 = total_de_density[num_a_vals - i - 2]
    rho2 = total_de_density[num_a_vals - i - 1]
    w_vals.append(-1.0 - np.log(rho2 / rho1) / np.log(a2 / a1) / 3)
w_vals = np.array(w_vals[::-1])

print(w_vals)


ax[0, 0].plot(a_vals, a_vals**(-4) * densities['de'], color='k')
ax[0, 0].plot(a_vals, total_de_density, color='r')
ax[0, 0].set_xscale('log')
ax[0, 0].set_yscale('log')

ax[0, 1].plot(a_vals, w_vals, color='k')
ax[0, 1].set_xscale('log')

print(densities['de'])
print(total_de_density)
cs = CubicSpline(np.log(a_vals), np.log(total_de_density))
a_vals = np.logspace(-5, 0, 100)
w_vals = - cs(np.log(a_vals), 1) / 3 - 1

ax[1, 0].plot(a_vals, w_vals, color='k')
ax[1, 0].set_xscale('log')

print(a_vals)
print(w_vals)

plt.show()



