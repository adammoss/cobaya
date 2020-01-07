import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

matplotlib.use('TkAgg')

import camb
from camb import model, initialpower
from camb.dark_energy import DarkEnergyPPF, DarkEnergyFluid, AxionEffectiveFluid

"""
max_z_early = 5700.0
min_z_early = 500.0
max_z_late = 30.0
min_z_late = 0.0
w_bbn = -1
w_early = [0.5]
w_late = [-1.0]
w_dark_ages = 0.5
"""

max_z_early = 10000.0
min_z_early = 500.0
max_z_late = 3.0
min_z_late = 0.0
w_bbn = -1
w_early = [0.333, 0, 0.333]
w_late = [-1.0]
w_dark_ages = 0.0

max_a_early = 1 / (1 + min_z_early)
min_a_early = 1 / (1 + max_z_early)
max_a_late = 1 / (1 + min_z_late)
min_a_late = 1 / (1 + max_z_late)

w_early_bins = len(w_early)
w_late_bins = len(w_late)

num_a_vals = 1000
a_vals = np.logspace(-5, 0, num_a_vals)
z_vals = 1 / a_vals - 1

def w(a):
    if a < min_a_early:
        return w_bbn
    elif max_a_early > a > min_a_early:
        if w_early_bins > 0:
            idx = int(
                w_early_bins * (np.log10(a) - np.log10(min_a_early)) / (np.log10(max_a_early) - np.log10(min_a_early)))
            return w_early[idx]
        else:
            return -1
    elif a < min_a_late:
        return w_dark_ages
    elif max_a_late > a > min_a_late:
        if w_late_bins > 0:
            idx = int(
                w_late_bins * (np.log10(a) - np.log10(min_a_late)) / (np.log10(max_a_late) - np.log10(min_a_late)))
            return w_late[idx]
        else:
            return -1
    else:
        if w_late_bins > 0:
            return w_late[w_late_bins - 1]
        else:
            return -1


# LCDM
pars = camb.CAMBparams()
pars.set_cosmology(H0=68.33, ombh2=0.02239, omch2=0.1177, mnu=0.06, omk=0, tau=0.068)
pars.InitPower.set_params(As=2.14e-9, ns=0.9687, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0)
results_lcdm = camb.get_results(pars)
powers_lcdm = results_lcdm.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL_lcdm = powers_lcdm['total']

# Axion Model
n = 3
ac = 10**(-3.696)
f = 0.058
wn = (n - 1) / (n + 1)
pars = camb.CAMBparams()
pars.set_cosmology(H0=71.6, ombh2=0.02258, omch2=0.1299, mnu=0.06, omk=0, tau=0.068)
pars.InitPower.set_params(As=2.177e-9, ns=0.9880, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0)
pars.DarkEnergy = AxionEffectiveFluid()
min_omega = 0.0
max_omega = 0.001
for i in range(100):
    trial_omega = (min_omega + max_omega) / 2
    pars.DarkEnergy.set_params(wn, trial_omega, ac)
    results_axion = camb.get_background(pars)
    if results_axion.get_Omega('de', 1 / ac - 1) > f:
        max_omega = trial_omega
    else:
        min_omega = trial_omega
    if abs(results_axion.get_Omega('de', 1 / ac - 1) - f) < 0.0001:
        break
print(results_axion.get_Omega('de', 1 / ac - 1))
results_axion = camb.get_results(pars)
powers_axion = results_axion.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL_axion = powers_axion['total']
axion_density = results_axion.get_dark_energy_rho_w(a_vals)[0]
axion_w_vals = [-1.0]
for i in range(num_a_vals - 1):
    a1 = a_vals[num_a_vals - i - 2]
    a2 = a_vals[num_a_vals - i - 1]
    rho1 = axion_density[num_a_vals - i - 2]
    rho2 = axion_density[num_a_vals - i - 1]
    axion_w_vals.append(-1.0 - np.log(rho2 / rho1) / np.log(a2 / a1) / 3)
axion_w_vals = np.array(axion_w_vals[::-1])

# PPF
w_vals = np.array([w(a) for a in a_vals])
pars.DarkEnergy = DarkEnergyPPF()
pars.DarkEnergy.set_w_a_table(a_vals, w_vals)
results_ppf = camb.get_results(pars)
rho_ppf, _ = results_ppf.get_dark_energy_rho_w(a_vals)
powers_ppf = results_ppf.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL_ppf = powers_ppf['total']

# PPF - axion w mimic
pars.DarkEnergy = DarkEnergyPPF()
pars.DarkEnergy.set_w_a_table(a_vals, axion_w_vals)
results_ppf_axion = camb.get_results(pars)
rho_ppf_axion, _ = results_ppf_axion.get_dark_energy_rho_w(a_vals)
powers_ppf_axion = results_ppf_axion.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL_ppf_axion = powers_ppf_axion['total']

# Fluid
pars.DarkEnergy = DarkEnergyFluid()
pars.DarkEnergy.set_w_a_table(a_vals, w_vals)
results_fluid = camb.get_results(pars)
rho_fluid, _ = results_fluid.get_dark_energy_rho_w(a_vals)
powers_fluid = results_fluid.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL_fluid = powers_fluid['total']

# Fluid - axion w mimic
pars.DarkEnergy = DarkEnergyFluid()
pars.DarkEnergy.set_w_a_table(a_vals, axion_w_vals)
results_fluid_axion = camb.get_results(pars)
rho_fluid_axion, _ = results_fluid_axion.get_dark_energy_rho_w(a_vals)
powers_fluid_axion = results_fluid_axion.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL_fluid_axion = powers_fluid_axion['total']

fig, ax = plt.subplots(2, 2, figsize=(12, 12))

ax[0, 0].plot(a_vals, results_lcdm.get_Omega('de', z_vals), color='k')
ax[0, 0].plot(a_vals, results_axion.get_Omega('de', z_vals), color='b')
ax[0, 0].plot(a_vals, results_ppf_axion.get_Omega('de', z_vals), color='r', ls=':')
ax[0, 0].plot(a_vals, results_ppf.get_Omega('de', z_vals), color='r')
ax[0, 0].plot(a_vals, results_fluid_axion.get_Omega('de', z_vals), color='g', ls='-.')
ax[0, 0].plot(a_vals, results_fluid.get_Omega('de', z_vals), color='g')
ax[0, 0].set_ylabel(r'$\rho/\rho_{tot}$')
ax[0, 0].set_xlabel('$a$')
ax[0, 0].set_xscale('log')

ax[0, 1].plot(a_vals, results_lcdm.get_dark_energy_rho_w(a_vals)[0], color='k')
ax[0, 1].plot(a_vals, results_axion.get_dark_energy_rho_w(a_vals)[0], color='b')
ax[0, 1].plot(a_vals, results_ppf_axion.get_dark_energy_rho_w(a_vals)[0], color='r', ls=':')
ax[0, 1].plot(a_vals, results_ppf.get_dark_energy_rho_w(a_vals)[0], color='r')
ax[0, 1].plot(a_vals, results_fluid_axion.get_dark_energy_rho_w(a_vals)[0], color='g')
ax[0, 1].plot(a_vals, results_fluid.get_dark_energy_rho_w(a_vals)[0], color='g', ls='-.')
ax[0, 1].set_ylabel(r'$\rho_{de}$')
ax[0, 1].set_xlabel('$a$')
ax[0, 1].set_xscale('log')
ax[0, 1].set_yscale('log')

ls = np.arange(totCL_ppf.shape[0])

ax[1, 0].plot(ls, totCL_axion[:, 0] - totCL_ppf_axion[:, 0], color='r')
ax[1, 0].plot(ls, totCL_axion[:, 0] - totCL_fluid_axion[:, 0], color='g')
ax[1, 0].set_xlim([2, 2000])
ax[1, 0].set_ylim([-100, 100])

ax[1, 1].plot(ls, totCL_lcdm[:, 0] - totCL_axion[:, 0], color='b')
ax[1, 1].plot(ls, totCL_lcdm[:, 0] - totCL_ppf[:, 0], color='r')
ax[1, 1].plot(ls, totCL_lcdm[:, 0] - totCL_fluid[:, 0], color='g')
ax[1, 0].set_xlim([2, 2000])
ax[1, 1].set_ylim([-100, 100])

plt.show()
