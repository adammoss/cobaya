import numpy as np
import tkinter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def omega_axion_a(a):
    return 2 * omega_axion / ((a / ac) ** (3 * (wn + 1)) + 1)


def omega_de_a(a):
    return (omega_lambda + omega_axion_a(a)) / (omega_lambda + omega_axion_a(a) + omega_m / (a/ac)**3)


def omega_m_a(a):
    return (omega_m / (a/ac)**3) / (omega_lambda + omega_axion_a(a) + omega_m / (a/ac) ** 3)


def omega_lambda_a(a):
    return omega_lambda / (omega_lambda + omega_axion_a(a) + omega_m / (a/ac) ** 3)


a = np.logspace(-5, 0, 100)

#omega_axion = 0.0
#omega_m = 0.99999999999
omega_axion = 0.05
omega_m = 0.94999999999

n = 3
ac = 10**(-3.74)
omega_lambda = 1 - omega_axion - omega_m
wn = (n - 1) / (n + 1)

plt.plot(a, omega_de_a(a), color='red')
plt.plot(a, omega_m_a(a), color='blue')
plt.plot(a, omega_lambda_a(a), color='green')

loga_min = -5.0
w_bins = 10
x = [-1, -1, -1, 0.4, 0.5, 0.5, 0.5, 0, -1, -1, -1]

w = lambda a: x[min(int(w_bins * np.log10(a) / loga_min), w_bins - 1)]

num_a_vals = 1000
a_vals = np.logspace(-5, 0, num_a_vals)
w_vals = np.array([w(a) for a in a_vals])

omm_test = 0.3
omm_test = omm_test
omde_test = 1.0 - omm_test
omm = []
omde = []
a = []
for i in range(num_a_vals - 1):
    a1 = a_vals[num_a_vals - i - 2]
    a2 = a_vals[num_a_vals - i - 1]
    w2 = w_vals[num_a_vals - i - 2]
    omm_test = omm_test * (a1 / a2) ** (-3)
    omde_test = omde_test * (a1 / a2) ** (-3 * (1 + w2))
    omm.append(omm_test / (omm_test + omde_test))
    omde.append(omde_test / (omm_test + omde_test))
    a.append(a2)
a = np.array(a)
omm = np.array(omm)
omde = np.array(omde)

plt.plot(a, omm, color='black')
plt.plot(a, omde, color='cyan')
plt.xscale('log')
plt.yscale('log')
plt.show()


