import numpy as np
from scipy.integrate import odeint, simps
from math import log10, pi
import os


# function to calculate omegarh2 using CMB temperature and N_eff
def omegarcalc(Neff):
    # H0/h in SI units
    hubconv = 3.24078E-18
    Tcmb = 2.72548
    ab = 7.56577E-16
    c = 299792458
    G = 6.67430E-11

    rhogamma = ab * pow(Tcmb, 4) / pow(c, 2)
    rhonu = Neff * 7. / 8. * pow(4. / 11., 4. / 3.) * rhogamma
    rhocritoh2 = 3. * pow(hubconv, 2) / (8. * pi * G)

    omegarh2 = (rhogamma + rhonu) / rhocritoh2
    return omegarh2


# repeated factor in EoM and GW density
def fac(x, ombh2, omch2, omrh2, omlh2):
    fac = pow((ombh2 + omch2) * x + omrh2 + omlh2 * pow(x, 4), 0.5)
    return fac


def eomsolver(ombh2, omch2, h, omgwh2=1.0E-6, n_T=3.0, N_eff=3.046, kmin=1E-6, kmax=0.5, k_piv=0.05, acc=1.0,
              scaleconstant=3650.0, cutoff=5.0, cache=False):
    # is in Hubble times (up to factor of h)
    tmin = 1E-12
    # tmax set below because of h dependence

    # Number of k and t samples
    N = 500 * acc  # k steps
    M = 1000 * acc  # t steps
    N = int(N)
    M = int(M)

    # hubble value (/h) in Mpc
    hubbleconv = 3.34E-4

    # calculated parameters
    omrh2 = omegarcalc(N_eff)
    oml = 1 - (omrh2 + ombh2 + omch2) / pow(h, 2)
    omlh2 = oml * pow(h, 2)
    tmax = 1.0 / h

    # initialise arrays
    avec = np.empty(M)
    D = np.empty((M, N))
    Ddot = np.empty((M, N))
    rho = np.empty((M, N))
    press = np.empty((M, N))
    tvec = np.logspace(log10(tmin), log10(tmax), M)
    rho_t = np.empty(M)
    pres_t = np.empty(M)

    # functions for integration - coupled Friedmann equation and EoM
    def F(x, a, k):

        factor = fac(x[0], ombh2, omch2, omrh2, omlh2)
        f1 = factor / x[0]
        f2 = x[2]
        f3 = -3 / pow(x[0], 2) * factor * x[2] - pow(k / (hubbleconv * x[0]), 2.0) * x[1]

        return f1, f2, f3

    # Check if we can use cached result (i.e k_min and k_max are in the cached range)
    if cache and os.path.isfile('kvec.npy'):
        kvec = np.load('kvec.npy')
        if kvec[0] > kmin or kvec[-1] < kmax:
            print('Recalculating as k out of range')
            cache = False

    if not cache or not os.path.isfile('kvec.npy'):

        kvec = np.logspace(log10(kmin), log10(kmax), N)

        for i in range(0, N):  # loop over k's

            # initial conditions on D(a) and D'(a)
            init = pow(omrh2, 0.25) * pow(2 * tmin, 0.5), 1.0, -pow(kvec[i] / hubbleconv, 2) / (3 * pow(omrh2, 0.5))

            # solve differential equation for each k
            sol = odeint(F, init, tvec, args=(kvec[i],))

            # put values into matrix
            for j in range(0, M):
                avec[j] = sol[j, 0]
                D[j][i] = sol[j, 1]
                Ddot[j][i] = sol[j, 2]

                # density and pressure of gws
                hfactor = fac(avec[j], ombh2, omch2, omrh2, omlh2)
                wback = (omrh2 - 3. * omlh2 * pow(avec[j], 4)) / (3. * pow(hfactor, 2))
                rho[j][i] = pow(Ddot[j][i], 2) / 8. + \
                            pow(kvec[i] * D[j][i] / hubbleconv / avec[j], 2) / 8. + \
                            hfactor * Ddot[j][i] * D[j][i] / pow(avec[j], 2)
                press[j][i] = -5. * pow(Ddot[j][i], 2) / 24. + \
                              7. * pow(kvec[i] * D[j][i] / hubbleconv / avec[j], 2) / 24. + \
                              0.5 * (1 + wback) * hfactor * Ddot[j][i] * D[j][i] / pow(avec[j], 2)

        np.save('kvec.npy', kvec)
        np.save('avec.npy', avec)
        np.save('rho.npy', rho)
        np.save('press.npy', press)

    else:

        avec = np.load('avec.npy')
        rho = np.load('rho.npy')
        press = np.load('press.npy')

    w = np.divide(press, rho)

    ###################################
    # Smooth away oscillations for large w
    for i in range(0, N):
        smooth = 0
        for j in range(0, M):
            if w[j][i] < -0.2:
                smooth += 1
            else:
                break
        # this is the 'critical time' above which EoS is smoothed to 1/3
        # large scale constant -> w=1/3 later
        tc = scaleconstant * tvec[smooth - 1] / kvec[i]
        for j in range(0, M):
            if tvec[j] > tc:
                w[j][i] = 1.0 / 3.0
                press[j][i] = 1.0 / 3.0 * rho[j][i]

    ################################
    # k-integration

    # shouldn't matter because normalisation is set via density
    A_T = 1.0

    ki = np.where((kvec >= kmin) & (kvec <= kmax))

    power_spectrum = A_T * np.power(kvec[ki] / k_piv, n_T)

    for i in range(0, M):
        rho_t[i] = simps(rho[i, ki] * power_spectrum, np.log(kvec[ki]))
        pres_t[i] = simps(press[i, ki] * power_spectrum, np.log(kvec[ki]))

    w_t = np.divide(pres_t, rho_t)

    # Get rid of very large peaks around transition
    for i in range(0, M):
        if w_t[i] > cutoff:
            w_t[i] = cutoff
            pres_t[i] = cutoff * rho_t[i]
        elif w_t[i] < -cutoff:
            w_t[i] = -cutoff
            pres_t[i] = -cutoff * rho_t[i]

    # trim values for a>1 and normalise density today
    avec, rho_t, w_t = camb_prep(avec, rho_t, w_t)
    cons = rho_t[-1] / omgwh2
    rho_t = rho_t / cons

    # calculate effective equation of state for GWs+cosmological constant
    weff = np.divide(w_t * rho_t - omlh2, rho_t + omlh2)
    # weff=w_t

    # return avec and weff
    return avec, weff, rho_t + omlh2


# function to slightly fiddle with avec dependent parameters so they can be used in camb as effective fluid
# problem is that my code goes slightly past a=1.0 so need to trim off last few data points
def camb_prep(avec, y1, y2):
    greater = np.where(avec > 1.0)
    avec = np.delete(avec, greater)
    y1 = np.delete(y1, greater)
    y2 = np.delete(y2, greater)
    avec = np.append(avec, 1.0)
    y1 = np.append(y1, y1[-1])
    y2 = np.append(y2, y2[-1])
    return avec, y1, y2
