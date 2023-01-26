import numpy as np
import pickle as pckl
import fctns
from numba import njit
from scipy.special import wofz
import lmfit as lm
import random as rand
from scipy.constants import Boltzmann, electron_volt, angstrom, elementary_charge, epsilon_0
import matplotlib.pyplot as plt

# epsilon r diamond
epsilon_r = 5.5
e_charge = elementary_charge

# diamond lattice parameters
unit_cell_length = 0.3567 #nano meters
carbon_dens = 8 / unit_cell_length ** 3

#conversion factors
Milli = 10 **(-3)
Nano = 10 ** (-9)
Mega = 10 **6

w0_FWHM = 13 * Milli  # 72 * Milli #72.43
omega = np.linspace(-6., 6., 5001)

pi = np.pi

epsilon_r = 5.5
el_fac = e_charge / (4 * pi * epsilon_0 * epsilon_r * Nano**2)


@njit
def lorentzjit(omega, omega0):
    x = 2 * (omega - omega0) / w0_FWHM
    return 1 / (1 + x ** 2)


@njit
def inhomlinejit(omega, points):
    out_arr = np.zeros(len(omega))
    k = 0
    for om in omega:
        s = 0
        for i in points:
            s = s + lorentzjit(om, i)
        out_arr[k] = s
        k = k + 1
    return out_arr


def V(x, x0, amp, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))
    return amp * np.real(wofz((x - x0 + 1j * gamma) / sigma / np.sqrt(2))) / sigma \
           / np.sqrt(2 * np.pi)

# funtion creating traps in in a pillar

def trap_gen_pillar(radius, height, ppm, **kwargs):

    box_volume = 4 * radius ** 2 * height
    rho_defects = ppm / 10 ** 6
    n_carbon = box_volume * carbon_dens
    n_defects = int(rho_defects * n_carbon)
    pillar_pos = []

    for i in range(n_defects):
        x = 2 * radius * np.random.rand() - radius
        y = 2 * radius * np.random.rand() - radius
        z = 2 * height * np.random.rand() - height
        if x**2 + y**2 <= radius**2:
            pillar_pos.append([x, y, z])

    return pillar_pos

def trap_gen_bulk(length, ppm, **kwargs):

    box_volume = length ** 3
    rho_defects = ppm / 10 ** 6
    n_carbon = box_volume * carbon_dens
    n_defects = int(rho_defects * n_carbon)
    return 2 * length * np.random.rand(n_defects, 3) - length

'''
function that produces the approximate FWHM, Gaussian and Lorentzian componenents, as well as the Fit results
of a Voigt fit of the inhomogeneously broadenend line
'''

def lw_monte_carlo_bulk(ppm, n_charges, length, **kwargs):
    '''
    :param ppm: trap density in parts per million
    :param n_charges: number of charges distributed on traps
    :param length: length of volume in which traps are distributed
    :return: FWHM, gaussian component, lorentzian component, fit result
    '''
    gmodel = lm.Model(V)
    gmodel.set_param_hint('amp', min=0)
    gmodel.set_param_hint('alpha', min=0)
    gmodel.set_param_hint('gamma', min=0)
    params = gmodel.make_params(x0=0., amp=1.1, alpha= .1, gamma=w0_FWHM / 2)

    positions = list(trap_gen_bulk(length, ppm))

    if n_charges % 2 == 0:
        nc = n_charges - 1
    else:
        nc = n_charges

    charge = np.array([1 for i in range(int((nc + 1) / 2))] + [-1 for i in range(int((nc - 1) / 2))]) # insure charge neutrality (excluding the vacancy)

    shifts = []
    realizations = 5000 # corresponds to 5000 differnt charge configurations

    for real in range(realizations): #loop over configurations
        pos_n = np.array(rand.sample(positions, nc))
        shift, _ = fctns.diff_rot(pos_n, charge)
        shifts.append(shift)

    line = inhomlinejit(omega, np.array(shifts)) # produce inhom. line from individual shifts
    line = line / max(line)
    result = gmodel.fit(line, params, x=omega)
    lwl = 2 * result.params['gamma'].value
    lwg = 2 * result.params['alpha'].value
    lw = 0.5346 * lwl + np.sqrt(0.2166 * lwl ** 2 + lwg ** 2)

    return lw, lwl, lwg, result # lw = FWHM, lwl = lorentzian component, lwg = gaussian component, result = full fit results

def main():
    pass

if __name__ == '__main__':
    main()